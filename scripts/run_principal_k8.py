from __future__ import annotations

# ruff: noqa: E402
import argparse
import copy
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_phase2_physical_baselines import (
    make_plots as make_phase2_plots,
)
from scripts.run_phase2_physical_baselines import (
    run_phase2_physical_baselines,
)
from scripts.run_phase2_physical_baselines import (
    write_records as write_phase2_records,
)
from scripts.run_phase2_physical_baselines import (
    write_summary as write_phase2_summary,
)
from scripts.run_phase2_physical_baselines import (
    write_summary_csv as write_phase2_summary_csv,
)
from scripts.run_phase3_candidate_retrieval import (
    BinaryMassBank,
    build_method_coordinates,
    compute_candidate_matches,
    fit_feature_scaler,
    generate_waveforms,
    prepare_matching,
    query_rows_for_k,
    raw_feature_matrix_from_masses,
    raw_feature_matrix_from_metadata,
    summarize_query_rows,
    write_query_results,
    write_runs,
    write_summary_json,
)
from scripts.run_phase3_candidate_retrieval import (
    make_plots as make_phase3_plots,
)
from scripts.run_phase3_candidate_retrieval import (
    write_summary_csv as write_phase3_summary_csv,
)
from scripts.run_robustness_experiments import experiment_config, file_digest, run_one
from scripts.run_scaling_validation import make_plots as make_phase1_plots
from scripts.run_scaling_validation import write_records, write_summary, write_summary_csv

from gw_mismatch_learning.config import load_config
from gw_mismatch_learning.datasets.gw import load_distance_matrix_dataset
from gw_mismatch_learning.models.encoders import MLPEncoder
from gw_mismatch_learning.models.metric_learning import encode_array
from gw_mismatch_learning.retrieval.index import SklearnNeighborIndex
from gw_mismatch_learning.utils.seeds import set_seed

FROZEN = [
    Path("outputs/scaling_validation"),
    Path("outputs/phase2_physical_baselines"),
    Path("outputs/phase3_candidate_retrieval"),
    Path("docs/paper_validation/checksums.sha256"),
    Path("docs/paper_validation/freeze_manifest.json"),
    Path("docs/paper_validation/mismatch_supervised_representation_framework.tex"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/principal_k8.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        raise ValueError("Choose --resume or --overwrite, not both")
    cfg = load_config(args.config)
    validate_inputs(cfg)
    if args.dry_run:
        print("Validated principal-k8 inputs; no files written")
        return
    output = Path(cfg["output_dir"])
    if output.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(f"Output exists: {output}; use --resume or --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    before = manifest(FROZEN)
    dump_json(output / "frozen_manifest_before.json", before)
    run_phase1(cfg, output / "phase_i_scaling", args.resume, args.overwrite)
    run_phase2(cfg, output / "phase_ii_physical_baselines", args.resume, args.overwrite)
    run_phase3(cfg, output / "phase_iii_candidate_retrieval", args.resume, args.overwrite)
    make_rmse_figure()
    after = manifest(FROZEN)
    dump_json(output / "frozen_manifest_after.json", after)
    if before != after:
        raise RuntimeError("Frozen k=4 artifacts changed")


def validate_inputs(cfg: dict[str, Any]) -> None:
    if int(cfg["seed"]) != 1234 or int(cfg["latent_dim"]) != 8:
        raise ValueError("Principal configuration must use seed=1234 and latent_dim=8")
    for size in cfg["bank_sizes"]:
        path = Path(str(cfg["cache_template"]).format(bank_size=size))
        if not path.is_file():
            raise FileNotFoundError(path)
    for key in ["latent_sweep_reference", "phase3_config"]:
        if not Path(cfg[key]).is_file():
            raise FileNotFoundError(cfg[key])


def run_phase1(cfg: dict[str, Any], output: Path, resume: bool, overwrite: bool) -> None:
    base = load_config(cfg["base_config"])
    records = []
    for size in cfg["bank_sizes"]:
        spec = {
            "bank_size": int(size),
            "latent_dim": 8,
            "seed": 1234,
            "cache_path": str(cfg["cache_template"]).format(bank_size=size),
        }
        run_dir = output / f"bank_size_{size}"
        metrics_path = run_dir / "metrics.json"
        full = experiment_config(base, spec)
        if metrics_path.exists() and resume:
            record = json.loads(metrics_path.read_text())
            validate_phase1_record(record, spec)
            records.append(record)
            continue
        if metrics_path.exists() and not overwrite:
            raise FileExistsError(metrics_path)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_text = yaml.safe_dump(full, sort_keys=False)
        (run_dir / "config.yaml").write_text(config_text)
        if int(size) == 8192:
            record = json.loads(Path(cfg["latent_sweep_reference"]).read_text())
            validate_sweep_reference(record, spec)
            record = copy.deepcopy(record)
            record["reference_reused"] = str(cfg["latent_sweep_reference"])
        else:
            record = run_one(full, spec)
        record.update(
            {
                "status": "completed",
                "run_id": f"bank_size_{size}",
                "bank_size": int(size),
                "latent_dim": 8,
                "seed": 1234,
                "git_commit": git_commit(),
                "config_sha256": sha256_text(config_text),
            }
        )
        dump_json(metrics_path, record)
        records.append(record)
    write_records(output / "runs.csv", records)
    write_summary(output / "summary.json", records)
    write_summary_csv(output / "summary.csv", records)
    make_phase1_plots(output, records)


def validate_phase1_record(record: dict[str, Any], spec: dict[str, Any]) -> None:
    if record.get("status") != "completed" or record.get("cache_reused") is not True:
        raise RuntimeError(f"Invalid completed record for {spec}")
    for key in ["bank_size", "latent_dim", "seed", "cache_path"]:
        if str(record.get(key)) != str(spec[key]):
            raise RuntimeError(f"Phase I resume mismatch: {key}")
    if record.get("cache_sha256") != file_digest(Path(spec["cache_path"])):
        raise RuntimeError("Phase I cache hash mismatch")


def validate_sweep_reference(record: dict[str, Any], spec: dict[str, Any]) -> None:
    validate_phase1_record(record, spec)
    frozen = (
        json.loads(Path("outputs/scaling_validation/bank_size_8192/config.yaml").read_text())
        if False
        else None
    )
    del frozen


def run_phase2(cfg: dict[str, Any], output: Path, resume: bool, overwrite: bool) -> None:
    summary_path = output / "summary.json"
    if summary_path.exists() and resume:
        payload = json.loads(summary_path.read_text())
        if payload.get("num_completed") != len(payload.get("records", [])):
            raise RuntimeError("Invalid Phase II resume output")
        return
    if summary_path.exists() and not overwrite:
        raise FileExistsError(summary_path)
    phase2 = load_config(cfg["phase2_config"])
    phase2["output_dir"] = str(output)
    phase2["phase1_output_dir"] = str(Path(cfg["output_dir"]) / "phase_i_scaling")
    records = run_phase2_physical_baselines(phase2)
    frozen = json.loads((Path(cfg["phase2_frozen_output"]) / "summary.json").read_text())
    frozen_physical = {
        (r["bank_size"], r["coordinate_system"]): r
        for r in frozen["records"]
        if r["coordinate_system"] != "learned_latent"
    }
    for record in records:
        if record["coordinate_system"] != "learned_latent":
            if record != frozen_physical[(record["bank_size"], record["coordinate_system"])]:
                raise RuntimeError("Physical baseline changed")
    output.mkdir(parents=True, exist_ok=True)
    write_phase2_records(output / "runs.csv", records)
    write_phase2_summary(output / "summary.json", phase2, records)
    write_phase2_summary_csv(output / "summary.csv", records)
    make_phase2_plots(output, records)


def run_phase3(cfg: dict[str, Any], output: Path, resume: bool, overwrite: bool) -> None:
    summary_path = output / "summary.json"
    if summary_path.exists() and resume:
        payload = json.loads(summary_path.read_text())
        if len(payload.get("summary_records", [])) != 27:
            raise RuntimeError("Invalid Phase III resume output")
        return
    if summary_path.exists() and not overwrite:
        raise FileExistsError(summary_path)
    p3 = load_config(cfg["phase3_config"])
    base = load_config(cfg["base_config"])
    base["model"]["embedding_dim"] = 8
    cache = Path(p3["bank_cache_path"])
    dataset = load_distance_matrix_dataset(cache)
    reference = load_phase3_reference(Path(cfg["phase3_frozen_output"]) / "query_results.csv")
    phase1 = json.loads(
        (Path(cfg["output_dir"]) / "phase_i_scaling/bank_size_8192/metrics.json").read_text()
    )
    set_seed(1234)
    pairs = __import__(
        "gw_mismatch_learning.datasets.pairs", fromlist=["sample_pairs"]
    ).sample_pairs(dataset.features, dataset.distance, 8192 * 32, seed=1234)
    pair_ds = __import__(
        "gw_mismatch_learning.datasets.distance_regression", fromlist=["DistanceRegressionDataset"]
    ).DistanceRegressionDataset(pairs)
    encoder = MLPEncoder(5, 8, (32, 16))
    history = __import__(
        "gw_mismatch_learning.models.metric_learning", fromlist=["train_distance_regression"]
    ).train_distance_regression(encoder, pair_ds, epochs=10, batch_size=128, learning_rate=0.001)
    if not np.isclose(history.losses[-1], phase1["final_train_loss"], rtol=1e-6):
        raise RuntimeError("Phase III k8 training failed to reproduce Phase I")
    raw = raw_feature_matrix_from_metadata(dataset.metadata)
    scaler = fit_feature_scaler(raw)
    features = scaler.transform(raw)
    bank_z = encode_array(encoder, features)
    query_bank = BinaryMassBank(reference["query_m1"], reference["query_m2"])
    query_features = scaler.transform(
        raw_feature_matrix_from_masses(query_bank.mass_1, query_bank.mass_2)
    )
    start = time.perf_counter()
    query_z = encode_array(encoder, query_features)
    embedding_time = time.perf_counter() - start
    methods = build_method_coordinates(
        {"methods": [{"name": "learned_latent", "label": "Learned latent k=8", "type": "learned"}]},
        features,
        query_features,
        bank_z,
        query_z,
    )
    bank_waveforms = generate_waveforms(
        BinaryMassBank(
            np.asarray(dataset.metadata["mass_1"]), np.asarray(dataset.metadata["mass_2"])
        ),
        dataset.metadata,
    )
    query_waveforms = generate_waveforms(query_bank, dataset.metadata)
    prepared_bank, prepared_queries, psd, match_fn = prepare_matching(
        bank_waveforms, query_waveforms, dataset.metadata
    )
    method = methods[0]
    start = time.perf_counter()
    index = SklearnNeighborIndex(method["bank"])
    _, nearest = index.query(method["query"], top_k=max(p3["candidate_k"]))
    retrieval = time.perf_counter() - start
    rows = []
    summaries = []
    candidate_total = 0.0
    for top_k in p3["candidate_k"]:
        indices = nearest[:, :top_k]
        start = time.perf_counter()
        matches = compute_candidate_matches(
            prepared_queries,
            prepared_bank,
            indices,
            psd=psd,
            match_fn=match_fn,
            f_lower=float(dataset.metadata["f_lower"]),
        )
        candidate_time = time.perf_counter() - start
        candidate_total += candidate_time
        current = query_rows_for_k(
            method,
            top_k,
            indices,
            matches,
            reference["best_index"],
            reference["best_match"],
            query_bank,
        )
        rows.extend(current)
        summary = summarize_query_rows(method, top_k, current, 8192)
        summary.update(
            {
                "latent_retrieval_time_seconds": retrieval,
                "candidate_matching_time_seconds": candidate_time,
                "total_candidate_pipeline_time_seconds": embedding_time
                + retrieval
                + candidate_time,
            }
        )
        summaries.append(summary)
    frozen_summary = json.loads((Path(cfg["phase3_frozen_output"]) / "summary.json").read_text())
    physical = [r for r in frozen_summary["summary_records"] if r["method"] != "learned_latent"]
    summaries.extend(physical)
    frozen_rows = read_csv(Path(cfg["phase3_frozen_output"]) / "query_results.csv")
    rows.extend([r for r in frozen_rows if r["method"] != "learned_latent"])
    run_record = {
        "status": "completed",
        "method": "learned_latent",
        "latent_dim": 8,
        "seed": 1234,
        "cache_path": str(cache),
        "cache_sha256": file_digest(cache),
        "cache_reused": True,
        "training_history": history.losses,
        "parameter_count": sum(p.numel() for p in encoder.parameters()),
        "latent_embedding_time_seconds": embedding_time,
        "latent_retrieval_time_seconds": retrieval,
        "candidate_matching_time_seconds_all_k": candidate_total,
        "git_commit": git_commit(),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_query_results(output / "query_results.csv", rows)
    write_phase3_summary_csv(output / "summary.csv", summaries)
    write_runs(output / "runs.csv", [run_record])
    metadata = {
        **{k: v for k, v in frozen_summary.items() if k != "summary_records"},
        "objective": "principal_k8_phase3",
        "latent_dim": 8,
        "seed": 1234,
        "exhaustive_reference_reused": str(Path(cfg["phase3_frozen_output"]) / "query_results.csv"),
    }
    write_summary_json(output / "summary.json", metadata, summaries)
    make_phase3_plots(output, summaries)


def load_phase3_reference(path: Path) -> dict[str, np.ndarray]:
    rows = [
        r for r in read_csv(path) if r["method"] == "learned_latent" and int(r["candidate_k"]) == 1
    ]
    rows.sort(key=lambda r: int(r["query_index"]))
    return {
        "query_m1": np.asarray([float(r["query_m1"]) for r in rows], dtype=np.float32),
        "query_m2": np.asarray([float(r["query_m2"]) for r in rows], dtype=np.float32),
        "best_index": np.asarray([int(r["exhaustive_best_index"]) for r in rows]),
        "best_match": np.asarray(
            [float(r["exhaustive_best_match"]) for r in rows], dtype=np.float32
        ),
    }


def make_rmse_figure() -> None:
    import matplotlib.pyplot as plt

    rows = read_csv(Path("outputs/latent_dimension_sweep/aggregate.csv"))
    rows = sorted(
        [r for r in rows if r["metric"] == "distance_rmse"], key=lambda r: int(r["latent_dim"])
    )
    x = [int(r["latent_dim"]) for r in rows]
    mean = [float(r["mean"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(
        x,
        mean,
        yerr=[
            [m - float(r["ci95_low"]) for m, r in zip(mean, rows, strict=True)],
            [float(r["ci95_high"]) - m for m, r in zip(mean, rows, strict=True)],
        ],
        marker="o",
        capsize=3,
    )
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Distance RMSE")
    ax.grid(True, alpha=0.3)
    fig.savefig(
        "outputs/latent_dimension_sweep/figures/distance_rmse_multiseed_ci.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def manifest(paths: list[Path]) -> dict[str, str]:
    files = []
    for path in paths:
        files.extend([path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()])
    return {str(p): file_digest(p) for p in sorted(files)}


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


if __name__ == "__main__":
    main()
