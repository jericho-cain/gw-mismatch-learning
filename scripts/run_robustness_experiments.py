from __future__ import annotations

# ruff: noqa: E402 -- executable scripts add the repository roots before local imports.
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
from scipy.stats import t as student_t

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts.run_scaling_validation import evaluate_encoder

from gw_mismatch_learning.config import load_config
from gw_mismatch_learning.datasets.distance_regression import DistanceRegressionDataset
from gw_mismatch_learning.datasets.gw import load_distance_matrix_dataset
from gw_mismatch_learning.datasets.pairs import sample_pairs
from gw_mismatch_learning.models.encoders import MLPEncoder
from gw_mismatch_learning.models.metric_learning import train_distance_regression
from gw_mismatch_learning.utils.seeds import set_seed

METRICS = [
    "final_train_loss",
    "pearson",
    "spearman",
    "distance_mae",
    "distance_rmse",
    "learned_recall_at_5",
    "learned_recall_at_10",
    "learned_recall_at_20",
    "learned_recovered_best_match_at_5",
    "learned_recovered_best_match_at_10",
    "learned_recovered_best_match_at_20",
    "training_time_seconds",
    "evaluation_time_seconds",
    "parameter_count",
]
FROZEN_PATHS = [
    Path("outputs/scaling_validation"),
    Path("docs/paper_validation/freeze_manifest.json"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    specs = build_specs(cfg)
    validate_specs(specs)
    estimate = estimate_runtime(cfg, len(specs))
    print(f"Planned runs: {len(specs)}; cache-based estimate: {estimate:.1f} seconds")
    if args.dry_run:
        return
    run_experiment(cfg, specs, overwrite=args.overwrite, resume=args.resume)


def build_specs(cfg: dict[str, Any]) -> list[dict[str, int | str]]:
    objective = str(cfg["objective"])
    seeds = [int(seed) for seed in cfg["seeds"]]
    if objective == "latent_dimension_sweep":
        return [
            {
                "bank_size": int(cfg["bank_size"]),
                "latent_dim": int(dim),
                "seed": seed,
                "cache_path": str(cfg["cache_path"]),
            }
            for dim in cfg["latent_dimensions"]
            for seed in seeds
        ]
    if objective == "seed_robustness":
        return [
            {
                "bank_size": int(size),
                "latent_dim": 4,
                "seed": seed,
                "cache_path": str(cfg["cache_template"]).format(bank_size=int(size)),
            }
            for size in cfg["bank_sizes"]
            for seed in seeds
        ]
    raise ValueError(f"Unknown objective: {objective}")


def validate_specs(specs: list[dict[str, Any]]) -> None:
    keys = [(x["bank_size"], x["latent_dim"], x["seed"]) for x in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate run specification")
    missing = sorted({str(x["cache_path"]) for x in specs if not Path(x["cache_path"]).is_file()})
    if missing:
        raise FileNotFoundError("Required frozen caches are missing: " + ", ".join(missing))


def experiment_config(base: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["seed"] = int(spec["seed"])
    cfg["gw_data"]["run_seed"] = int(spec["seed"])
    cfg["gw_data"]["num_waveforms"] = int(spec["bank_size"])
    cfg["gw_data"]["cache_path"] = str(spec["cache_path"])
    cfg["gw_data"]["overwrite"] = False
    cfg["pairs"]["num_pairs"] = max(int(cfg["pairs"]["num_pairs"]), int(spec["bank_size"]) * 32)
    cfg["model"]["embedding_dim"] = int(spec["latent_dim"])
    cfg["outputs"] = {"save_metrics": False, "save_plots": False}
    return cfg


def run_experiment(
    cfg: dict[str, Any],
    specs: list[dict[str, Any]],
    overwrite: bool,
    resume: bool = False,
) -> None:
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    before = frozen_manifest()
    write_json(output / "frozen_manifest_before.json", before, refuse=False)
    base = load_config(cfg["base_config"])
    records = []
    for spec in specs:
        run_id = f"n{spec['bank_size']}_k{spec['latent_dim']}_seed{spec['seed']}"
        run_dir = output / "runs" / run_id
        metrics_path = run_dir / "metrics.json"
        full = experiment_config(base, spec)
        if run_dir.exists() and resume:
            valid, reason, existing = validate_existing_run(run_dir, full, spec)
            if valid:
                records.append(existing)
                print(f"Skipping validated completed run {run_id}")
                continue
            raise RuntimeError(
                f"Existing run {run_dir} is not valid ({reason}); it was left untouched. "
                "Inspect it or pass --overwrite explicitly."
            )
        if metrics_path.exists():
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            if existing.get("status") == "completed" and not overwrite:
                raise FileExistsError(
                    f"Completed run exists: {run_dir}; pass --overwrite explicitly"
                )
        run_dir.mkdir(parents=True, exist_ok=True)
        config_text = yaml.safe_dump(full, sort_keys=False)
        (run_dir / "config.yaml").write_text(config_text, encoding="utf-8")
        try:
            record = run_one(full, spec)
            record["status"] = "completed"
        except Exception as exc:  # noqa: BLE001
            record = {
                **spec,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        record.update(
            {"run_id": run_id, "git_commit": git_commit(), "seed_policy": "shared_global_seed"}
        )
        record["config_sha256"] = digest_bytes(config_text.encode("utf-8"))
        record["warnings"] = [
            "Frozen Phase I has no validation split; validation_loss is unavailable."
        ]
        record["validation_loss"] = None
        write_json(metrics_path, record, refuse=False)
        records.append(record)
    require_complete(records, cfg)
    aggregates = aggregate_records(records, group_key(cfg))
    write_csv(output / "runs.csv", records)
    write_json(
        output / "summary.json",
        {"objective": cfg["objective"], "records": records, "aggregates": aggregates},
        refuse=False,
    )
    write_csv(output / "aggregate.csv", aggregates)
    make_plots(output, aggregates, group_key(cfg), cfg["objective"])
    after = frozen_manifest()
    write_json(output / "frozen_manifest_after.json", after, refuse=False)
    if before != after:
        raise RuntimeError("Frozen-output manifest changed during robustness experiment")


def validate_existing_run(
    run_dir: Path,
    expected_config: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    config_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.json"
    if not config_path.is_file() or not metrics_path.is_file():
        return False, "missing config.yaml or metrics.json", {}
    expected_text = yaml.safe_dump(expected_config, sort_keys=False)
    actual_text = config_path.read_text(encoding="utf-8")
    expected_hash = digest_bytes(expected_text.encode("utf-8"))
    actual_hash = digest_bytes(actual_text.encode("utf-8"))
    if actual_hash != expected_hash:
        return False, "configuration hash mismatch", {}
    record = json.loads(metrics_path.read_text(encoding="utf-8"))
    if record.get("status") != "completed":
        return False, "completion marker is not completed", record
    for key in ("bank_size", "latent_dim", "seed", "cache_path"):
        if str(record.get(key)) != str(spec[key]):
            return False, f"requested field mismatch: {key}", record
    recorded_config_hash = record.get("config_sha256")
    if recorded_config_hash is not None and recorded_config_hash != actual_hash:
        return False, "recorded configuration hash mismatch", record
    cache_path = Path(spec["cache_path"])
    if not cache_path.is_file():
        return False, "source cache is missing", record
    if record.get("cache_sha256") != file_digest(cache_path):
        return False, "source cache hash mismatch", record
    if record.get("cache_reused") is not True:
        return False, "cache reuse marker is not true", record
    if not record.get("training_history"):
        return False, "training history is missing", record
    record.setdefault("config_sha256", actual_hash)
    return True, "valid", record


def run_one(cfg: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    cache = Path(spec["cache_path"])
    before = file_digest(cache)
    dataset = load_distance_matrix_dataset(cache)
    set_seed(int(spec["seed"]))
    pairs = sample_pairs(
        dataset.features, dataset.distance, int(cfg["pairs"]["num_pairs"]), seed=int(spec["seed"])
    )
    encoder = MLPEncoder(
        input_dim=int(cfg["model"]["input_dim"]),
        embedding_dim=int(spec["latent_dim"]),
        hidden_dims=tuple(cfg["model"]["hidden_dims"]),
    )
    start = time.perf_counter()
    history = train_distance_regression(
        encoder,
        DistanceRegressionDataset(pairs),
        epochs=int(cfg["training"]["epochs"]),
        batch_size=int(cfg["training"]["batch_size"]),
        learning_rate=float(cfg["training"]["learning_rate"]),
    )
    training_time = time.perf_counter() - start
    start = time.perf_counter()
    metrics = evaluate_encoder(cfg, dataset, encoder)
    evaluation_time = time.perf_counter() - start
    if before != file_digest(cache):
        raise RuntimeError(f"Frozen cache changed: {cache}")
    return {
        **spec,
        **metrics,
        "cache_reused": True,
        "cache_sha256": before,
        "pair_sampling_seed": int(spec["seed"]),
        "model_initialization_seed": int(spec["seed"]),
        "data_loader_seed": int(spec["seed"]),
        "train_validation_split_seed": None,
        "training_history": history.losses,
        "final_train_loss": history.losses[-1],
        "training_time_seconds": training_time,
        "evaluation_time_seconds": evaluation_time,
        "parameter_count": sum(p.numel() for p in encoder.parameters()),
    }


def aggregate_records(records: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    if any(r.get("status") != "completed" for r in records):
        raise ValueError("Refusing to aggregate missing or failed runs")
    rows = []
    for value in sorted({int(r[group]) for r in records}):
        selected = [r for r in records if int(r[group]) == value]
        if len(selected) != 5:
            raise ValueError(f"Expected exactly five runs for {group}={value}, got {len(selected)}")
        for metric in METRICS:
            values = [float(r[metric]) for r in selected]
            mean, std, low, high = student_t_summary(values)
            rows.append(
                {
                    group: value,
                    "metric": metric,
                    "n": 5,
                    "degrees_of_freedom": 4,
                    "mean": mean,
                    "sample_std": std,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return rows


def student_t_summary(values: list[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        raise ValueError("Student-t interval requires at least two observations")
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    margin = float(student_t.ppf(0.975, df=array.size - 1) * std / np.sqrt(array.size))
    return mean, std, mean - margin, mean + margin


def require_complete(records: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    expected = len(cfg["seeds"]) * len(cfg.get("latent_dimensions", cfg.get("bank_sizes", [])))
    if len(records) != expected or any(r.get("status") != "completed" for r in records):
        raise RuntimeError("Not all requested runs completed; no aggregate results were produced")
    if cfg["objective"] == "seed_robustness":
        verify_frozen_seed_reproduction(records)


def verify_frozen_seed_reproduction(records: list[dict[str, Any]]) -> None:
    reproducibility_metrics = [
        metric
        for metric in METRICS
        if metric not in {"training_time_seconds", "evaluation_time_seconds"}
    ]
    for record in records:
        if int(record["seed"]) != 1234:
            continue
        frozen_path = Path(
            f"outputs/scaling_validation/bank_size_{record['bank_size']}/metrics.json"
        )
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        for metric in reproducibility_metrics:
            if metric == "parameter_count":
                continue
            if not np.isclose(float(record[metric]), float(frozen[metric]), rtol=1e-6, atol=1e-8):
                raise RuntimeError(
                    f"Frozen-seed reproduction failed for bank {record['bank_size']}: {metric}"
                )


def group_key(cfg: dict[str, Any]) -> str:
    return "latent_dim" if cfg["objective"] == "latent_dimension_sweep" else "bank_size"


def estimate_runtime(cfg: dict[str, Any], run_count: int) -> float:
    frozen = json.loads(Path("outputs/scaling_validation/bank_size_8192/metrics.json").read_text())
    per_run = float(frozen["training_time_seconds"]) + float(frozen["evaluation_time_seconds"])
    if cfg["objective"] == "latent_dimension_sweep":
        return per_run * run_count
    total = 0.0
    for size in cfg["bank_sizes"]:
        row = json.loads(
            Path(f"outputs/scaling_validation/bank_size_{size}/metrics.json").read_text()
        )
        total += (
            float(row["training_time_seconds"]) + float(row["evaluation_time_seconds"])
        ) * len(cfg["seeds"])
    return total


def make_plots(output: Path, rows: list[dict[str, Any]], group: str, objective: str) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output / "figures"
    plot_dir.mkdir(exist_ok=True)
    specs = [
        (
            ["learned_recall_at_5", "learned_recall_at_10", "learned_recall_at_20"],
            "Recall",
            "learned_recall_multiseed_ci.png",
        ),
        (
            [
                "learned_recovered_best_match_at_5",
                "learned_recovered_best_match_at_10",
                "learned_recovered_best_match_at_20",
            ],
            "Best-match recovery",
            "learned_best_match_recovery_multiseed_ci.png",
        ),
        (["pearson", "spearman"], "Correlation", "distance_correlation_multiseed_ci.png"),
        (
            ["training_time_seconds", "evaluation_time_seconds"],
            "Runtime (s)",
            "runtime_multiseed_ci.png",
        ),
    ]
    for metrics, ylabel, filename in specs:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for metric in metrics:
            selected = sorted([r for r in rows if r["metric"] == metric], key=lambda r: r[group])
            x = [r[group] for r in selected]
            y = [r["mean"] for r in selected]
            ax.errorbar(
                x,
                y,
                yerr=[
                    [r["mean"] - r["ci95_low"] for r in selected],
                    [r["ci95_high"] - r["mean"] for r in selected],
                ],
                marker="o",
                capsize=3,
                label=metric.replace("learned_", "").replace("_", " "),
            )
        ax.set_xlabel("Latent dimension" if group == "latent_dim" else "Waveform bank size")
        if group == "bank_size":
            ax.set_xscale("log", base=2)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.savefig(plot_dir / filename, dpi=250, bbox_inches="tight")
        plt.close(fig)
    if group == "latent_dim":
        plot_dual_axis_dimension_metrics(rows, plot_dir)
        plot_dimension_saturation_comparisons(rows, plot_dir / "comparison")


def plot_dual_axis_dimension_metrics(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    import matplotlib.pyplot as plt

    pearson = sorted(
        [row for row in rows if row["metric"] == "pearson"],
        key=lambda row: row["latent_dim"],
    )
    recall = sorted(
        [row for row in rows if row["metric"] == "learned_recall_at_10"],
        key=lambda row: row["latent_dim"],
    )
    if [row["latent_dim"] for row in pearson] != [row["latent_dim"] for row in recall]:
        raise ValueError("Pearson and Recall@10 latent-dimension rows do not align")

    dimensions = [row["latent_dim"] for row in pearson]
    fig, left = plt.subplots(figsize=(6.8, 4.2))
    right = left.twinx()
    left.errorbar(
        dimensions,
        [row["mean"] for row in pearson],
        yerr=[
            [row["mean"] - row["ci95_low"] for row in pearson],
            [row["ci95_high"] - row["mean"] for row in pearson],
        ],
        color="tab:blue",
        marker="o",
        capsize=3,
        label="Pearson correlation",
    )
    right.errorbar(
        dimensions,
        [row["mean"] for row in recall],
        yerr=[
            [row["mean"] - row["ci95_low"] for row in recall],
            [row["ci95_high"] - row["mean"] for row in recall],
        ],
        color="tab:orange",
        marker="s",
        capsize=3,
        label="Recall@10",
    )
    left.set_xlabel("Latent dimension")
    left.set_ylabel("Pearson correlation", color="tab:blue")
    right.set_ylabel("Recall@10", color="tab:orange")
    left.tick_params(axis="y", labelcolor="tab:blue")
    right.tick_params(axis="y", labelcolor="tab:orange")
    left.set_xticks(dimensions)
    left.grid(True, alpha=0.3)
    lines = left.get_lines() + right.get_lines()
    left.legend(lines, [line.get_label() for line in lines], loc="lower right", fontsize=8)
    fig.savefig(
        plot_dir / "pearson_and_recall_at_10_vs_latent_dim.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_dimension_saturation_comparisons(rows: list[dict[str, Any]], comparison_dir: Path) -> None:
    import matplotlib.pyplot as plt

    comparison_dir.mkdir(parents=True, exist_ok=True)
    pearson = sorted(
        [row for row in rows if row["metric"] == "pearson"],
        key=lambda row: row["latent_dim"],
    )
    recall = sorted(
        [row for row in rows if row["metric"] == "learned_recall_at_10"],
        key=lambda row: row["latent_dim"],
    )
    dimensions = [row["latent_dim"] for row in pearson]
    if dimensions != [row["latent_dim"] for row in recall]:
        raise ValueError("Pearson and Recall@10 latent-dimension rows do not align")

    def values_and_errors(metric_rows: list[dict[str, Any]]):
        means = np.asarray([row["mean"] for row in metric_rows], dtype=float)
        errors = np.asarray(
            [
                [row["mean"] - row["ci95_low"] for row in metric_rows],
                [row["ci95_high"] - row["mean"] for row in metric_rows],
            ],
            dtype=float,
        )
        return means, errors

    pearson_mean, pearson_error = values_and_errors(pearson)
    recall_mean, recall_error = values_and_errors(recall)

    # Option A: aligned panels make the different saturation rates directly comparable.
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.7, 6.2), sharex=True, constrained_layout=True
    )
    top.errorbar(
        dimensions,
        pearson_mean,
        yerr=pearson_error,
        color="tab:blue",
        marker="o",
        capsize=3,
        linewidth=1.8,
        label="Pearson correlation",
    )
    bottom.errorbar(
        dimensions,
        recall_mean,
        yerr=recall_error,
        color="tab:orange",
        marker="s",
        capsize=3,
        linewidth=1.8,
        label="Recall@10",
    )
    top.set_ylabel("Pearson correlation")
    bottom.set_ylabel("Recall@10")
    bottom.set_xlabel("Latent dimension")
    top.set_ylim(0.90, 1.005)
    bottom.set_ylim(0.48, 0.79)
    bottom.set_xticks(dimensions)
    for axis in (top, bottom):
        axis.grid(True, alpha=0.25)
        axis.legend(loc="lower right", frameon=False)
    fig.savefig(
        comparison_dir / "correlation_vs_recall_twopanel.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Option B: compact dual-axis view with a restrained plateau annotation.
    fig, left = plt.subplots(figsize=(7.0, 4.4))
    right = left.twinx()
    pearson_artist = left.errorbar(
        dimensions,
        pearson_mean,
        yerr=pearson_error,
        color="tab:blue",
        marker="o",
        capsize=3,
        linewidth=1.8,
        label="Pearson correlation",
    )
    recall_artist = right.errorbar(
        dimensions,
        recall_mean,
        yerr=recall_error,
        color="tab:orange",
        marker="s",
        capsize=3,
        linewidth=1.8,
        label="Recall@10",
    )
    left.axvspan(4, 16, color="tab:orange", alpha=0.07)
    right.annotate(
        "retrieval plateau",
        xy=(8, recall_mean[3]),
        xytext=(5.2, 0.765),
        color="tab:orange",
        arrowprops={"arrowstyle": "->", "color": "tab:orange", "lw": 1.0},
    )
    left.set_xlabel("Latent dimension")
    left.set_ylabel("Pearson correlation", color="tab:blue")
    right.set_ylabel("Recall@10", color="tab:orange")
    left.set_ylim(0.90, 1.005)
    right.set_ylim(0.48, 0.79)
    left.set_xticks(dimensions)
    left.tick_params(axis="y", labelcolor="tab:blue")
    right.tick_params(axis="y", labelcolor="tab:orange")
    left.grid(True, alpha=0.25)
    left.legend(
        [pearson_artist, recall_artist],
        ["Pearson correlation", "Recall@10"],
        loc="lower right",
        frameon=False,
    )
    fig.savefig(
        comparison_dir / "correlation_vs_recall_dualaxis.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Option C: normalize values and intervals by each metric's observed maximum mean.
    pearson_scale = float(pearson_mean.max())
    recall_scale = float(recall_mean.max())
    fig, axis = plt.subplots(figsize=(7.0, 4.4))
    axis.errorbar(
        dimensions,
        pearson_mean / pearson_scale,
        yerr=pearson_error / pearson_scale,
        color="tab:blue",
        marker="o",
        capsize=3,
        linewidth=1.8,
        label="Normalized Pearson correlation",
    )
    axis.errorbar(
        dimensions,
        recall_mean / recall_scale,
        yerr=recall_error / recall_scale,
        color="tab:orange",
        marker="s",
        capsize=3,
        linewidth=1.8,
        label="Normalized Recall@10",
    )
    axis.axvspan(4, 16, color="0.5", alpha=0.06)
    axis.set_xlabel("Latent dimension")
    axis.set_ylabel("Fraction of observed maximum")
    axis.set_ylim(0.72, 1.035)
    axis.set_xticks(dimensions)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower right", frameon=False)
    fig.savefig(
        comparison_dir / "correlation_vs_recall_normalized.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_publication_distance_correlation_vs_recall(
    rows: list[dict[str, Any]], output_path: Path
) -> None:
    import matplotlib.pyplot as plt

    pearson = sorted(
        [row for row in rows if row["metric"] == "pearson"],
        key=lambda row: row["latent_dim"],
    )
    recall = sorted(
        [row for row in rows if row["metric"] == "learned_recall_at_10"],
        key=lambda row: row["latent_dim"],
    )
    dimensions = [row["latent_dim"] for row in pearson]
    if dimensions != [2, 3, 4, 8, 16]:
        raise ValueError(f"Unexpected latent dimensions: {dimensions}")
    if dimensions != [row["latent_dim"] for row in recall]:
        raise ValueError("Pearson and Recall@10 latent-dimension rows do not align")

    def means_and_errors(metric_rows: list[dict[str, Any]]):
        means = np.asarray([row["mean"] for row in metric_rows], dtype=float)
        errors = np.asarray(
            [
                [row["mean"] - row["ci95_low"] for row in metric_rows],
                [row["ci95_high"] - row["mean"] for row in metric_rows],
            ],
            dtype=float,
        )
        return means, errors

    pearson_mean, pearson_error = means_and_errors(pearson)
    recall_mean, recall_error = means_and_errors(recall)
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(7.0, 4.8),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    top.errorbar(
        dimensions,
        pearson_mean,
        yerr=pearson_error,
        color="tab:blue",
        marker="o",
        capsize=3,
        linewidth=1.8,
    )
    bottom.errorbar(
        dimensions,
        recall_mean,
        yerr=recall_error,
        color="tab:orange",
        marker="s",
        capsize=3,
        linewidth=1.8,
    )
    top.set_ylabel("Pearson correlation")
    bottom.set_ylabel("Recall@10")
    bottom.set_xlabel("Latent dimension")
    top.set_ylim(0.90, 1.005)
    bottom.set_ylim(0.48, 0.79)
    for axis in (top, bottom):
        axis.set_xlim(1.7, 16.3)
        axis.set_xticks(dimensions)
        axis.grid(True, alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def frozen_manifest() -> dict[str, str]:
    files = []
    for path in FROZEN_PATHS:
        files.extend(
            sorted(p for p in ([path] if path.is_file() else path.rglob("*")) if p.is_file())
        )
    return {str(path): file_digest(path) for path in files}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def write_json(path: Path, payload: Any, refuse: bool = True) -> None:
    if refuse and path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    scalar = [{k: v for k, v in row.items() if not isinstance(v, (list, dict))} for row in records]
    fields = sorted({key for row in scalar for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar)


if __name__ == "__main__":
    main()
