from __future__ import annotations

import argparse
import copy
import csv
import json
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gw_mismatch_learning.config import load_config
from gw_mismatch_learning.datasets.distance_regression import DistanceRegressionDataset
from gw_mismatch_learning.datasets.gw import DistanceMatrixDataset, load_distance_matrix_dataset
from gw_mismatch_learning.datasets.pairs import sample_pairs
from gw_mismatch_learning.models.encoders import MLPEncoder
from gw_mismatch_learning.models.metric_learning import encode_array, train_distance_regression
from gw_mismatch_learning.retrieval.index import SklearnNeighborIndex
from gw_mismatch_learning.utils.io import ensure_dir
from gw_mismatch_learning.utils.seeds import set_seed
from gw_mismatch_learning.waveforms.banks import BinaryMassBank, sample_binary_mass_bank
from gw_mismatch_learning.waveforms.generate import generate_pycbc_fd_waveform


@dataclass(frozen=True)
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.std).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3_candidate_retrieval.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.get("output_dir", "outputs/phase3_candidate_retrieval"))
    protect_output_dir(
        output_dir,
        allow_overwrite=bool(config.get("allow_overwrite_output", False)),
    )
    output_dir = ensure_dir(output_dir)
    query_results, summary_records, run_records, metadata = run_phase3_candidate_retrieval(config)
    write_query_results(output_dir / "query_results.csv", query_results)
    write_summary_csv(output_dir / "summary.csv", summary_records)
    write_runs(output_dir / "runs.csv", run_records)
    write_summary_json(output_dir / "summary.json", metadata, summary_records)
    make_plots(output_dir, summary_records)


def protect_output_dir(output_dir: Path, allow_overwrite: bool = False) -> None:
    if allow_overwrite:
        return
    protected = [
        output_dir / "runs.csv",
        output_dir / "summary.csv",
        output_dir / "summary.json",
        output_dir / "query_results.csv",
    ]
    existing = [path for path in protected if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Phase III output files already exist. Use a new output_dir or set "
            f"allow_overwrite_output: true to replace them intentionally. Existing: {joined}"
        )


def run_phase3_candidate_retrieval(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    total_start = time.perf_counter()
    base_config = load_config(config["base_config"])
    bank_cache_path = Path(config["bank_cache_path"])
    if not bank_cache_path.exists():
        raise FileNotFoundError(f"Missing frozen bank cache: {bank_cache_path}")

    dataset = load_distance_matrix_dataset(bank_cache_path)
    bank_size = int(config.get("bank_size", len(dataset.features)))
    if bank_size != len(dataset.features):
        raise ValueError(
            f"Configured bank_size={bank_size} but cache contains {len(dataset.features)}"
        )

    training_config = phase1_training_config(base_config, bank_size, bank_cache_path)
    set_seed(int(training_config.get("seed", 1234)))
    train_start = time.perf_counter()
    encoder, final_train_loss = train_encoder(training_config, dataset)
    training_time = time.perf_counter() - train_start

    bank_raw = raw_feature_matrix_from_metadata(dataset.metadata)
    scaler = fit_feature_scaler(bank_raw)
    bank_features = scaler.transform(bank_raw)
    bank_embeddings = encode_array(encoder, bank_features)

    query_start = time.perf_counter()
    query_bank = sample_query_bank(
        num_queries=int(config["num_queries"]),
        metadata=dataset.metadata,
        bank_masses=bank_raw[:, :2],
        seed=int(config["query_seed"]),
    )
    query_raw = raw_feature_matrix_from_masses(query_bank.mass_1, query_bank.mass_2)
    query_features = scaler.transform(query_raw)
    query_waveforms = generate_waveforms(query_bank, dataset.metadata)
    query_generation_time = time.perf_counter() - query_start

    bank_waveform_start = time.perf_counter()
    bank_waveforms = generate_waveforms(
        BinaryMassBank(
            mass_1=np.asarray(dataset.metadata["mass_1"], dtype=np.float32),
            mass_2=np.asarray(dataset.metadata["mass_2"], dtype=np.float32),
        ),
        dataset.metadata,
    )
    bank_waveform_generation_time = time.perf_counter() - bank_waveform_start
    prepared_bank, prepared_queries, psd, match_fn = prepare_matching(
        bank_waveforms,
        query_waveforms,
        dataset.metadata,
    )

    exhaustive_start = time.perf_counter()
    exhaustive_matches = compute_query_bank_matches(
        prepared_queries,
        prepared_bank,
        psd=psd,
        match_fn=match_fn,
        f_lower=float(dataset.metadata["f_lower"]),
    )
    exhaustive_matching_time = time.perf_counter() - exhaustive_start
    exhaustive_best_indices = exhaustive_matches.argmax(axis=1)
    exhaustive_best_matches = exhaustive_matches.max(axis=1)

    embedding_start = time.perf_counter()
    query_embeddings = encode_array(encoder, query_features)
    latent_embedding_time = time.perf_counter() - embedding_start

    methods = build_method_coordinates(
        config,
        bank_features,
        query_features,
        bank_embeddings,
        query_embeddings,
    )
    k_values = [int(value) for value in config["candidate_k"]]
    query_results: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []

    for method in methods:
        retrieval_start = time.perf_counter()
        index = SklearnNeighborIndex(method["bank"])
        _, nearest = index.query(method["query"], top_k=max(k_values))
        retrieval_time = time.perf_counter() - retrieval_start

        candidate_total_time = 0.0
        for top_k in k_values:
            candidate_indices = nearest[:, :top_k]
            candidate_start = time.perf_counter()
            candidate_matches = compute_candidate_matches(
                prepared_queries,
                prepared_bank,
                candidate_indices,
                psd=psd,
                match_fn=match_fn,
                f_lower=float(dataset.metadata["f_lower"]),
            )
            candidate_time = time.perf_counter() - candidate_start
            candidate_total_time += candidate_time

            method_query_rows = query_rows_for_k(
                method,
                top_k,
                candidate_indices,
                candidate_matches,
                exhaustive_best_indices,
                exhaustive_best_matches,
                query_bank,
            )
            query_results.extend(method_query_rows)
            summary = summarize_query_rows(method, top_k, method_query_rows, bank_size)
            summary["latent_retrieval_time_seconds"] = retrieval_time
            summary["candidate_matching_time_seconds"] = candidate_time
            summary["total_candidate_pipeline_time_seconds"] = (
                latent_embedding_time + retrieval_time + candidate_time
            )
            summary_records.append(summary)

        run_records.append(
            {
                "status": "completed",
                "method": method["name"],
                "method_label": method["label"],
                "bank_size": bank_size,
                "num_queries": int(config["num_queries"]),
                "k_values": ",".join(str(value) for value in k_values),
                "exhaustive_match_evaluations": int(config["num_queries"]) * bank_size,
                "query_waveform_generation_time_seconds": query_generation_time,
                "bank_waveform_generation_time_seconds": bank_waveform_generation_time,
                "training_time_seconds": training_time,
                "exhaustive_matching_time_seconds": exhaustive_matching_time,
                "latent_embedding_time_seconds": latent_embedding_time,
                "latent_retrieval_time_seconds": retrieval_time,
                "candidate_matching_time_seconds_all_k": candidate_total_time,
                "total_runtime_seconds": time.perf_counter() - total_start,
                "peak_memory_mb": peak_memory_mb(),
                "final_train_loss": final_train_loss,
            }
        )

    metadata = {
        "objective": "phase3_candidate_retrieval",
        "bank_size": bank_size,
        "bank_cache_path": str(bank_cache_path),
        "num_queries": int(config["num_queries"]),
        "query_seed": int(config["query_seed"]),
        "candidate_k": k_values,
        "methods": [
            {
                "name": method["name"],
                "label": method["label"],
                "feature_names": method["feature_names"],
            }
            for method in methods
        ],
        "exhaustive_match_evaluations": int(config["num_queries"]) * bank_size,
        "waveform_settings": waveform_settings(dataset.metadata),
        "training_protocol": {
            "seed": int(training_config.get("seed", 1234)),
            "pairs_num_pairs": int(training_config["pairs"]["num_pairs"]),
            "model": training_config["model"],
            "training": training_config["training"],
            "final_train_loss": final_train_loss,
        },
        "timings": {
            "query_waveform_generation_time_seconds": query_generation_time,
            "bank_waveform_generation_time_seconds": bank_waveform_generation_time,
            "training_time_seconds": training_time,
            "exhaustive_matching_time_seconds": exhaustive_matching_time,
            "latent_embedding_time_seconds": latent_embedding_time,
            "total_runtime_seconds": time.perf_counter() - total_start,
            "peak_memory_mb": peak_memory_mb(),
        },
    }
    return query_results, summary_records, run_records, metadata


def phase1_training_config(
    base_config: dict[str, Any],
    bank_size: int,
    bank_cache_path: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["seed"] = int(base_config.get("seed", 1234))
    config["gw_data"]["num_waveforms"] = bank_size
    config["gw_data"]["run_seed"] = config["seed"]
    config["gw_data"]["cache_path"] = str(bank_cache_path)
    config["gw_data"]["overwrite"] = False
    config["pairs"]["num_pairs"] = max(int(config["pairs"]["num_pairs"]), bank_size * 32)
    return config


def train_encoder(
    config: dict[str, Any],
    dataset: DistanceMatrixDataset,
) -> tuple[MLPEncoder, float]:
    seed = int(config.get("seed", 1234))
    pairs = sample_pairs(
        dataset.features,
        dataset.distance,
        int(config["pairs"]["num_pairs"]),
        seed=seed,
    )
    model_cfg = config["model"]
    encoder = MLPEncoder(
        input_dim=int(model_cfg["input_dim"]),
        embedding_dim=int(model_cfg["embedding_dim"]),
        hidden_dims=tuple(int(dim) for dim in model_cfg.get("hidden_dims", [64, 32])),
    )
    train_cfg = config["training"]
    history = train_distance_regression(
        encoder,
        DistanceRegressionDataset(pairs),
        epochs=int(train_cfg["epochs"]),
        batch_size=int(train_cfg["batch_size"]),
        learning_rate=float(train_cfg["learning_rate"]),
    )
    return encoder, history.losses[-1]


def fit_feature_scaler(raw_features: np.ndarray) -> FeatureScaler:
    mean = raw_features.mean(axis=0, keepdims=True)
    std = raw_features.std(axis=0, keepdims=True).clip(min=1e-6)
    return FeatureScaler(mean=mean.astype(np.float32), std=std.astype(np.float32))


def raw_feature_matrix_from_metadata(metadata: dict[str, Any]) -> np.ndarray:
    return raw_feature_matrix_from_masses(
        np.asarray(metadata["mass_1"], dtype=np.float32),
        np.asarray(metadata["mass_2"], dtype=np.float32),
    )


def raw_feature_matrix_from_masses(mass_1: np.ndarray, mass_2: np.ndarray) -> np.ndarray:
    total_mass = mass_1 + mass_2
    eta = (mass_1 * mass_2) / np.square(total_mass)
    chirp_mass = np.power(mass_1 * mass_2, 3.0 / 5.0) / np.power(total_mass, 1.0 / 5.0)
    return np.column_stack([mass_1, mass_2, total_mass, eta, chirp_mass]).astype(np.float32)


def sample_query_bank(
    num_queries: int,
    metadata: dict[str, Any],
    bank_masses: np.ndarray,
    seed: int,
) -> BinaryMassBank:
    bank_mass_set = {(float(row[0]), float(row[1])) for row in bank_masses.astype(np.float32)}
    mass_1: list[float] = []
    mass_2: list[float] = []
    rng_seed = seed
    while len(mass_1) < num_queries:
        remaining = num_queries - len(mass_1)
        batch = sample_binary_mass_bank(
            num_waveforms=remaining,
            mass_1_range=(float(metadata["mass_1_min"]), float(metadata["mass_1_max"])),
            mass_2_range=(float(metadata["mass_2_min"]), float(metadata["mass_2_max"])),
            seed=rng_seed,
        )
        for left, right in zip(batch.mass_1, batch.mass_2, strict=True):
            key = (float(left), float(right))
            if key not in bank_mass_set:
                mass_1.append(float(left))
                mass_2.append(float(right))
        rng_seed += 1
    return BinaryMassBank(
        mass_1=np.asarray(mass_1, dtype=np.float32),
        mass_2=np.asarray(mass_2, dtype=np.float32),
    )


def generate_waveforms(bank: BinaryMassBank, metadata: dict[str, Any]) -> list[Any]:
    return [
        generate_pycbc_fd_waveform(
            mass1=float(mass_1),
            mass2=float(mass_2),
            approximant=str(metadata.get("approximant", "IMRPhenomD")),
            delta_f=float(metadata.get("delta_f", 1.0 / 16.0)),
            f_lower=float(metadata.get("f_lower", 20.0)),
            f_final=float(metadata.get("f_final", 512.0)),
        )
        for mass_1, mass_2 in zip(bank.mass_1, bank.mass_2, strict=True)
    ]


def prepare_matching(
    bank_waveforms: list[Any],
    query_waveforms: list[Any],
    metadata: dict[str, Any],
):
    try:
        from pycbc.filter import match
        from pycbc.psd import aLIGOZeroDetHighPower
    except ImportError as exc:
        raise ImportError("pycbc is required for Phase III candidate retrieval.") from exc

    psd_name = str(metadata.get("psd", "aLIGOZeroDetHighPower"))
    if psd_name != "aLIGOZeroDetHighPower":
        raise ValueError(f"Unsupported PSD for Phase III: {psd_name}")

    max_len = max(
        max(len(waveform) for waveform in bank_waveforms),
        max(len(waveform) for waveform in query_waveforms),
    )
    prepared_bank = resize_waveforms(bank_waveforms, max_len)
    prepared_queries = resize_waveforms(query_waveforms, max_len)
    psd = aLIGOZeroDetHighPower(max_len, float(metadata["delta_f"]), float(metadata["f_lower"]))
    return prepared_bank, prepared_queries, psd, match


def resize_waveforms(waveforms: list[Any], length: int) -> list[Any]:
    prepared = []
    for waveform in waveforms:
        copy = waveform.copy()
        copy.resize(length)
        prepared.append(copy)
    return prepared


def compute_query_bank_matches(
    queries: list[Any],
    bank: list[Any],
    *,
    psd: Any,
    match_fn: Any,
    f_lower: float,
) -> np.ndarray:
    matches = np.zeros((len(queries), len(bank)), dtype=np.float32)
    for query_index, query in enumerate(queries):
        for bank_index, template in enumerate(bank):
            value, _ = match_fn(query, template, psd=psd, low_frequency_cutoff=f_lower)
            matches[query_index, bank_index] = float(value)
    return matches


def compute_candidate_matches(
    queries: list[Any],
    bank: list[Any],
    candidate_indices: np.ndarray,
    *,
    psd: Any,
    match_fn: Any,
    f_lower: float,
) -> np.ndarray:
    matches = np.zeros(candidate_indices.shape, dtype=np.float32)
    for query_index, query in enumerate(queries):
        for rank, bank_index in enumerate(candidate_indices[query_index]):
            value, _ = match_fn(query, bank[int(bank_index)], psd=psd, low_frequency_cutoff=f_lower)
            matches[query_index, rank] = float(value)
    return matches


def build_method_coordinates(
    config: dict[str, Any],
    bank_features: np.ndarray,
    query_features: np.ndarray,
    bank_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
) -> list[dict[str, Any]]:
    feature_indices = {"m1": 0, "m2": 1, "M": 2, "eta": 3, "chirp_mass": 4}
    methods = []
    for method in config["methods"]:
        method_type = str(method["type"])
        if method_type == "learned":
            methods.append(
                {
                    "name": str(method["name"]),
                    "label": str(method.get("label", method["name"])),
                    "feature_names": ["learned_latent_coordinates"],
                    "bank": bank_embeddings,
                    "query": query_embeddings,
                }
            )
        elif method_type == "physical":
            names = [str(name) for name in method["feature_names"]]
            columns = [feature_indices[name] for name in names]
            methods.append(
                {
                    "name": str(method["name"]),
                    "label": str(method.get("label", method["name"])),
                    "feature_names": names,
                    "bank": bank_features[:, columns],
                    "query": query_features[:, columns],
                }
            )
        else:
            raise ValueError(f"Unsupported retrieval method type: {method_type}")
    return methods


def query_rows_for_k(
    method: dict[str, Any],
    top_k: int,
    candidate_indices: np.ndarray,
    candidate_matches: np.ndarray,
    exhaustive_best_indices: np.ndarray,
    exhaustive_best_matches: np.ndarray,
    query_bank: BinaryMassBank,
) -> list[dict[str, Any]]:
    rows = []
    candidate_best_rank = candidate_matches.argmax(axis=1)
    for query_index, rank in enumerate(candidate_best_rank):
        candidate_index = int(candidate_indices[query_index, rank])
        candidate_match = float(candidate_matches[query_index, rank])
        exhaustive_match = float(exhaustive_best_matches[query_index])
        delta_match = exhaustive_match - candidate_match
        rows.append(
            {
                "query_index": query_index,
                "query_m1": float(query_bank.mass_1[query_index]),
                "query_m2": float(query_bank.mass_2[query_index]),
                "method": method["name"],
                "method_label": method["label"],
                "candidate_k": top_k,
                "exhaustive_best_index": int(exhaustive_best_indices[query_index]),
                "candidate_best_index": candidate_index,
                "exact_best_recovered": int(
                    int(exhaustive_best_indices[query_index]) in candidate_indices[query_index]
                ),
                "exhaustive_best_match": exhaustive_match,
                "candidate_best_match": candidate_match,
                "delta_match": delta_match,
                "match_retention_ratio": candidate_match / exhaustive_match
                if exhaustive_match > 0.0
                else np.nan,
            }
        )
    return rows


def summarize_query_rows(
    method: dict[str, Any],
    top_k: int,
    rows: list[dict[str, Any]],
    bank_size: int,
) -> dict[str, Any]:
    delta = np.asarray([row["delta_match"] for row in rows], dtype=np.float64)
    retention = np.asarray([row["match_retention_ratio"] for row in rows], dtype=np.float64)
    recovered = np.asarray([row["exact_best_recovered"] for row in rows], dtype=np.float64)
    candidate_fraction = top_k / bank_size
    return {
        "method": method["name"],
        "method_label": method["label"],
        "feature_names": ",".join(method["feature_names"]),
        "candidate_k": top_k,
        "bank_size": bank_size,
        "num_queries": len(rows),
        "candidate_fraction": candidate_fraction,
        "candidate_reduction": bank_size / top_k,
        "exhaustive_match_evaluations": len(rows) * bank_size,
        "candidate_match_evaluations": len(rows) * top_k,
        "match_evaluation_reduction": bank_size / top_k,
        "exact_best_recovery_rate": float(recovered.mean()),
        "delta_match_mean": float(delta.mean()),
        "delta_match_median": float(np.median(delta)),
        "delta_match_max": float(delta.max()),
        "delta_match_p95": float(np.percentile(delta, 95)),
        "delta_match_p99": float(np.percentile(delta, 99)),
        "fraction_delta_match_le_1e_4": float((delta <= 1e-4).mean()),
        "fraction_delta_match_le_1e_3": float((delta <= 1e-3).mean()),
        "fraction_delta_match_le_1e_2": float((delta <= 1e-2).mean()),
        "match_retention_mean": float(retention.mean()),
        "match_retention_median": float(np.median(retention)),
        "match_retention_p05": float(np.percentile(retention, 5)),
        "match_retention_p01": float(np.percentile(retention, 1)),
    }


def waveform_settings(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "approximant",
        "psd",
        "delta_f",
        "f_lower",
        "f_final",
        "mass_1_min",
        "mass_1_max",
        "mass_2_min",
        "mass_2_max",
        "spin_model",
    ]
    return {key: metadata[key] for key in keys if key in metadata}


def write_query_results(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows)


def write_runs(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(
    path: Path,
    metadata: dict[str, Any],
    summary_records: list[dict[str, Any]],
) -> None:
    payload = {
        **metadata,
        "num_summary_records": len(summary_records),
        "summary_records": summary_records,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def make_plots(output_dir: Path, records: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    plot_dir = ensure_dir(output_dir / "figures")
    plot_metric_vs_fraction(
        records,
        "exact_best_recovery_rate",
        "Exact-best recovery rate",
        plot_dir / "exact_best_recovery_vs_candidate_fraction.png",
    )
    plot_delta_quantiles(records, plot_dir / "delta_match_vs_candidate_fraction.png")
    plot_reduction_vs_recovery(records, plot_dir / "reduction_vs_exact_best_recovery.png")
    plot_method_comparison_at_k(records, plot_dir / "method_comparison_exact_best_recovery.png")
    plot_match_evaluations(records, plot_dir / "candidate_vs_exhaustive_evaluations.png")
    plt.close("all")


def plot_metric_vs_fraction(
    records: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for method in method_names(records):
        rows = sorted(
            [row for row in records if row["method"] == method],
            key=lambda row: row["candidate_k"],
        )
        ax.plot(
            [row["candidate_fraction"] for row in rows],
            [row[metric] for row in rows],
            marker="o",
            label=rows[0]["method_label"],
        )
    ax.set_xscale("log")
    ax.set_xlabel("Candidate fraction K / N")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_delta_quantiles(records: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharex=True)
    for method in method_names(records):
        rows = sorted(
            [row for row in records if row["method"] == method],
            key=lambda row: row["candidate_k"],
        )
        axes[0].plot(
            [row["candidate_fraction"] for row in rows],
            [row["delta_match_median"] for row in rows],
            marker="o",
            label=rows[0]["method_label"],
        )
        axes[1].plot(
            [row["candidate_fraction"] for row in rows],
            [row["delta_match_p95"] for row in rows],
            marker="o",
            label=rows[0]["method_label"],
        )
    for ax, title in zip(axes, ["Median DeltaM", "95th percentile DeltaM"], strict=True):
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_xlabel("Candidate fraction K / N")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Match loss DeltaM")
    axes[1].legend(fontsize=8)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_reduction_vs_recovery(records: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for method in method_names(records):
        rows = sorted(
            [row for row in records if row["method"] == method],
            key=lambda row: row["candidate_k"],
        )
        ax.plot(
            [row["match_evaluation_reduction"] for row in rows],
            [row["exact_best_recovery_rate"] for row in rows],
            marker="o",
            label=rows[0]["method_label"],
        )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Matched-filter evaluation reduction N / K")
    ax.set_ylabel("Exact-best recovery rate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison_at_k(records: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    rows = sorted(records, key=lambda row: (row["method_label"], row["candidate_k"]))
    for method in method_names(rows):
        method_rows = [row for row in rows if row["method"] == method]
        ax.plot(
            [row["candidate_k"] for row in method_rows],
            [row["exact_best_recovery_rate"] for row in method_rows],
            marker="o",
            label=method_rows[0]["method_label"],
        )
    ax.set_xscale("log")
    ax.set_xlabel("Candidate K")
    ax.set_ylabel("Exact-best recovery rate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_match_evaluations(records: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(
        [row for row in records if row["method"] == method_names(records)[0]],
        key=lambda row: row["candidate_k"],
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.axhline(
        rows[0]["exhaustive_match_evaluations"],
        color="black",
        linestyle="--",
        label="Exhaustive",
    )
    ax.plot(
        [row["candidate_k"] for row in rows],
        [row["candidate_match_evaluations"] for row in rows],
        marker="o",
        label="Candidate-only",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Candidate K")
    ax.set_ylabel("Exact match evaluations")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def method_names(records: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(row["method"] for row in records))


def peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


if __name__ == "__main__":
    main()
