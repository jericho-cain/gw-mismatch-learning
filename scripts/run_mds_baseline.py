from __future__ import annotations

# ruff: noqa: E402
import argparse
import csv
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gw_mismatch_learning.config import load_config
from gw_mismatch_learning.datasets.gw import load_distance_matrix_dataset
from gw_mismatch_learning.evaluation.geometry import (
    distance_correlations,
    distance_error_summary,
    pairwise_euclidean,
)
from gw_mismatch_learning.evaluation.retrieval import retrieval_report
from gw_mismatch_learning.retrieval.index import SklearnNeighborIndex

FROZEN_PATHS = [
    Path("outputs/scaling_validation"),
    Path("outputs/principal_k8"),
    Path("outputs/latent_dimension_sweep"),
    Path("docs/paper_validation/mismatch_supervised_representation_framework.tex"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mds_baseline.yaml")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        raise ValueError("Choose --resume or --overwrite")
    config = load_config(args.config)
    validate_config(config)
    if args.dry_run:
        print(
            resource_estimate(
                int(load_distance_matrix_dataset(config["source_cache"]).distance.shape[0])
            )
        )
        return
    output = Path(config["output_dir"])
    summary_path = output / "summary.json"
    if summary_path.exists() and args.resume:
        validate_completed_output(config, summary_path)
        print("Validated completed MDS output; nothing rerun")
        return
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}; use --resume or --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    before = frozen_manifest()
    dump_json(output / "frozen_manifest_before.json", before)
    validation = validate_small_banks(config)
    dump_json(output / "validation.json", validation)
    if args.validate_only:
        print(json.dumps(validation, indent=2))
        return
    run_full(config, output, validation)
    after = frozen_manifest()
    dump_json(output / "frozen_manifest_after.json", after)
    if before != after:
        raise RuntimeError("A frozen artifact changed")


def double_center_squared_distances(distance: np.ndarray) -> np.ndarray:
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be square")
    squared = np.square(distance, dtype=np.float64)
    row_mean = squared.mean(axis=1, keepdims=True)
    grand_mean = float(row_mean.mean())
    squared -= row_mean
    squared -= row_mean.T
    squared += grand_mean
    squared *= -0.5
    return (squared + squared.T) * 0.5


def partial_eigenpairs(
    gram: np.ndarray, count: int, tolerance: float = 1e-9, maxiter: int = 5000
) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = eigsh(gram, k=count, which="LA", tol=tolerance, maxiter=maxiter)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def full_eigensystem(gram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = eigh(gram, driver="evd", check_finite=False)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def spectral_diagnostics(values: np.ndarray, relative_tolerance: float) -> dict[str, Any]:
    scale = max(float(np.max(np.abs(values))), 1.0)
    threshold = relative_tolerance * scale
    positive = values > threshold
    negative = values < -threshold
    near_zero = ~(positive | negative)
    positive_mass = float(values[positive].sum())
    negative_mass = float(np.abs(values[negative]).sum())
    return {
        "near_zero_threshold": threshold,
        "num_positive": int(positive.sum()),
        "num_near_zero": int(near_zero.sum()),
        "num_negative": int(negative.sum()),
        "total_positive_spectral_mass": positive_mass,
        "total_absolute_negative_spectral_mass": negative_mass,
        "negative_to_positive_mass_ratio": negative_mass / positive_mass,
        "largest_eigenvalue": float(values.max()),
        "most_negative_eigenvalue": float(values.min()),
    }


def coordinates(values: np.ndarray, vectors: np.ndarray, dimension: int) -> np.ndarray:
    positive = np.flatnonzero(values > 0)
    if positive.size < dimension:
        raise ValueError(f"Only {positive.size} positive eigenvalues; requested k={dimension}")
    selected = positive[:dimension]
    return vectors[:, selected] * np.sqrt(values[selected])[None, :]


def evaluate_coordinates(
    coordinates_array: np.ndarray, mismatch: np.ndarray, k_values: list[int]
) -> dict[str, float]:
    latent_distance = pairwise_euclidean(coordinates_array)
    report = {}
    report.update(distance_correlations(latent_distance, mismatch))
    report.update(distance_error_summary(latent_distance, mismatch))
    index = SklearnNeighborIndex(coordinates_array)
    _, retrieved = index.query(coordinates_array, top_k=max(k_values) + 1)
    for top_k in k_values:
        metrics = retrieval_report(mismatch, retrieved, top_k=top_k)
        report[f"recall_at_{top_k}"] = metrics[f"recall_at_{top_k}"]
        report[f"recovered_best_match_at_{top_k}"] = metrics["recovered_best_match_at_k"]
    report["normalized_stress"] = float(
        np.sqrt(np.sum((latent_distance - mismatch) ** 2) / np.sum(mismatch**2))
    )
    return report


def validate_small_banks(config: dict[str, Any]) -> dict[str, Any]:
    records = []
    for size in config["validation_bank_sizes"]:
        path = Path(f"data/processed/waveform_bank_{size}_mismatch.h5")
        dataset = load_distance_matrix_dataset(path)
        start = time.perf_counter()
        gram = double_center_squared_distances(dataset.distance)
        full_values, full_vectors = full_eigensystem(gram)
        partial_values, partial_vectors = partial_eigenpairs(
            gram,
            count=max(config["dimensions"]),
            tolerance=float(config["partial_eigensolver_tolerance"]),
            maxiter=int(config["partial_eigensolver_maxiter"]),
        )
        coordinate_distance_errors = []
        for dimension in config["dimensions"]:
            full_distance = pairwise_euclidean(
                coordinates(full_values, full_vectors, int(dimension))
            )
            partial_distance = pairwise_euclidean(
                coordinates(partial_values, partial_vectors, int(dimension))
            )
            coordinate_distance_errors.append(
                float(np.max(np.abs(full_distance - partial_distance)))
            )
        records.append(
            {
                "bank_size": int(size),
                "runtime_seconds": time.perf_counter() - start,
                "peak_memory_mb": peak_memory_mb(),
                "max_leading_eigenvalue_abs_error": float(
                    np.max(np.abs(full_values[:16] - partial_values[:16]))
                ),
                "max_reconstructed_distance_abs_error": max(coordinate_distance_errors),
                "full_spectral_diagnostics": spectral_diagnostics(
                    full_values, float(config["near_zero_relative_tolerance"])
                ),
            }
        )
    return {"status": "completed", "records": records}


def run_full(config: dict[str, Any], output: Path, validation: dict[str, Any]) -> None:
    total_start = time.perf_counter()
    source = Path(config["source_cache"])
    dataset = load_distance_matrix_dataset(source)
    gram_start = time.perf_counter()
    gram = double_center_squared_distances(dataset.distance)
    gram_time = time.perf_counter() - gram_start
    spectrum_start = time.perf_counter()
    all_values = eigh(gram, eigvals_only=True, driver="evd", check_finite=False)
    all_values = np.sort(all_values)[::-1]
    spectrum_time = time.perf_counter() - spectrum_start
    partial_start = time.perf_counter()
    values, vectors = partial_eigenpairs(
        gram,
        max(config["dimensions"]),
        float(config["partial_eigensolver_tolerance"]),
        int(config["partial_eigensolver_maxiter"]),
    )
    partial_time = time.perf_counter() - partial_start
    if np.count_nonzero(values > 0) < max(config["dimensions"]):
        raise RuntimeError("Fewer than 16 positive leading eigenvalues")
    records = []
    for dimension in config["dimensions"]:
        start = time.perf_counter()
        metric = evaluate_coordinates(
            coordinates(values, vectors, int(dimension)),
            dataset.distance,
            [int(k) for k in config["k_values"]],
        )
        records.append(
            {
                "status": "completed",
                "dimension": int(dimension),
                **metric,
                "evaluation_runtime_seconds": time.perf_counter() - start,
            }
        )
    diagnostics = spectral_diagnostics(all_values, float(config["near_zero_relative_tolerance"]))
    diagnostics["leading_eigenvalues"] = [float(x) for x in values]
    diagnostics["smallest_eigenvalues"] = [float(x) for x in all_values[-16:]]
    dump_json(output / "eigenspectrum.json", diagnostics)
    write_csv(output / "metrics_by_dimension.csv", records)
    comparisons = comparison_rows(config, records)
    write_csv(output / "comparisons.csv", comparisons)
    run_record = {
        "status": "completed",
        "source_cache": str(source),
        "source_cache_sha256": file_digest(source),
        "git_commit": git_commit(),
        "gram_runtime_seconds": gram_time,
        "full_spectrum_runtime_seconds": spectrum_time,
        "partial_eigensolver_runtime_seconds": partial_time,
        "total_runtime_seconds": time.perf_counter() - total_start,
        "peak_memory_mb": peak_memory_mb(),
        "converged": True,
    }
    write_csv(output / "runs.csv", [run_record])
    serialized_config = {
        **config,
        "git_commit": git_commit(),
        "source_cache_sha256": file_digest(source),
    }
    dump_json(output / "config.json", serialized_config)
    summary = {
        "status": "completed",
        "objective": "classical_mds_in_sample_reference",
        "configuration": serialized_config,
        "run": run_record,
        "validation": validation,
        "eigenspectrum": diagnostics,
        "metrics_by_dimension": records,
        "comparisons": comparisons,
        "phase_iii_available": False,
        "phase_iii_limitation": "No parametric out-of-sample mapping exists.",
    }
    dump_json(output / "summary.json", summary)
    make_plots(output / "figures", records, comparisons, diagnostics)


def comparison_rows(
    config: dict[str, Any], mds_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    aggregates = read_csv(Path(config["learned_aggregate"]))
    learned_runs = read_csv(Path(config["learned_runs"]))
    metrics = [
        "pearson",
        "distance_rmse",
        "learned_recall_at_10",
        "learned_recovered_best_match_at_10",
    ]
    mds_names = {
        "learned_recall_at_10": "recall_at_10",
        "learned_recovered_best_match_at_10": "recovered_best_match_at_10",
    }
    rows = []
    for mds in mds_records:
        dimension = int(mds["dimension"])
        for metric in metrics:
            aggregate = next(
                r for r in aggregates if int(r["latent_dim"]) == dimension and r["metric"] == metric
            )
            seed = next(
                r
                for r in learned_runs
                if int(r["latent_dim"]) == dimension and int(r["seed"]) == 1234
            )
            rows.append(
                {
                    "dimension": dimension,
                    "metric": metric,
                    "mds": mds[mds_names.get(metric, metric)],
                    "encoder_mean": float(aggregate["mean"]),
                    "encoder_ci95_low": float(aggregate["ci95_low"]),
                    "encoder_ci95_high": float(aggregate["ci95_high"]),
                    "encoder_seed_1234": float(seed[metric]),
                }
            )
    return rows


def make_plots(
    plot_dir: Path,
    records: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, filename in [
        ("pearson", "Pearson correlation", "mds_vs_encoder_pearson.png"),
        ("distance_rmse", "Distance RMSE", "mds_vs_encoder_rmse.png"),
        ("learned_recall_at_10", "Recall@10", "mds_vs_encoder_recall_at_10.png"),
        ("learned_recovered_best_match_at_10", "Best@10", "mds_vs_encoder_best_at_10.png"),
    ]:
        rows = [r for r in comparisons if r["metric"] == metric]
        x = [r["dimension"] for r in rows]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(x, [r["mds"] for r in rows], marker="s", label="Classical MDS")
        mean = [r["encoder_mean"] for r in rows]
        ax.errorbar(
            x,
            mean,
            yerr=[
                [m - r["encoder_ci95_low"] for m, r in zip(mean, rows, strict=True)],
                [r["encoder_ci95_high"] - m for m, r in zip(mean, rows, strict=True)],
            ],
            marker="o",
            capsize=3,
            label="Encoder (five-seed mean)",
        )
        ax.set_xlabel("Embedding dimension")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.savefig(plot_dir / filename, dpi=250, bbox_inches="tight")
        plt.close(fig)
    positive = diagnostics["leading_eigenvalues"]
    negative = diagnostics["smallest_eigenvalues"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(range(1, len(positive) + 1), positive, marker="o")
    axes[0].set_title("Leading eigenvalues")
    axes[1].plot(range(1, len(negative) + 1), negative, marker="o")
    axes[1].set_title("Most negative eigenvalues")
    for ax in axes:
        ax.set_xlabel("Rank")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Eigenvalue")
    fig.savefig(plot_dir / "mds_eigenspectrum.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def validate_config(config: dict[str, Any]) -> None:
    for key in ["source_cache", "learned_aggregate", "learned_runs"]:
        if not Path(config[key]).is_file():
            raise FileNotFoundError(config[key])
    if max(config["dimensions"]) > 16:
        raise ValueError("This validation is configured for k<=16")


def validate_completed_output(config: dict[str, Any], summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "completed":
        raise RuntimeError("Output is not completed")
    if summary["configuration"]["source_cache_sha256"] != file_digest(Path(config["source_cache"])):
        raise RuntimeError("Source cache hash mismatch")
    if summary["configuration"]["dimensions"] != config["dimensions"]:
        raise RuntimeError("Dimension mismatch")


def resource_estimate(size: int) -> dict[str, Any]:
    return {
        "size": size,
        "source_float32_mib": size * size * 4 / 2**20,
        "gram_float64_mib": size * size * 8 / 2**20,
        "complexity": "full spectrum O(N^3); partial leading eigenpairs iterative",
    }


def peak_memory_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def frozen_manifest() -> dict[str, str]:
    files = []
    for path in FROZEN_PATHS:
        files.extend([path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()])
    return {str(p): file_digest(p) for p in sorted(files)}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


if __name__ == "__main__":
    main()
