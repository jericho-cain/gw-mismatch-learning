from __future__ import annotations

import argparse
import copy
import csv
import json
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gw_mismatch_learning.config import load_config
from gw_mismatch_learning.datasets.distance_regression import DistanceRegressionDataset
from gw_mismatch_learning.datasets.gw import (
    DistanceMatrixDataset,
    load_distance_matrix_dataset,
    save_distance_matrix_dataset,
)
from gw_mismatch_learning.datasets.pairs import sample_pairs
from gw_mismatch_learning.evaluation.geometry import (
    distance_correlations,
    distance_error_summary,
    pairwise_euclidean,
)
from gw_mismatch_learning.evaluation.retrieval import retrieval_report
from gw_mismatch_learning.models.encoders import MLPEncoder
from gw_mismatch_learning.models.metric_learning import encode_array, train_distance_regression
from gw_mismatch_learning.retrieval.index import SklearnNeighborIndex
from gw_mismatch_learning.utils.io import ensure_dir
from gw_mismatch_learning.utils.seeds import set_seed
from gw_mismatch_learning.waveforms.banks import sample_binary_mass_bank
from gw_mismatch_learning.waveforms.generate import generate_pycbc_fd_waveform
from gw_mismatch_learning.waveforms.tiny_bank import (
    _package_version,
    compute_pycbc_mismatch_matrix,
    mass_feature_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scaling_validation.yaml")
    parser.add_argument("--max-bank-size", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scaling_config = load_config(args.config)
    bank_sizes = [int(value) for value in scaling_config.get("bank_sizes", [])]
    if args.max_bank_size is not None:
        bank_sizes = [value for value in bank_sizes if value <= args.max_bank_size]
    if args.dry_run:
        print(f"Planned bank sizes: {bank_sizes}")
        return

    output_dir = ensure_dir(scaling_config.get("output_dir", "outputs/scaling_validation"))
    records = run_scaling_validation(scaling_config, bank_sizes)
    write_records(output_dir / "runs.csv", records)
    write_summary(output_dir / "summary.json", records)
    write_summary_csv(output_dir / "summary.csv", records)
    make_plots(output_dir, records)


def run_scaling_validation(
    scaling_config: dict[str, Any],
    bank_sizes: list[int],
) -> list[dict[str, Any]]:
    base_config = load_config(scaling_config["base_config"])
    output_dir = ensure_dir(scaling_config.get("output_dir", "outputs/scaling_validation"))
    cache_template = str(
        scaling_config.get("cache_template", "data/processed/waveform_bank_{bank_size}_mismatch.h5")
    )
    records: list[dict[str, Any]] = []
    for bank_size in bank_sizes:
        run_dir = ensure_dir(output_dir / f"bank_size_{bank_size}")
        config = config_for_bank_size(
            base_config,
            bank_size,
            cache_template,
            scaling_config.get("max_estimated_runtime_minutes"),
        )
        config_path = run_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        print(f"Running bank_size_{bank_size}...")
        try:
            record = run_one_bank(config, bank_size)
            record["status"] = "completed"
            print(
                f"Completed bank_size_{bank_size} in "
                f"{record['total_runtime_seconds']:.1f}s"
            )
        except Exception as exc:  # noqa: BLE001
            record = failed_record(config, bank_size, exc)
            print(f"Failed bank_size_{bank_size}: {exc}")
        record["run_id"] = f"bank_size_{bank_size}"
        record["bank_size"] = bank_size
        record["config_path"] = str(config_path)
        records.append(record)

        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return records


def config_for_bank_size(
    base_config: dict[str, Any],
    bank_size: int,
    cache_template: str,
    max_estimated_runtime_minutes: Any,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["seed"] = int(base_config.get("seed", 1234))
    config["gw_data"]["num_waveforms"] = bank_size
    config["gw_data"]["run_seed"] = config["seed"]
    config["gw_data"]["cache_path"] = cache_template.format(bank_size=bank_size)
    config["gw_data"]["overwrite"] = False
    if max_estimated_runtime_minutes is not None:
        config["gw_data"]["max_estimated_runtime_minutes"] = float(max_estimated_runtime_minutes)
    config["pairs"]["num_pairs"] = max(int(config["pairs"]["num_pairs"]), bank_size * 32)
    config["outputs"] = {
        **config.get("outputs", {}),
        "save_plots": False,
        "save_metrics": False,
    }
    return config


def run_one_bank(config: dict[str, Any], bank_size: int) -> dict[str, Any]:
    total_start = time.perf_counter()
    set_seed(int(config.get("seed", 1234)))
    dataset, data_timings, cache_reused = load_or_create_timed_dataset(config["gw_data"])

    train_start = time.perf_counter()
    encoder, final_train_loss = train_encoder(config, dataset)
    training_time = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    metrics = evaluate_encoder(config, dataset, encoder)
    evaluation_time = time.perf_counter() - eval_start

    cache_path = Path(config["gw_data"]["cache_path"])
    record: dict[str, Any] = {
        "distance_target": str(config.get("distance_target", "mismatch")),
        "cache_path": str(cache_path),
        "cache_reused": cache_reused,
        "num_waveform_pairs": int(bank_size * (bank_size - 1) // 2),
        "cache_file_size_bytes": cache_path.stat().st_size if cache_path.exists() else 0,
        "waveform_generation_time_seconds": data_timings["waveform_generation_time_seconds"],
        "pairwise_mismatch_time_seconds": data_timings["pairwise_mismatch_time_seconds"],
        "training_time_seconds": training_time,
        "evaluation_time_seconds": evaluation_time,
        "total_runtime_seconds": time.perf_counter() - total_start,
        "peak_memory_mb": peak_memory_mb(),
        "final_train_loss": final_train_loss,
    }
    record.update(metrics)
    return record


def load_or_create_timed_dataset(
    gw_config: dict[str, Any],
) -> tuple[DistanceMatrixDataset, dict[str, float], bool]:
    cache_path = Path(gw_config["cache_path"])
    if cache_path.exists() and not bool(gw_config.get("overwrite", False)):
        dataset = load_distance_matrix_dataset(cache_path)
        return (
            dataset,
            {
                "waveform_generation_time_seconds": 0.0,
                "pairwise_mismatch_time_seconds": 0.0,
            },
            True,
        )

    seed = int(gw_config.get("seed", gw_config.get("run_seed", 1234)))
    num_waveforms = int(gw_config.get("num_waveforms", 32))
    bank = sample_binary_mass_bank(
        num_waveforms=num_waveforms,
        mass_1_range=tuple(gw_config.get("mass_1_range", [20.0, 40.0])),
        mass_2_range=tuple(gw_config.get("mass_2_range", [10.0, 30.0])),
        seed=seed,
    )

    print("Generating waveform bank...")
    generation_start = time.perf_counter()
    waveforms = [
        generate_pycbc_fd_waveform(
            mass1=float(mass_1),
            mass2=float(mass_2),
            approximant=str(gw_config.get("approximant", "IMRPhenomD")),
            delta_f=float(gw_config.get("delta_f", 1.0 / 16.0)),
            f_lower=float(gw_config.get("f_lower", 20.0)),
            f_final=float(gw_config.get("f_final", 512.0)),
        )
        for mass_1, mass_2 in zip(bank.mass_1, bank.mass_2, strict=True)
    ]
    waveform_generation_time = time.perf_counter() - generation_start

    mismatch_start = time.perf_counter()
    mismatch = compute_pycbc_mismatch_matrix(
        waveforms=waveforms,
        delta_f=float(gw_config.get("delta_f", 1.0 / 16.0)),
        f_lower=float(gw_config.get("f_lower", 20.0)),
        psd_name=str(gw_config.get("psd", "aLIGOZeroDetHighPower")),
        max_estimated_runtime_minutes=float(gw_config.get("max_estimated_runtime_minutes", 30.0)),
        benchmark_pairs=int(gw_config.get("benchmark_pairs", 128)),
        show_progress=bool(gw_config.get("show_progress", True)),
    )
    pairwise_mismatch_time = time.perf_counter() - mismatch_start

    dataset = DistanceMatrixDataset(
        features=mass_feature_matrix(bank),
        distance=mismatch,
        metadata={
            "mass_1": bank.mass_1,
            "mass_2": bank.mass_2,
            "approximant": str(gw_config.get("approximant", "IMRPhenomD")),
            "mass_1_min": float(gw_config.get("mass_1_range", [20.0, 40.0])[0]),
            "mass_1_max": float(gw_config.get("mass_1_range", [20.0, 40.0])[1]),
            "mass_2_min": float(gw_config.get("mass_2_range", [10.0, 30.0])[0]),
            "mass_2_max": float(gw_config.get("mass_2_range", [10.0, 30.0])[1]),
            "spin_model": "nonspinning",
            "sample_rate": "not_applicable_frequency_domain",
            "delta_f": float(gw_config.get("delta_f", 1.0 / 16.0)),
            "f_lower": float(gw_config.get("f_lower", 20.0)),
            "f_final": float(gw_config.get("f_final", 512.0)),
            "psd": str(gw_config.get("psd", "aLIGOZeroDetHighPower")),
            "num_waveforms": num_waveforms,
            "pairwise_overlap_count": int(num_waveforms * (num_waveforms - 1) // 2),
            "max_estimated_runtime_minutes": float(
                gw_config.get("max_estimated_runtime_minutes", 30.0)
            ),
            "pycbc_version": _package_version("pycbc"),
            "lalsuite_version": _package_version("lalsuite"),
            "seed": seed,
        },
    )
    save_distance_matrix_dataset(cache_path, dataset)
    return (
        dataset,
        {
            "waveform_generation_time_seconds": waveform_generation_time,
            "pairwise_mismatch_time_seconds": pairwise_mismatch_time,
        },
        False,
    )


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
    pair_dataset = DistanceRegressionDataset(pairs)
    model_cfg = config["model"]
    encoder = MLPEncoder(
        input_dim=int(model_cfg["input_dim"]),
        embedding_dim=int(model_cfg["embedding_dim"]),
        hidden_dims=tuple(int(dim) for dim in model_cfg.get("hidden_dims", [64, 32])),
    )
    train_cfg = config["training"]
    history = train_distance_regression(
        encoder,
        pair_dataset,
        epochs=int(train_cfg["epochs"]),
        batch_size=int(train_cfg["batch_size"]),
        learning_rate=float(train_cfg["learning_rate"]),
    )
    return encoder, history.losses[-1]


def evaluate_encoder(
    config: dict[str, Any],
    dataset: DistanceMatrixDataset,
    encoder: MLPEncoder,
) -> dict[str, float]:
    embeddings = encode_array(encoder, dataset.features)
    latent_distance = pairwise_euclidean(embeddings)
    report: dict[str, float] = {}
    report.update(distance_correlations(latent_distance, dataset.distance))
    report.update(distance_error_summary(latent_distance, dataset.distance))

    k_values = [int(value) for value in config.get("evaluation", {}).get("k_values", [5, 10, 20])]
    max_k = max(k_values)
    index = SklearnNeighborIndex(embeddings)
    _, retrieved = index.query(embeddings, top_k=max_k + 1)
    for top_k in k_values:
        metrics = retrieval_report(dataset.distance, retrieved, top_k=top_k)
        report[f"learned_recall_at_{top_k}"] = metrics[f"recall_at_{top_k}"]
        report[f"learned_recovered_best_match_at_{top_k}"] = metrics[
            "recovered_best_match_at_k"
        ]

    physical_features = physical_parameter_features(dataset.metadata)
    if physical_features is not None:
        physical_distance = pairwise_euclidean(physical_features)
        physical_index = SklearnNeighborIndex(physical_features)
        _, physical_retrieved = physical_index.query(physical_features, top_k=max_k + 1)
        for top_k in k_values:
            metrics = retrieval_report(dataset.distance, physical_retrieved, top_k=top_k)
            report[f"physical_parameter_recall_at_{top_k}"] = metrics[f"recall_at_{top_k}"]
            report[f"physical_parameter_recovered_best_match_at_{top_k}"] = metrics[
                "recovered_best_match_at_k"
            ]
        physical_corr = distance_correlations(physical_distance, dataset.distance)
        report["physical_parameter_pearson"] = physical_corr["pearson"]
        report["physical_parameter_spearman"] = physical_corr["spearman"]
    return report


def physical_parameter_features(metadata: dict[str, Any]) -> np.ndarray | None:
    if "mass_1" not in metadata or "mass_2" not in metadata:
        return None
    features = np.column_stack(
        [
            np.asarray(metadata["mass_1"], dtype=np.float32),
            np.asarray(metadata["mass_2"], dtype=np.float32),
        ]
    )
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True).clip(min=1e-6)
    return ((features - mean) / std).astype(np.float32)


def failed_record(config: dict[str, Any], bank_size: int, exc: Exception) -> dict[str, Any]:
    cache_path = Path(config["gw_data"]["cache_path"])
    return {
        "status": "failed",
        "distance_target": str(config.get("distance_target", "mismatch")),
        "cache_path": str(cache_path),
        "num_waveform_pairs": int(bank_size * (bank_size - 1) // 2),
        "cache_file_size_bytes": cache_path.stat().st_size if cache_path.exists() else 0,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "peak_memory_mb": peak_memory_mb(),
    }


def peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_summary(path: Path, records: list[dict[str, Any]]) -> None:
    completed = [record for record in records if record.get("status") == "completed"]
    payload = {
        "objective": "scaling_validation",
        "num_runs": len(records),
        "num_completed": len(completed),
        "num_failed": len(records) - len(completed),
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    completed = [record for record in records if record.get("status") == "completed"]
    rows = []
    metric_keys = [
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
        "physical_parameter_recall_at_5",
        "physical_parameter_recall_at_10",
        "physical_parameter_recall_at_20",
        "physical_parameter_recovered_best_match_at_5",
        "physical_parameter_recovered_best_match_at_10",
        "physical_parameter_recovered_best_match_at_20",
        "total_runtime_seconds",
        "pairwise_mismatch_time_seconds",
        "training_time_seconds",
        "evaluation_time_seconds",
    ]
    for record in completed:
        for metric in metric_keys:
            if metric in record:
                rows.append(
                    {
                        "bank_size": record["bank_size"],
                        "metric": metric,
                        "value": record[metric],
                    }
                )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bank_size", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def make_plots(output_dir: Path, records: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    completed = sorted(
        [record for record in records if record.get("status") == "completed"],
        key=lambda record: record["bank_size"],
    )
    if not completed:
        return
    plot_dir = ensure_dir(output_dir / "figures")
    plot_single_metric(
        completed,
        "pearson",
        "Pearson correlation",
        plot_dir / "pearson_vs_bank_size.png",
    )
    plot_single_metric(
        completed,
        "spearman",
        "Spearman correlation",
        plot_dir / "spearman_vs_bank_size.png",
    )
    plot_single_metric(completed, "distance_mae", "MAE", plot_dir / "mae_vs_bank_size.png")
    plot_single_metric(completed, "distance_rmse", "RMSE", plot_dir / "rmse_vs_bank_size.png")
    plot_multi_metric(
        completed,
        ["learned_recall_at_5", "learned_recall_at_10", "learned_recall_at_20"],
        "Recall",
        plot_dir / "recall_at_k_vs_bank_size.png",
    )
    plot_multi_metric(
        completed,
        [
            "learned_recovered_best_match_at_5",
            "learned_recovered_best_match_at_10",
            "learned_recovered_best_match_at_20",
        ],
        "Best-match recovery",
        plot_dir / "best_match_recovery_vs_bank_size.png",
    )
    plot_multi_metric(
        completed,
        [
            "physical_parameter_recall_at_5",
            "physical_parameter_recall_at_10",
            "physical_parameter_recall_at_20",
        ],
        "Physical baseline recall",
        plot_dir / "physical_baseline_recall_at_k_vs_bank_size.png",
    )
    plot_multi_metric(
        completed,
        [
            "physical_parameter_recovered_best_match_at_5",
            "physical_parameter_recovered_best_match_at_10",
            "physical_parameter_recovered_best_match_at_20",
        ],
        "Physical baseline best-match recovery",
        plot_dir / "physical_baseline_best_match_recovery_vs_bank_size.png",
    )
    plot_learned_vs_physical_at_k(
        completed,
        learned_prefix="learned_recall_at",
        physical_prefix="physical_parameter_recall_at",
        ylabel="Recall",
        output_path=plot_dir / "learned_vs_physical_recall.png",
    )
    plot_learned_vs_physical_at_k(
        completed,
        learned_prefix="learned_recovered_best_match_at",
        physical_prefix="physical_parameter_recovered_best_match_at",
        ylabel="Best-match recovery",
        output_path=plot_dir / "learned_vs_physical_best_match_recovery.png",
    )
    plot_multi_metric(
        completed,
        [
            "total_runtime_seconds",
            "waveform_generation_time_seconds",
            "pairwise_mismatch_time_seconds",
            "training_time_seconds",
            "evaluation_time_seconds",
        ],
        "Runtime (seconds)",
        plot_dir / "runtime_vs_bank_size.png",
    )
    plot_single_metric(
        completed,
        "pairwise_mismatch_time_seconds",
        "Pairwise mismatch time (seconds)",
        plot_dir / "pairwise_mismatch_time_vs_bank_size.png",
    )
    plt.close("all")


def plot_single_metric(
    records: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plot_multi_metric(records, [metric], ylabel, output_path)


def plot_multi_metric(
    records: list[dict[str, Any]],
    metrics: list[str],
    ylabel: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    plotted = False
    for metric in metrics:
        rows = [record for record in records if metric in record]
        if not rows:
            continue
        ax.plot(
            [record["bank_size"] for record in rows],
            [record[metric] for record in rows],
            marker="o",
            label=metric.replace("_", " "),
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Waveform bank size")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(metrics) > 1:
        ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_learned_vs_physical_at_k(
    records: list[dict[str, Any]],
    learned_prefix: str,
    physical_prefix: str,
    ylabel: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4), sharey=True)
    plotted = False
    for ax, top_k in zip(axes, [5, 10, 20], strict=True):
        learned_metric = f"{learned_prefix}_{top_k}"
        physical_metric = f"{physical_prefix}_{top_k}"
        rows = [
            record for record in records if learned_metric in record and physical_metric in record
        ]
        if not rows:
            continue
        ax.plot(
            [record["bank_size"] for record in rows],
            [record[learned_metric] for record in rows],
            marker="o",
            label="Learned",
        )
        ax.plot(
            [record["bank_size"] for record in rows],
            [record[physical_metric] for record in rows],
            marker="o",
            label=r"Component masses $(m_1,m_2)$",
        )
        ax.set_xscale("log", base=2)
        bank_sizes = [record["bank_size"] for record in rows]
        ax.set_xticks(bank_sizes)
        ax.set_xticklabels([rf"$2^{{{int(round(np.log2(size)))}}}$" for size in bank_sizes])
        ax.set_title(f"K={top_k}")
        ax.set_xlabel("Waveform bank size")
        ax.grid(True, alpha=0.3)
        plotted = True
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=8)
    if not plotted:
        plt.close(fig)
        return
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
