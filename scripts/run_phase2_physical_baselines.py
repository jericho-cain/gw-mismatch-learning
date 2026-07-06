from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gw_mismatch_learning.config import load_config
from gw_mismatch_learning.datasets.gw import DistanceMatrixDataset, load_distance_matrix_dataset
from gw_mismatch_learning.evaluation.geometry import distance_correlations, pairwise_euclidean
from gw_mismatch_learning.evaluation.retrieval import retrieval_report
from gw_mismatch_learning.retrieval.index import SklearnNeighborIndex
from gw_mismatch_learning.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_physical_baselines.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_dir(config.get("output_dir", "outputs/phase2_physical_baselines"))
    records = run_phase2_physical_baselines(config)
    write_records(output_dir / "runs.csv", records)
    write_summary(output_dir / "summary.json", config, records)
    write_summary_csv(output_dir / "summary.csv", records)
    make_plots(output_dir, records)


def run_phase2_physical_baselines(config: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for bank_size in [int(value) for value in config["bank_sizes"]]:
        cache_path = Path(str(config["cache_template"]).format(bank_size=bank_size))
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing validated mismatch cache for bank_size={bank_size}: {cache_path}"
            )
        dataset = load_distance_matrix_dataset(cache_path)
        records.append(learned_reference_record(config, bank_size, cache_path))
        for coordinate_config in config["coordinate_systems"]:
            records.append(
                evaluate_coordinate_system(
                    config,
                    bank_size,
                    cache_path,
                    dataset,
                    coordinate_config,
                )
            )
    return records


def learned_reference_record(
    config: dict[str, Any],
    bank_size: int,
    cache_path: Path,
) -> dict[str, Any]:
    phase1_path = (
        Path(config.get("phase1_output_dir", "outputs/scaling_validation"))
        / f"bank_size_{bank_size}"
        / "metrics.json"
    )
    if not phase1_path.exists():
        raise FileNotFoundError(
            f"Missing frozen Phase I learned metrics for bank_size={bank_size}: {phase1_path}"
        )
    metrics = json.loads(phase1_path.read_text(encoding="utf-8"))
    record = base_record(
        config=config,
        bank_size=bank_size,
        cache_path=cache_path,
        coordinate_system="learned_latent",
        coordinate_label="Learned latent",
        feature_names=["learned_latent_coordinates"],
        dimensionality=int(metrics.get("model_embedding_dim", 4)),
        reference_type="frozen_phase_i_learned_representation",
    )
    record["source_metrics_path"] = str(phase1_path)
    record["pearson"] = metrics.get("pearson")
    record["spearman"] = metrics.get("spearman")
    for top_k in k_values(config):
        record[f"recall_at_{top_k}"] = metrics[f"learned_recall_at_{top_k}"]
        record[f"recovered_best_match_at_{top_k}"] = metrics[
            f"learned_recovered_best_match_at_{top_k}"
        ]
    return record


def evaluate_coordinate_system(
    config: dict[str, Any],
    bank_size: int,
    cache_path: Path,
    dataset: DistanceMatrixDataset,
    coordinate_config: dict[str, Any],
) -> dict[str, Any]:
    feature_names = [str(name) for name in coordinate_config["feature_names"]]
    features = standardized_coordinate_features(dataset, feature_names)
    record = base_record(
        config=config,
        bank_size=bank_size,
        cache_path=cache_path,
        coordinate_system=str(coordinate_config["name"]),
        coordinate_label=str(coordinate_config.get("label", coordinate_config["name"])),
        feature_names=feature_names,
        dimensionality=features.shape[1],
        reference_type="fixed_physical_coordinate_system",
    )

    max_k = max(k_values(config))
    index = SklearnNeighborIndex(features)
    _, retrieved = index.query(features, top_k=max_k + 1)
    for top_k in k_values(config):
        metrics = retrieval_report(dataset.distance, retrieved, top_k=top_k)
        record[f"recall_at_{top_k}"] = metrics[f"recall_at_{top_k}"]
        record[f"recovered_best_match_at_{top_k}"] = metrics["recovered_best_match_at_k"]

    coordinate_distance = pairwise_euclidean(features)
    record.update(distance_correlations(coordinate_distance, dataset.distance))
    return record


def standardized_coordinate_features(
    dataset: DistanceMatrixDataset,
    feature_names: list[str],
) -> np.ndarray:
    raw = raw_coordinate_features(dataset, feature_names)
    return zscore(raw)


def raw_coordinate_features(dataset: DistanceMatrixDataset, feature_names: list[str]) -> np.ndarray:
    mass_1 = np.asarray(dataset.metadata["mass_1"], dtype=np.float32)
    mass_2 = np.asarray(dataset.metadata["mass_2"], dtype=np.float32)
    total_mass = mass_1 + mass_2
    eta = (mass_1 * mass_2) / np.square(total_mass)
    chirp_mass = np.power(mass_1 * mass_2, 3.0 / 5.0) / np.power(total_mass, 1.0 / 5.0)
    q = mass_2 / mass_1
    values = {
        "m1": mass_1,
        "m2": mass_2,
        "M": total_mass,
        "eta": eta,
        "chirp_mass": chirp_mass,
        "q": q,
    }
    missing = sorted(set(feature_names) - set(values))
    if missing:
        raise KeyError(f"Unsupported coordinate features: {missing}")
    return np.column_stack([values[name] for name in feature_names]).astype(np.float32)


def zscore(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True).clip(min=1e-6)
    return ((features - mean) / std).astype(np.float32)


def base_record(
    config: dict[str, Any],
    bank_size: int,
    cache_path: Path,
    coordinate_system: str,
    coordinate_label: str,
    feature_names: list[str],
    dimensionality: int,
    reference_type: str,
) -> dict[str, Any]:
    return {
        "bank_size": bank_size,
        "num_waveform_pairs": int(bank_size * (bank_size - 1) // 2),
        "cache_path": str(cache_path),
        "coordinate_system": coordinate_system,
        "coordinate_label": coordinate_label,
        "feature_names": ",".join(feature_names),
        "dimensionality": int(dimensionality),
        "standardization_method": str(config.get("standardization", "zscore_per_bank")),
        "q_convention": str(config.get("q_convention", "")),
        "reference_type": reference_type,
        "status": "completed",
    }


def k_values(config: dict[str, Any]) -> list[int]:
    return [int(value) for value in config.get("k_values", [5, 10, 20])]


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_summary(path: Path, config: dict[str, Any], records: list[dict[str, Any]]) -> None:
    payload = {
        "objective": "phase2_physical_baselines",
        "num_runs": len(records),
        "num_completed": sum(record.get("status") == "completed" for record in records),
        "num_failed": sum(record.get("status") != "completed" for record in records),
        "bank_sizes": [int(value) for value in config["bank_sizes"]],
        "k_values": k_values(config),
        "standardization_method": str(config.get("standardization", "zscore_per_bank")),
        "q_convention": str(config.get("q_convention", "")),
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    metric_names = [
        "pearson",
        "spearman",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "recovered_best_match_at_5",
        "recovered_best_match_at_10",
        "recovered_best_match_at_20",
    ]
    fieldnames = [
        "bank_size",
        "coordinate_system",
        "coordinate_label",
        "feature_names",
        "dimensionality",
        "standardization_method",
        "q_convention",
        *metric_names,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def make_plots(output_dir: Path, records: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    plot_dir = ensure_dir(output_dir / "figures")
    plot_grouped_bars(
        records,
        bank_size=8192,
        metrics=["recall_at_5", "recall_at_10", "recall_at_20"],
        ylabel="Recall",
        output_path=plot_dir / "recall_at_k_by_coordinate_system_8192.png",
    )
    plot_grouped_bars(
        records,
        bank_size=8192,
        metrics=[
            "recovered_best_match_at_5",
            "recovered_best_match_at_10",
            "recovered_best_match_at_20",
        ],
        ylabel="Best-match recovery",
        output_path=plot_dir / "best_match_recovery_by_coordinate_system_8192.png",
    )
    plot_metric_vs_bank_size(
        records,
        metric="recall_at_10",
        ylabel="Recall@10",
        output_path=plot_dir / "recall_at_10_vs_bank_size_by_coordinate_system.png",
    )
    plot_metric_vs_bank_size(
        records,
        metric="recovered_best_match_at_10",
        ylabel="Best-match recovery@10",
        output_path=plot_dir / "best_match_recovery_at_10_vs_bank_size_by_coordinate_system.png",
    )
    plot_learned_vs_five_feature(records, plot_dir / "learned_latent_vs_five_feature_input.png")
    plt.close("all")


def plot_grouped_bars(
    records: list[dict[str, Any]],
    bank_size: int,
    metrics: list[str],
    ylabel: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    rows = [record for record in records if record["bank_size"] == bank_size]
    x = np.arange(len(rows))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for offset, metric in enumerate(metrics):
        ax.bar(
            x + (offset - 1) * width,
            [float(row[metric]) for row in rows],
            width=width,
            label=metric.replace("recovered_best_match", "best").replace("_", " "),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([row["coordinate_label"] for row in rows], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_bank_size(
    records: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for coordinate_system in coordinate_system_order(records):
        rows = sorted(
            [record for record in records if record["coordinate_system"] == coordinate_system],
            key=lambda record: record["bank_size"],
        )
        if not rows:
            continue
        ax.plot(
            [record["bank_size"] for record in rows],
            [float(record[metric]) for record in rows],
            marker="o",
            label=rows[0]["coordinate_label"],
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Waveform bank size")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_learned_vs_five_feature(records: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    comparisons = [
        ("recall_at_10", "Recall@10"),
        ("recovered_best_match_at_10", "Best-match recovery@10"),
    ]
    for ax, (metric, ylabel) in zip(axes, comparisons, strict=True):
        for coordinate_system in ["learned_latent", "five_feature_input"]:
            rows = sorted(
                [
                    record
                    for record in records
                    if record["coordinate_system"] == coordinate_system
                ],
                key=lambda record: record["bank_size"],
            )
            ax.plot(
                [record["bank_size"] for record in rows],
                [float(record[metric]) for record in rows],
                marker="o",
                label=rows[0]["coordinate_label"],
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Waveform bank size")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
    axes[-1].legend(fontsize=8)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def coordinate_system_order(records: list[dict[str, Any]]) -> list[str]:
    order = []
    for record in records:
        name = str(record["coordinate_system"])
        if name not in order:
            order.append(name)
    return order


if __name__ == "__main__":
    main()
