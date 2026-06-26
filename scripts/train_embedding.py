from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gw_mismatch_learning.config import load_config
from gw_mismatch_learning.datasets.distance_regression import DistanceRegressionDataset
from gw_mismatch_learning.datasets.gw import load_or_create_gw_mismatch_dataset
from gw_mismatch_learning.datasets.pairs import sample_pairs
from gw_mismatch_learning.datasets.synthetic import make_synthetic_distance_dataset
from gw_mismatch_learning.evaluation.geometry import (
    distance_correlations,
    distance_error_summary,
    pairwise_euclidean,
)
from gw_mismatch_learning.evaluation.plots import plot_distance_scatter, plot_retrieval_rank_heatmap
from gw_mismatch_learning.evaluation.retrieval import retrieval_report, true_neighbors_from_distance
from gw_mismatch_learning.models.encoders import MLPEncoder
from gw_mismatch_learning.models.metric_learning import encode_array, train_distance_regression
from gw_mismatch_learning.retrieval.index import SklearnNeighborIndex
from gw_mismatch_learning.utils.io import ensure_dir
from gw_mismatch_learning.utils.seeds import set_seed


def load_experiment_dataset(config: dict, seed: int):
    if "gw_data" in config:
        return load_or_create_gw_mismatch_dataset(config["gw_data"], seed=seed)

    data_cfg = config.get("synthetic_data", config.get("mock_data", {}))
    return make_synthetic_distance_dataset(
        num_samples=int(data_cfg["num_samples"]),
        input_dim=int(data_cfg["input_dim"]),
        metric=data_cfg.get("metric", "cosine"),
        noise=float(data_cfg.get("noise", 0.0)),
        seed=seed,
    )


def evaluation_k_values(config: dict) -> list[int]:
    evaluation_cfg = config.get("evaluation", {})
    if "k_values" in evaluation_cfg:
        return [int(value) for value in evaluation_cfg["k_values"]]
    return [int(evaluation_cfg.get("top_k", 5))]


def physical_parameter_features(metadata: dict) -> np.ndarray | None:
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


def add_retrieval_metrics(
    report: dict[str, float],
    prefix: str,
    target_distance: np.ndarray,
    retrieved: np.ndarray,
    k_values: list[int],
    primary_k: int,
) -> None:
    for top_k in k_values:
        metrics = retrieval_report(target_distance, retrieved, top_k=top_k)
        if prefix:
            for key, value in metrics.items():
                if key.startswith("recall_at_"):
                    metric_key = key
                elif key == "recovered_best_match_at_k":
                    metric_key = f"recovered_best_match_at_{top_k}"
                else:
                    metric_key = f"{key}_at_{top_k}"
                report[f"{prefix}_{metric_key}"] = value
        if top_k == primary_k and not prefix:
            report.update(metrics)


def save_metrics_json(
    report: dict[str, float],
    config_path: str | Path,
    data_file_path: str | None,
    output_cfg: dict,
) -> None:
    if not bool(output_cfg.get("save_metrics", True)):
        return

    output_dir = ensure_dir(output_cfg.get("plot_dir", "outputs/smoke"))
    metrics_path = Path(output_cfg.get("metrics_path", output_dir / "metrics.json"))
    ensure_dir(metrics_path.parent)
    payload = {
        "config_path": str(config_path),
        "data_file_path": data_file_path,
        "metrics": {key: float(value) for key, value in report.items()},
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run(config_path: str | Path) -> dict[str, float]:
    config = load_config(config_path)
    seed = int(config.get("seed", 1234))
    set_seed(seed)

    dataset = load_experiment_dataset(config, seed=seed)

    pair_cfg = config["pairs"]
    pairs = sample_pairs(dataset.features, dataset.distance, int(pair_cfg["num_pairs"]), seed=seed)
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

    embeddings = encode_array(encoder, dataset.features)
    latent_distance = pairwise_euclidean(embeddings)
    report = distance_correlations(latent_distance, dataset.distance)
    report.update(distance_error_summary(latent_distance, dataset.distance))

    k_values = evaluation_k_values(config)
    top_k = k_values[0]
    max_k = max(k_values)
    index = SklearnNeighborIndex(embeddings)
    _, retrieved = index.query(embeddings, top_k=max_k + 1)
    add_retrieval_metrics(report, "", dataset.distance, retrieved, k_values, primary_k=top_k)
    add_retrieval_metrics(
        report,
        "learned",
        dataset.distance,
        retrieved,
        k_values,
        primary_k=top_k,
    )

    physical_features = physical_parameter_features(getattr(dataset, "metadata", {}))
    if physical_features is not None:
        physical_distance = pairwise_euclidean(physical_features)
        physical_index = SklearnNeighborIndex(physical_features)
        _, physical_retrieved = physical_index.query(physical_features, top_k=max_k + 1)
        add_retrieval_metrics(
            report,
            "physical_parameter",
            dataset.distance,
            physical_retrieved,
            k_values,
            primary_k=top_k,
        )
        physical_corr = distance_correlations(physical_distance, dataset.distance)
        report["physical_parameter_pearson"] = physical_corr["pearson"]
        report["physical_parameter_spearman"] = physical_corr["spearman"]

    report["final_train_loss"] = history.losses[-1]

    output_cfg = config.get("outputs", {})
    data_file_path = config.get("gw_data", {}).get("cache_path")
    if bool(output_cfg.get("save_plots", False)):
        plot_dir = ensure_dir(output_cfg.get("plot_dir", "outputs/smoke"))
        scatter_fig, _ = plot_distance_scatter(latent_distance, dataset.distance)
        scatter_fig.savefig(plot_dir / "distance_scatter.png", dpi=150, bbox_inches="tight")
        true_neighbors = true_neighbors_from_distance(dataset.distance)
        retrieval_fig, _ = plot_retrieval_rank_heatmap(true_neighbors, retrieved, top_k=max_k)
        retrieval_fig.savefig(plot_dir / "retrieval_heatmap.png", dpi=150, bbox_inches="tight")
        report["plots_generated"] = 2.0
    if output_cfg:
        save_metrics_json(report, config_path, data_file_path, output_cfg)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()
    report = run(args.config)
    for key, value in report.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
