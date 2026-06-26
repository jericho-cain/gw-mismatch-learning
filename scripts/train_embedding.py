from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gw_mismatch_learning.config import load_config
from gw_mismatch_learning.datasets.pairs import PairDataset, sample_pairs
from gw_mismatch_learning.datasets.waveform_dataset import make_mock_waveform_dataset
from gw_mismatch_learning.evaluation.geometry import distance_correlations, pairwise_euclidean
from gw_mismatch_learning.evaluation.retrieval import retrieval_report
from gw_mismatch_learning.models.encoders import MLPEncoder
from gw_mismatch_learning.models.metric_learning import encode_array, train_distance_regression
from gw_mismatch_learning.retrieval.index import SklearnNeighborIndex
from gw_mismatch_learning.utils.seeds import set_seed


def run(config_path: str | Path) -> dict[str, float]:
    config = load_config(config_path)
    seed = int(config.get("seed", 1234))
    set_seed(seed)

    mock_cfg = config["mock_data"]
    dataset = make_mock_waveform_dataset(
        num_samples=int(mock_cfg["num_samples"]),
        input_dim=int(mock_cfg["input_dim"]),
        noise=float(mock_cfg.get("noise", 0.0)),
        seed=seed,
    )

    pair_cfg = config["pairs"]
    pairs = sample_pairs(dataset.features, dataset.mismatch, int(pair_cfg["num_pairs"]), seed=seed)
    pair_dataset = PairDataset(pairs)

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
    report = distance_correlations(latent_distance, dataset.mismatch)

    top_k = int(config.get("evaluation", {}).get("top_k", 5))
    index = SklearnNeighborIndex(embeddings)
    _, retrieved = index.query(embeddings, top_k=top_k)
    report.update(retrieval_report(dataset.mismatch, retrieved, top_k=top_k))
    report["final_train_loss"] = history.losses[-1]
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
