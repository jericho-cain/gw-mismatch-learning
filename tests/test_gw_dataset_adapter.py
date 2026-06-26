import json

import numpy as np
import pytest
from scripts.train_embedding import run

from gw_mismatch_learning.datasets.gw import (
    DistanceMatrixDataset,
    load_distance_matrix_dataset,
    load_or_create_gw_mismatch_dataset,
    save_distance_matrix_dataset,
)
from gw_mismatch_learning.waveforms.generate import generate_pycbc_fd_waveform


def test_distance_matrix_dataset_round_trip(tmp_path) -> None:
    dataset = DistanceMatrixDataset(
        features=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        distance=np.array([[0.0, 0.2], [0.2, 0.0]], dtype=np.float32),
        metadata={
            "mass_1": np.array([30.0, 35.0], dtype=np.float32),
            "approximant": "IMRPhenomD",
            "f_lower": 20.0,
        },
    )
    path = tmp_path / "dataset.h5"

    save_distance_matrix_dataset(path, dataset)
    loaded = load_distance_matrix_dataset(path)

    np.testing.assert_allclose(loaded.features, dataset.features)
    np.testing.assert_allclose(loaded.distance, dataset.distance)
    np.testing.assert_allclose(loaded.metadata["mass_1"], dataset.metadata["mass_1"])
    assert loaded.metadata["approximant"] == "IMRPhenomD"
    assert loaded.metadata["f_lower"] == 20.0


def test_load_or_create_uses_cache_without_pycbc(tmp_path) -> None:
    path = tmp_path / "cached.h5"
    dataset = DistanceMatrixDataset(
        features=np.array([[0.0], [1.0]], dtype=np.float32),
        distance=np.array([[0.0, 0.1], [0.1, 0.0]], dtype=np.float32),
        metadata={},
    )
    save_distance_matrix_dataset(path, dataset)

    loaded = load_or_create_gw_mismatch_dataset({"cache_path": str(path)}, seed=1)
    np.testing.assert_allclose(loaded.distance, dataset.distance)


def test_pycbc_waveform_generation_is_lazy() -> None:
    pytest.importorskip("pycbc")
    waveform = generate_pycbc_fd_waveform(mass1=30.0, mass2=20.0)
    assert len(waveform) > 0


def test_training_pipeline_accepts_cached_gw_distance_dataset(tmp_path) -> None:
    features = np.array(
        [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0], [2.0, 2.0], [2.1, 2.0]],
        dtype=np.float32,
    )
    distance = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=-1)
    cache_path = tmp_path / "gw_cache.h5"
    save_distance_matrix_dataset(
        cache_path,
        DistanceMatrixDataset(
            features=features,
            distance=distance,
            metadata={
                "mass_1": np.array([20.0, 21.0, 30.0, 31.0, 40.0, 41.0], dtype=np.float32),
                "mass_2": np.array([10.0, 10.5, 15.0, 15.5, 20.0, 20.5], dtype=np.float32),
            },
        ),
    )
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 9",
                "gw_data:",
                f"  cache_path: {cache_path}",
                "pairs:",
                "  num_pairs: 24",
                "model:",
                "  input_dim: 2",
                "  embedding_dim: 2",
                "  hidden_dims: [8]",
                "training:",
                "  epochs: 1",
                "  batch_size: 12",
                "  learning_rate: 0.001",
                "evaluation:",
                "  top_k: 2",
                "outputs:",
                "  save_plots: false",
                f"  plot_dir: {output_dir}",
            ]
        ),
        encoding="utf-8",
    )

    report = run(config_path)
    assert "recall_at_2" in report
    assert "physical_parameter_recall_at_2" in report
    assert report["candidate_reduction_factor"] == 3.0
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["data_file_path"] == str(cache_path)
    assert "physical_parameter_recall_at_2" in metrics["metrics"]
