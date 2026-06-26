from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gw_mismatch_learning.datasets.synthetic import make_synthetic_distance_dataset


@dataclass(frozen=True)
class MockWaveformDataset:
    """Small synthetic dataset for developing mismatch-learning code."""

    features: np.ndarray
    mismatch: np.ndarray


def make_mock_waveform_dataset(
    num_samples: int = 64,
    input_dim: int = 16,
    noise: float = 0.01,
    seed: int | None = None,
) -> MockWaveformDataset:
    synthetic = make_synthetic_distance_dataset(
        num_samples=num_samples,
        input_dim=input_dim,
        metric="cosine",
        noise=noise,
        seed=seed,
    )
    return MockWaveformDataset(features=synthetic.features, mismatch=synthetic.distance)
