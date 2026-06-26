from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(num_samples, input_dim)).astype(np.float32)
    normalized = features / np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-8)
    cosine_similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    mismatch = 0.5 * (1.0 - cosine_similarity)
    if noise > 0:
        jitter = rng.normal(scale=noise, size=mismatch.shape)
        mismatch = np.clip(mismatch + jitter, 0.0, None)
        mismatch = 0.5 * (mismatch + mismatch.T)
        np.fill_diagonal(mismatch, 0.0)
    return MockWaveformDataset(features=features, mismatch=mismatch.astype(np.float32))
