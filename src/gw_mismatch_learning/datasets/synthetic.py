from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SyntheticMetric = Literal["euclidean", "squared_euclidean", "cosine"]


@dataclass(frozen=True)
class SyntheticDistanceDataset:
    """Synthetic vectors with a known pairwise distance matrix."""

    features: np.ndarray
    distance: np.ndarray
    metric: SyntheticMetric


def make_synthetic_distance_dataset(
    num_samples: int = 128,
    input_dim: int = 128,
    metric: SyntheticMetric = "cosine",
    noise: float = 0.0,
    seed: int | None = None,
) -> SyntheticDistanceDataset:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(num_samples, input_dim)).astype(np.float32)
    distance = pairwise_feature_distance(features, metric=metric)

    if noise > 0:
        jitter = rng.normal(scale=noise, size=distance.shape).astype(np.float32)
        distance = np.clip(distance + jitter, 0.0, None)
        distance = 0.5 * (distance + distance.T)
        np.fill_diagonal(distance, 0.0)

    validate_distance_matrix(distance)
    return SyntheticDistanceDataset(features=features, distance=distance, metric=metric)


def pairwise_feature_distance(
    features: np.ndarray,
    metric: SyntheticMetric = "cosine",
) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")

    if metric == "cosine":
        normalized = features / np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-8)
        similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
        distance = 0.5 * (1.0 - similarity)
    elif metric == "euclidean":
        delta = features[:, None, :] - features[None, :, :]
        distance = np.linalg.norm(delta, axis=-1)
    elif metric == "squared_euclidean":
        delta = features[:, None, :] - features[None, :, :]
        distance = np.sum(delta**2, axis=-1)
    else:
        raise ValueError(f"Unsupported synthetic metric: {metric}")

    distance = distance.astype(np.float32)
    distance = 0.5 * (distance + distance.T)
    np.fill_diagonal(distance, 0.0)
    return distance


def validate_distance_matrix(distance: np.ndarray) -> None:
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance matrix must be square")
    if not np.all(np.isfinite(distance)):
        raise ValueError("distance matrix must contain finite values")
    if np.any(distance < -1e-7):
        raise ValueError("distance matrix must be non-negative")
    if not np.allclose(distance, distance.T, atol=1e-6):
        raise ValueError("distance matrix must be symmetric")
    if not np.allclose(np.diag(distance), 0.0, atol=1e-6):
        raise ValueError("distance matrix must have a zero diagonal")
