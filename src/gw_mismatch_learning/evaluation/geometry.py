from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr


def pairwise_euclidean(embeddings: np.ndarray) -> np.ndarray:
    return squareform(pdist(embeddings, metric="euclidean")).astype(np.float32)


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    return matrix[mask]


def distance_correlations(
    latent_distance: np.ndarray,
    target_distance: np.ndarray,
) -> dict[str, float]:
    if latent_distance.shape != target_distance.shape:
        raise ValueError("latent_distance and target_distance must have matching shapes")
    latent = upper_triangle_values(latent_distance)
    target = upper_triangle_values(target_distance)
    pearson = pearsonr(latent, target).statistic
    spearman = spearmanr(latent, target).statistic
    return {"pearson": float(pearson), "spearman": float(spearman)}


def distance_error_summary(
    latent_distance: np.ndarray,
    target_distance: np.ndarray,
) -> dict[str, float]:
    if latent_distance.shape != target_distance.shape:
        raise ValueError("latent_distance and target_distance must have matching shapes")
    error = upper_triangle_values(latent_distance - target_distance)
    absolute_error = np.abs(error)
    return {
        "distance_mae": float(np.mean(absolute_error)),
        "distance_rmse": float(np.sqrt(np.mean(error**2))),
    }
