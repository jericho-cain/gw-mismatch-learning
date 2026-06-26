from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr


def pairwise_euclidean(embeddings: np.ndarray) -> np.ndarray:
    return squareform(pdist(embeddings, metric="euclidean")).astype(np.float32)


def distance_correlations(
    latent_distance: np.ndarray,
    target_distance: np.ndarray,
) -> dict[str, float]:
    mask = np.triu(np.ones_like(target_distance, dtype=bool), k=1)
    latent = latent_distance[mask]
    target = target_distance[mask]
    pearson = pearsonr(latent, target).statistic
    spearman = spearmanr(latent, target).statistic
    return {"pearson": float(pearson), "spearman": float(spearman)}
