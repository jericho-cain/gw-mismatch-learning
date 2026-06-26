from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def fit_nearest_neighbors(embeddings: np.ndarray, metric: str = "euclidean") -> NearestNeighbors:
    index = NearestNeighbors(metric=metric)
    index.fit(embeddings)
    return index


def query_nearest_neighbors(
    index: NearestNeighbors,
    queries: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    distances, indices = index.kneighbors(queries, n_neighbors=top_k)
    return distances, indices
