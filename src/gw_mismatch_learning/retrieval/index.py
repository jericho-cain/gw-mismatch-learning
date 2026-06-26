from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

from gw_mismatch_learning.retrieval.nearest_neighbors import (
    fit_nearest_neighbors,
    query_nearest_neighbors,
)


@dataclass
class SklearnNeighborIndex:
    embeddings: np.ndarray
    metric: str = "euclidean"

    def __post_init__(self) -> None:
        self.index: NearestNeighbors = fit_nearest_neighbors(self.embeddings, metric=self.metric)

    def query(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        return query_nearest_neighbors(self.index, queries, top_k=top_k)
