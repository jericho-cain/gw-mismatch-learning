from __future__ import annotations

import numpy as np

from gw_mismatch_learning.retrieval.candidate_reduction import candidate_reduction_factor


def _without_self(neighbors: np.ndarray, row: int) -> np.ndarray:
    return neighbors[neighbors != row]


def top_k_neighbor_overlap(
    true_neighbors: np.ndarray,
    retrieved_neighbors: np.ndarray,
    top_k: int,
    exclude_self: bool = True,
) -> float:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    overlaps = []
    for row, (true_row, retrieved_row) in enumerate(
        zip(true_neighbors, retrieved_neighbors, strict=True)
    ):
        if exclude_self:
            true_row = _without_self(true_row, row)
            retrieved_row = _without_self(retrieved_row, row)
        true_set = set(true_row[:top_k])
        retrieved_set = set(retrieved_row[:top_k])
        overlaps.append(len(true_set & retrieved_set) / top_k)
    return float(np.mean(overlaps))


def recall_at_k(
    true_neighbors: np.ndarray,
    retrieved_neighbors: np.ndarray,
    top_k: int,
    num_relevant: int | None = None,
    exclude_self: bool = True,
) -> float:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if num_relevant is None:
        num_relevant = top_k
    if num_relevant <= 0:
        raise ValueError("num_relevant must be positive")

    recalls = []
    for row, (true_row, retrieved_row) in enumerate(
        zip(true_neighbors, retrieved_neighbors, strict=True)
    ):
        if exclude_self:
            true_row = _without_self(true_row, row)
            retrieved_row = _without_self(retrieved_row, row)
        relevant = set(true_row[:num_relevant])
        retrieved = set(retrieved_row[:top_k])
        recalls.append(len(relevant & retrieved) / len(relevant))
    return float(np.mean(recalls))


def true_neighbors_from_distance(distance_matrix: np.ndarray) -> np.ndarray:
    return np.argsort(distance_matrix, axis=1)


def recovered_best_match_at_k(
    mismatch_matrix: np.ndarray,
    retrieved_neighbors: np.ndarray,
    top_k: int,
    exclude_self: bool = True,
) -> float:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    recovered = []
    true_neighbors = true_neighbors_from_distance(mismatch_matrix)
    for row, retrieved in enumerate(retrieved_neighbors):
        true_row = true_neighbors[row]
        if exclude_self:
            true_row = _without_self(true_row, row)
            retrieved = _without_self(retrieved, row)
        best_true = true_row[0]
        recovered.append(best_true in set(retrieved[:top_k]))
    return float(np.mean(recovered))


def retrieval_report(
    mismatch_matrix: np.ndarray,
    retrieved_neighbors: np.ndarray,
    top_k: int,
    exclude_self: bool = True,
) -> dict[str, float]:
    true_neighbors = true_neighbors_from_distance(mismatch_matrix)
    return {
        f"recall_at_{top_k}": recall_at_k(
            true_neighbors,
            retrieved_neighbors,
            top_k,
            exclude_self=exclude_self,
        ),
        "top_k_neighbor_overlap": top_k_neighbor_overlap(
            true_neighbors,
            retrieved_neighbors,
            top_k,
            exclude_self=exclude_self,
        ),
        "recovered_best_match_at_k": recovered_best_match_at_k(
            mismatch_matrix,
            retrieved_neighbors,
            top_k,
            exclude_self=exclude_self,
        ),
        "candidate_reduction_factor": candidate_reduction_factor(mismatch_matrix.shape[0], top_k),
    }
