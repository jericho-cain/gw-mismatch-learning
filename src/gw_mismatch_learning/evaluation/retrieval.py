from __future__ import annotations

import numpy as np

from gw_mismatch_learning.retrieval.candidate_reduction import candidate_reduction_factor


def top_k_neighbor_overlap(
    true_neighbors: np.ndarray,
    retrieved_neighbors: np.ndarray,
    top_k: int,
) -> float:
    overlaps = []
    for true_row, retrieved_row in zip(true_neighbors, retrieved_neighbors, strict=True):
        true_set = set(true_row[:top_k])
        retrieved_set = set(retrieved_row[:top_k])
        overlaps.append(len(true_set & retrieved_set) / top_k)
    return float(np.mean(overlaps))


def true_neighbors_from_distance(distance_matrix: np.ndarray) -> np.ndarray:
    return np.argsort(distance_matrix, axis=1)


def recovered_best_match_at_k(
    mismatch_matrix: np.ndarray,
    retrieved_neighbors: np.ndarray,
    top_k: int,
) -> float:
    recovered = []
    true_neighbors = true_neighbors_from_distance(mismatch_matrix)
    for row, retrieved in enumerate(retrieved_neighbors):
        if true_neighbors[row, 0] == row:
            best_true = true_neighbors[row, 1]
        else:
            best_true = true_neighbors[row, 0]
        recovered.append(best_true in set(retrieved[:top_k]))
    return float(np.mean(recovered))


def retrieval_report(
    mismatch_matrix: np.ndarray,
    retrieved_neighbors: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    true_neighbors = true_neighbors_from_distance(mismatch_matrix)
    return {
        "top_k_neighbor_overlap": top_k_neighbor_overlap(
            true_neighbors,
            retrieved_neighbors,
            top_k,
        ),
        "recovered_best_match_at_k": recovered_best_match_at_k(
            mismatch_matrix,
            retrieved_neighbors,
            top_k,
        ),
        "candidate_reduction_factor": candidate_reduction_factor(mismatch_matrix.shape[0], top_k),
    }
