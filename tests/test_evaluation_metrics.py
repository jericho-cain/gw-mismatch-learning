import numpy as np

from gw_mismatch_learning.evaluation.geometry import distance_error_summary
from gw_mismatch_learning.evaluation.retrieval import (
    recall_at_k,
    top_k_neighbor_overlap,
    true_neighbors_from_distance,
)


def test_distance_error_summary_reports_zero_for_identical_distances() -> None:
    distance = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    report = distance_error_summary(distance, distance)
    assert report["distance_mae"] == 0.0
    assert report["distance_rmse"] == 0.0


def test_recall_at_k_excludes_self_neighbors() -> None:
    distance = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.1, 0.0, 0.3],
            [0.2, 0.3, 0.0],
        ],
        dtype=np.float32,
    )
    true_neighbors = true_neighbors_from_distance(distance)
    retrieved_neighbors = true_neighbors.copy()

    assert recall_at_k(true_neighbors, retrieved_neighbors, top_k=1) == 1.0
    assert top_k_neighbor_overlap(true_neighbors, retrieved_neighbors, top_k=1) == 1.0
