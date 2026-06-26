import numpy as np
import pytest

from gw_mismatch_learning.datasets.synthetic import (
    make_synthetic_distance_dataset,
    pairwise_feature_distance,
    validate_distance_matrix,
)


def test_synthetic_distance_dataset_defaults_to_square_distance_matrix() -> None:
    dataset = make_synthetic_distance_dataset(
        num_samples=16,
        input_dim=128,
        metric="cosine",
        seed=4,
    )
    assert dataset.features.shape == (16, 128)
    assert dataset.distance.shape == (16, 16)
    validate_distance_matrix(dataset.distance)


def test_pairwise_feature_distance_supports_euclidean() -> None:
    features = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    distance = pairwise_feature_distance(features, metric="euclidean")
    np.testing.assert_allclose(distance, [[0.0, 5.0], [5.0, 0.0]])


def test_validate_distance_matrix_rejects_non_symmetric_input() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        validate_distance_matrix(np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32))
