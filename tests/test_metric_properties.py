import numpy as np

from gw_mismatch_learning.evaluation.metric_properties import (
    chordal_from_mismatch,
    metric_property_report,
)


def test_euclidean_distance_matrix_passes_metric_checks_exhaustive() -> None:
    points = np.array([[0.0], [1.0], [3.0]])
    distance = np.abs(points - points.T)

    report = metric_property_report(distance, mode="exhaustive", tol=1e-12)

    assert report["nonnegative"]
    assert report["symmetric"]
    assert report["zero_diagonal"]
    assert report["num_offdiag_near_zero"] == 0
    assert report["num_triangle_tests"] == 27
    assert report["num_triangle_violations"] == 0
    assert report["max_triangle_violation"] == 0.0
    assert report["example_violation"] is None


def test_symmetric_nonnegative_matrix_can_violate_triangle_inequality() -> None:
    distance = np.array(
        [
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 1.0],
            [3.0, 1.0, 0.0],
        ]
    )

    report = metric_property_report(distance, mode="exhaustive", tol=1e-12)

    assert report["nonnegative"]
    assert report["symmetric"]
    assert report["zero_diagonal"]
    assert report["num_triangle_violations"] == 2
    assert report["triangle_violation_fraction"] == 2 / 27
    assert report["max_triangle_violation"] == 1.0
    assert report["example_violation"] is not None


def test_nonzero_diagonal_is_reported() -> None:
    distance = np.array([[0.1, 1.0], [1.0, 0.0]])

    report = metric_property_report(distance, mode="exhaustive", tol=1e-12)

    assert not report["zero_diagonal"]
    assert report["max_diagonal_abs"] == 0.1


def test_asymmetry_is_reported() -> None:
    distance = np.array([[0.0, 1.0], [1.1, 0.0]])

    report = metric_property_report(distance, mode="exhaustive", tol=1e-12)

    assert not report["symmetric"]
    assert np.isclose(report["max_symmetry_error"], 0.1)


def test_chordal_transform_of_mismatch_like_matrix_can_be_checked() -> None:
    mismatch = np.array(
        [
            [0.0, 0.5, 2.0],
            [0.5, 0.0, 0.5],
            [2.0, 0.5, 0.0],
        ]
    )
    chordal = chordal_from_mismatch(mismatch)

    mismatch_report = metric_property_report(mismatch, mode="exhaustive", tol=1e-12)
    chordal_report = metric_property_report(chordal, mode="exhaustive", tol=1e-12)

    assert mismatch_report["num_triangle_violations"] > 0
    assert chordal_report["num_triangle_violations"] == 0
    assert np.allclose(chordal, np.sqrt(2.0 * mismatch))


def test_sampled_mode_uses_requested_number_of_triples() -> None:
    distance = np.array(
        [
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 1.0],
            [3.0, 1.0, 0.0],
        ]
    )

    report = metric_property_report(distance, mode="sampled", num_triples=1000, seed=7)

    assert report["triangle_test_mode"] == "sampled"
    assert report["num_triangle_tests"] == 1000
    assert report["num_triangle_violations"] > 0
    assert report["example_violation"] is not None
