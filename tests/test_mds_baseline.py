from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scripts.run_mds_baseline import (
    coordinates,
    double_center_squared_distances,
    full_eigensystem,
    partial_eigenpairs,
    spectral_diagnostics,
)


def test_double_centering_matches_explicit_centering_matrix() -> None:
    distance = np.asarray([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    center = np.eye(3) - np.ones((3, 3)) / 3
    expected = -0.5 * center @ (distance**2) @ center
    np.testing.assert_allclose(double_center_squared_distances(distance), expected, atol=1e-14)


def test_coordinates_reconstruct_euclidean_toy_distances() -> None:
    points = np.asarray([[0, 0], [1, 0], [0, 2], [1, 2]], dtype=float)
    distance = squareform(pdist(points))
    values, vectors = full_eigensystem(double_center_squared_distances(distance))
    reconstructed = squareform(pdist(coordinates(values, vectors, 2)))
    np.testing.assert_allclose(reconstructed, distance, atol=1e-7)


def test_negative_eigenvalues_are_reported() -> None:
    non_euclidean = np.asarray([[0, 1, 1], [1, 0, 3], [1, 3, 0]], dtype=float)
    values, _ = full_eigensystem(double_center_squared_distances(non_euclidean))
    diagnostics = spectral_diagnostics(values, 1e-12)
    assert diagnostics["num_negative"] > 0
    assert diagnostics["total_absolute_negative_spectral_mass"] > 0
    assert diagnostics["most_negative_eigenvalue"] < 0


def test_partial_eigensolver_agrees_with_full_on_toy_problem() -> None:
    rng = np.random.default_rng(42)
    distance = squareform(pdist(rng.normal(size=(40, 6))))
    gram = double_center_squared_distances(distance)
    full_values, _ = full_eigensystem(gram)
    partial_values, _ = partial_eigenpairs(gram, 8, tolerance=1e-11)
    np.testing.assert_allclose(partial_values, full_values[:8], rtol=1e-8, atol=1e-9)
