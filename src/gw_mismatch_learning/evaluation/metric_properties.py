from __future__ import annotations

from typing import Literal

import numpy as np

TriangleMode = Literal["exhaustive", "sampled"]


def chordal_from_mismatch(mismatch: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(0.0, 2.0 * mismatch))


def metric_property_report(
    distance: np.ndarray,
    *,
    mode: TriangleMode = "exhaustive",
    num_triples: int = 1_000_000,
    seed: int = 123,
    tol: float = 1e-8,
) -> dict[str, object]:
    matrix = np.asarray(distance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance must be a square matrix")
    if mode not in {"exhaustive", "sampled"}:
        raise ValueError("mode must be 'exhaustive' or 'sampled'")
    if num_triples < 1:
        raise ValueError("num_triples must be positive")
    if tol < 0:
        raise ValueError("tol must be nonnegative")

    n = matrix.shape[0]
    diagonal = np.diag(matrix)
    offdiag_mask = ~np.eye(n, dtype=bool)
    symmetry_error = np.abs(matrix - matrix.T)
    triangle_report = (
        _exhaustive_triangle_report(matrix, tol=tol)
        if mode == "exhaustive"
        else _sampled_triangle_report(matrix, num_triples=num_triples, seed=seed, tol=tol)
    )

    return {
        "n": int(n),
        "nonnegative": bool(float(np.min(matrix)) >= -tol),
        "min_value": float(np.min(matrix)),
        "max_value": float(np.max(matrix)),
        "symmetric": bool(float(np.max(symmetry_error)) <= tol),
        "max_symmetry_error": float(np.max(symmetry_error)),
        "zero_diagonal": bool(float(np.max(np.abs(diagonal))) <= tol),
        "max_diagonal_abs": float(np.max(np.abs(diagonal))),
        "num_offdiag_near_zero": int(np.count_nonzero(np.abs(matrix[offdiag_mask]) <= tol)),
        **triangle_report,
    }


def _exhaustive_triangle_report(matrix: np.ndarray, *, tol: float) -> dict[str, object]:
    n = matrix.shape[0]
    num_tests = n**3
    num_violations = 0
    max_violation = 0.0
    positive_violation_sum = 0.0
    example: list[object] | None = None

    for j in range(n):
        rhs = matrix[:, j, None] + matrix[j, None, :]
        violations = matrix - rhs
        positive = violations > tol
        if not np.any(positive):
            continue

        positive_values = violations[positive]
        count = int(positive_values.size)
        num_violations += count
        positive_violation_sum += float(np.sum(positive_values))

        local_max = float(np.max(positive_values))
        if local_max > max_violation:
            max_violation = local_max

        if example is None:
            i, k = [int(value) for value in np.argwhere(positive)[0]]
            lhs = float(matrix[i, k])
            rhs_value = float(matrix[i, j] + matrix[j, k])
            example = [i, j, k, lhs, rhs_value, float(lhs - rhs_value)]

    return _triangle_summary(
        mode="exhaustive",
        num_tests=num_tests,
        num_violations=num_violations,
        max_violation=max_violation,
        positive_violation_sum=positive_violation_sum,
        example=example,
    )


def _sampled_triangle_report(
    matrix: np.ndarray,
    *,
    num_triples: int,
    seed: int,
    tol: float,
) -> dict[str, object]:
    n = matrix.shape[0]
    rng = np.random.default_rng(seed)
    i_values = rng.integers(0, n, size=num_triples)
    j_values = rng.integers(0, n, size=num_triples)
    k_values = rng.integers(0, n, size=num_triples)

    lhs = matrix[i_values, k_values]
    rhs = matrix[i_values, j_values] + matrix[j_values, k_values]
    violations = lhs - rhs
    positive = violations > tol
    num_violations = int(np.count_nonzero(positive))

    max_violation = 0.0
    positive_violation_sum = 0.0
    example: list[object] | None = None
    if num_violations:
        positive_values = violations[positive]
        max_violation = float(np.max(positive_values))
        positive_violation_sum = float(np.sum(positive_values))
        first = int(np.flatnonzero(positive)[0])
        example = [
            int(i_values[first]),
            int(j_values[first]),
            int(k_values[first]),
            float(lhs[first]),
            float(rhs[first]),
            float(violations[first]),
        ]

    return _triangle_summary(
        mode="sampled",
        num_tests=num_triples,
        num_violations=num_violations,
        max_violation=max_violation,
        positive_violation_sum=positive_violation_sum,
        example=example,
    )


def _triangle_summary(
    *,
    mode: TriangleMode,
    num_tests: int,
    num_violations: int,
    max_violation: float,
    positive_violation_sum: float,
    example: list[object] | None,
) -> dict[str, object]:
    return {
        "triangle_test_mode": mode,
        "num_triangle_tests": int(num_tests),
        "num_triangle_violations": int(num_violations),
        "triangle_violation_fraction": float(num_violations / num_tests),
        "max_triangle_violation": float(max_violation),
        "mean_positive_triangle_violation": float(
            positive_violation_sum / num_violations if num_violations else 0.0
        ),
        "example_violation": example,
    }
