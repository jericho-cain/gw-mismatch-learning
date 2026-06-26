from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def pairwise_distance_matrix(
    items: Sequence,
    distance_fn: Callable[[object, object], float],
) -> np.ndarray:
    n_items = len(items)
    distances = np.zeros((n_items, n_items), dtype=np.float32)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            value = float(distance_fn(items[i], items[j]))
            distances[i, j] = value
            distances[j, i] = value
    return distances
