from __future__ import annotations

import numpy as np

from gw_mismatch_learning.metrics.overlap import normalized_inner_product


def mismatch_from_match(match_value: float) -> float:
    return float(1.0 - match_value)


def toy_mismatch(left: np.ndarray, right: np.ndarray) -> float:
    return mismatch_from_match(normalized_inner_product(left, right))
