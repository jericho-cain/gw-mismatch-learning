from __future__ import annotations

import numpy as np


def normalized_inner_product(left: np.ndarray, right: np.ndarray) -> float:
    numerator = float(np.vdot(left, right).real)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        raise ValueError("Cannot compute overlap with a zero-norm vector")
    return numerator / denominator


def pycbc_match(left, right, psd=None, low_frequency_cutoff: float | None = None) -> float:
    try:
        from pycbc.filter import match
    except ImportError as exc:
        raise ImportError(
            "pycbc is required for pycbc_match. Install with `pip install -e .[gw]`."
        ) from exc

    kwargs = {}
    if psd is not None:
        kwargs["psd"] = psd
    if low_frequency_cutoff is not None:
        kwargs["low_frequency_cutoff"] = low_frequency_cutoff
    value, _ = match(left, right, **kwargs)
    return float(value)
