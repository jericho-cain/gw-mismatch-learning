from __future__ import annotations


def candidate_reduction_factor(bank_size: int, top_k: int) -> float:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    return float(bank_size) / float(top_k)
