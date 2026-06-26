from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PairBatch:
    left: np.ndarray
    right: np.ndarray
    distance: np.ndarray
    left_index: np.ndarray
    right_index: np.ndarray


def sample_pairs(
    features: np.ndarray,
    distance_matrix: np.ndarray,
    num_pairs: int,
    seed: int | None = None,
) -> PairBatch:
    rng = np.random.default_rng(seed)
    n_samples = features.shape[0]
    left_idx = rng.integers(0, n_samples, size=num_pairs)
    right_idx = rng.integers(0, n_samples, size=num_pairs)
    same = left_idx == right_idx
    right_idx[same] = (right_idx[same] + 1) % n_samples
    return PairBatch(
        left=features[left_idx],
        right=features[right_idx],
        distance=distance_matrix[left_idx, right_idx].astype(np.float32),
        left_index=left_idx.astype(np.int64),
        right_index=right_idx.astype(np.int64),
    )


class PairDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, pairs: PairBatch):
        self.left = torch.as_tensor(pairs.left, dtype=torch.float32)
        self.right = torch.as_tensor(pairs.right, dtype=torch.float32)
        self.distance = torch.as_tensor(pairs.distance, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.distance.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.left[index], self.right[index], self.distance[index]
