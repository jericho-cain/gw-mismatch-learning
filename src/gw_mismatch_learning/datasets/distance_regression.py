from __future__ import annotations

import torch
from torch.utils.data import Dataset

from gw_mismatch_learning.datasets.pairs import PairBatch


class DistanceRegressionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Torch dataset for regressing latent pair distances to target distances."""

    def __init__(self, pairs: PairBatch):
        self.left = torch.as_tensor(pairs.left, dtype=torch.float32)
        self.right = torch.as_tensor(pairs.right, dtype=torch.float32)
        self.distance = torch.as_tensor(pairs.distance, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.distance.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.left[index], self.right[index], self.distance[index]
