from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TripletBatch:
    anchor: np.ndarray
    positive: np.ndarray
    negative: np.ndarray


def sample_triplets(
    features: np.ndarray,
    distance_matrix: np.ndarray,
    num_triplets: int,
    seed: int | None = None,
) -> TripletBatch:
    rng = np.random.default_rng(seed)
    n_samples = features.shape[0]
    anchor_idx = rng.integers(0, n_samples, size=num_triplets)
    positive_idx = np.empty(num_triplets, dtype=np.int64)
    negative_idx = np.empty(num_triplets, dtype=np.int64)

    for row, anchor in enumerate(anchor_idx):
        order = np.argsort(distance_matrix[anchor])
        order = order[order != anchor]
        positive_idx[row] = order[0]
        negative_idx[row] = order[-1]

    return TripletBatch(
        anchor=features[anchor_idx],
        positive=features[positive_idx],
        negative=features[negative_idx],
    )


class TripletDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, triplets: TripletBatch):
        self.anchor = torch.as_tensor(triplets.anchor, dtype=torch.float32)
        self.positive = torch.as_tensor(triplets.positive, dtype=torch.float32)
        self.negative = torch.as_tensor(triplets.negative, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.anchor.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.anchor[index], self.positive[index], self.negative[index]
