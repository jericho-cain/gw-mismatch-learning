from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from gw_mismatch_learning.datasets.pairs import PairDataset
from gw_mismatch_learning.models.losses import DistanceRegressionLoss


@dataclass(frozen=True)
class TrainingHistory:
    losses: list[float]


def train_distance_regression(
    encoder: torch.nn.Module,
    dataset: PairDataset,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str | torch.device = "cpu",
) -> TrainingHistory:
    encoder.to(device)
    encoder.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    criterion = DistanceRegressionLoss()
    losses: list[float] = []

    for _ in range(epochs):
        epoch_losses: list[float] = []
        for left, right, distance in loader:
            left = left.to(device)
            right = right.to(device)
            distance = distance.to(device)
            optimizer.zero_grad()
            loss = criterion(encoder(left), encoder(right), distance)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    return TrainingHistory(losses=losses)


def encode_array(
    encoder: torch.nn.Module,
    features: np.ndarray,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    encoder.to(device)
    encoder.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(
                features[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            chunks.append(encoder(batch).cpu().numpy())
    return np.concatenate(chunks, axis=0)
