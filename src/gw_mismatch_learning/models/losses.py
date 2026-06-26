from __future__ import annotations

import torch
from torch import nn


class DistanceRegressionLoss(nn.Module):
    """Regress latent Euclidean distances to target mismatch distances."""

    def forward(
        self,
        left_z: torch.Tensor,
        right_z: torch.Tensor,
        target_distance: torch.Tensor,
    ) -> torch.Tensor:
        latent_distance = torch.linalg.vector_norm(left_z - right_z, dim=-1)
        return torch.mean((latent_distance - target_distance) ** 2)


class ContrastiveLoss(nn.Module):
    """Standard contrastive loss hook for future metric-learning experiments."""

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        left_z: torch.Tensor,
        right_z: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        distance = torch.linalg.vector_norm(left_z - right_z, dim=-1)
        positive = target * distance**2
        negative = (1.0 - target) * torch.clamp(self.margin - distance, min=0.0) ** 2
        return torch.mean(positive + negative)


class TripletMetricLoss(nn.Module):
    """Thin wrapper around PyTorch's triplet margin loss."""

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.loss = nn.TripletMarginLoss(margin=margin)

    def forward(
        self,
        anchor_z: torch.Tensor,
        positive_z: torch.Tensor,
        negative_z: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss(anchor_z, positive_z, negative_z)
