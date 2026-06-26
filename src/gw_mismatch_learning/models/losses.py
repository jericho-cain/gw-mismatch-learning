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
