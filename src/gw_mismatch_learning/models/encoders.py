from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MLPEncoder(nn.Module):
    """Basic MLP encoder for mapping waveform features into a latent geometry."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dims: Sequence[int] = (64, 32),
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, embedding_dim]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1], strict=True):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
