"""MLP baseline model."""

from __future__ import annotations

import torch
from torch import nn


class MLPModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
