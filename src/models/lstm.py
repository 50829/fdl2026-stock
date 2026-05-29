"""LSTM sequence model."""

from __future__ import annotations

import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=int(in_dim),
            hidden_size=int(hidden),
            num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(int(hidden), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)
