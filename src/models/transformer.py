"""Transformer sequence model."""

from __future__ import annotations

import torch
from torch import nn


class TransformerModel(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(int(in_dim), int(d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=int(d_model), nhead=int(nhead), dropout=float(dropout), batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.head = nn.Linear(int(d_model), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = self.enc(z)
        last = z[:, -1, :]
        return self.head(last).squeeze(-1)
