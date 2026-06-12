from __future__ import annotations

import torch
from torch import nn


class ITransformerModel(nn.Module):
    """Small iTransformer-style model using variables as tokens."""

    def __init__(
        self,
        in_dim: int,
        seq_len: int = 60,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        instance_norm: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.instance_norm = bool(instance_norm)

        self.series_proj = nn.Linear(self.seq_len, self.d_model)
        self.var_embedding = nn.Parameter(torch.zeros(1, self.in_dim, self.d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=self.d_model * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.instance_norm:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
            x = (x - mean) / std

        # [batch, time, features] -> [batch, features, time]
        z = x.transpose(1, 2)
        z = self.series_proj(z) + self.var_embedding
        z = self.encoder(z)
        pooled = self.norm(z.mean(dim=1))
        return self.head(pooled).squeeze(-1)
