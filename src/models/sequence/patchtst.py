from __future__ import annotations

import torch
from torch import nn


class PatchTSTModel(nn.Module):
    """Lightweight PatchTST-style encoder for cross-sectional ranking.

    This implementation patches the time axis and embeds each flattened
    patch. It is intentionally small enough for the existing full-universe
    sequence experiment runner.
    """

    def __init__(
        self,
        in_dim: int,
        seq_len: int = 60,
        patch_len: int = 10,
        stride: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        pooling: str = "last",
        instance_norm: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.seq_len = int(seq_len)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.d_model = int(d_model)
        self.pooling = str(pooling)
        self.instance_norm = bool(instance_norm)
        if self.patch_len < 2:
            raise ValueError("patch_len must be >= 2")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if self.seq_len < self.patch_len:
            raise ValueError("seq_len must be >= patch_len")

        self.n_patches = 1 + (self.seq_len - self.patch_len) // self.stride
        self.patch_proj = nn.Linear(self.patch_len * self.in_dim, self.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_patches, self.d_model))
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
        self.head = nn.Linear(self.d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.instance_norm:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
            x = (x - mean) / std

        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # [batch, patches, features, patch_len] -> [batch, patches, patch_len * features]
        patches = patches.transpose(2, 3).flatten(start_dim=2)
        z = self.patch_proj(patches) + self.pos_embedding[:, : patches.shape[1], :]
        z = self.encoder(z)
        if self.pooling == "mean":
            pooled = z.mean(dim=1)
        else:
            pooled = z[:, -1, :]
        pooled = self.norm(pooled)
        return self.head(pooled).squeeze(-1)
