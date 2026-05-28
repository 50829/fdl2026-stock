from __future__ import annotations

import torch
from torch import nn


class DummyModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_model(cfg: dict, in_dim: int) -> nn.Module:
    model_cfg = cfg.get("model", {})
    hidden = int(model_cfg.get("hidden", 64))
    dropout = float(model_cfg.get("dropout", 0.0))
    return DummyModel(in_dim=in_dim, hidden=hidden, dropout=dropout)
