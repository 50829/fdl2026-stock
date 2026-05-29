from __future__ import annotations

from torch import nn

from .lstm import LSTMModel
from .mlp import MLPModel
from .transformer import TransformerModel


def build_model(cfg: dict, in_dim: int) -> nn.Module:
    model_cfg = cfg.get("model", {})
    name = str(model_cfg.get("name", "mlp")).strip().lower()

    if name in {"mlp", "dummy"}:
        hidden = int(model_cfg.get("hidden", 256))
        dropout = float(model_cfg.get("dropout", 0.0))
        return MLPModel(in_dim=in_dim, hidden=hidden, dropout=dropout)

    if name == "lstm":
        hidden = int(model_cfg.get("hidden", 128))
        num_layers = int(model_cfg.get("num_layers", 1))
        dropout = float(model_cfg.get("dropout", 0.0))
        return LSTMModel(in_dim=in_dim, hidden=hidden, num_layers=num_layers, dropout=dropout)

    if name in {"transformer", "tf"}:
        d_model = int(model_cfg.get("d_model", 128))
        nhead = int(model_cfg.get("nhead", 4))
        num_layers = int(model_cfg.get("num_layers", 2))
        dropout = float(model_cfg.get("dropout", 0.0))
        return TransformerModel(in_dim=in_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)

    raise ValueError(f"Unknown model name: {name}")


__all__ = ["build_model", "MLPModel", "LSTMModel", "TransformerModel"]
