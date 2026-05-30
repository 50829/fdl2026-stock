from __future__ import annotations

from torch import nn

from .lrk import ALSTM, TCNModel
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

    if name in {"alstm"}:
        input_dim = int(model_cfg.get("input_dim", in_dim))
        if input_dim != int(in_dim):
            raise ValueError(f"input_dim mismatch: config={input_dim}, data={in_dim}")
        hidden_size = int(model_cfg.get("hidden_size", 128))
        num_layers = int(model_cfg.get("num_layers", 2))
        rnn_type = str(model_cfg.get("rnn_type", "GRU"))
        dropout = float(model_cfg.get("dropout", 0.2))
        attention_hidden_ratio = float(model_cfg.get("attention_hidden_ratio", 0.5))
        seq_len = int(model_cfg.get("seq_len", 60))
        use_attention = bool(model_cfg.get("use_attention", True))
        return ALSTM(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=rnn_type,
            dropout=dropout,
            attention_hidden_ratio=attention_hidden_ratio,
            seq_len=seq_len,
            use_attention=use_attention,
        )

    if name in {"tcn", "temporal_conv", "temporal_convolution"}:
        channels = int(model_cfg.get("channels", model_cfg.get("hidden", 64)))
        levels = int(model_cfg.get("levels", model_cfg.get("num_layers", 4)))
        kernel_size = int(model_cfg.get("kernel_size", 3))
        dropout = float(model_cfg.get("dropout", 0.2))
        use_attention = bool(model_cfg.get("use_attention", False))
        attention_hidden_ratio = float(model_cfg.get("attention_hidden_ratio", 0.5))
        return TCNModel(
            in_dim=in_dim,
            channels=channels,
            levels=levels,
            kernel_size=kernel_size,
            dropout=dropout,
            use_attention=use_attention,
            attention_hidden_ratio=attention_hidden_ratio,
        )

    raise ValueError(f"Unknown model name: {name}")


__all__ = ["build_model", "MLPModel", "LSTMModel", "TransformerModel", "ALSTM", "TCNModel"]
