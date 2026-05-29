from __future__ import annotations

import torch
from torch import nn


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int, attention_hidden_ratio: float = 0.5):
        super().__init__()
        attn_hidden = max(1, int(int(hidden_size) * float(attention_hidden_ratio)))
        self.score = nn.Sequential(
            nn.Linear(int(hidden_size), attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(h).squeeze(-1)
        attn = torch.softmax(scores, dim=1)
        c = torch.sum(h * attn.unsqueeze(-1), dim=1)
        return c, attn


class ALSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        rnn_type: str = "GRU",
        dropout: float = 0.2,
        attention_hidden_ratio: float = 0.5,
        seq_len: int = 60,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.rnn_type = str(rnn_type).upper()
        self.dropout = float(dropout)
        self.seq_len = int(seq_len)

        self.feature_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.Tanh(),
        )

        rnn_dropout = self.dropout if self.num_layers > 1 else 0.0
        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        self.attention = TemporalAttention(self.hidden_size, attention_hidden_ratio=attention_hidden_ratio)
        self.head = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.feature_proj(x)
        h, _ = self.rnn(e)
        c, _ = self.attention(h)
        h_last = h[:, -1, :]
        z = torch.cat([h_last, c], dim=-1)
        return self.head(z).squeeze(-1)

    def forward_with_attention(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e = self.feature_proj(x)
        h, _ = self.rnn(e)
        c, attn = self.attention(h)
        h_last = h[:, -1, :]
        z = torch.cat([h_last, c], dim=-1)
        score = self.head(z).squeeze(-1)
        return {"score": score, "attn_weights": attn, "hidden_states": h, "context": c, "last_hidden": h_last}

