from __future__ import annotations

import torch
from torch import nn


class CausalConv1d(nn.Module):
    """1D causal convolution over time.

    Input and output use shape [batch, channels, time]. Left padding ensures
    output at time t only depends on observations up to t.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_padding = int(dilation) * (int(kernel_size) - 1)
        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            dilation=int(dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """Residual TCN block with two causal dilated convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            CausalConv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.residual = (
            nn.Conv1d(int(in_channels), int(out_channels), kernel_size=1)
            if int(in_channels) != int(out_channels)
            else nn.Identity()
        )
        self.norm = nn.BatchNorm1d(int(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return self.norm(y + self.residual(x))


class TemporalAttentionPooling(nn.Module):
    def __init__(self, channels: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden = max(1, int(int(channels) * float(hidden_ratio)))
        self.score = nn.Sequential(
            nn.Linear(int(channels), hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, time, channels]
        scores = self.score(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class TCNModel(nn.Module):
    """Temporal Convolutional Network for stock ranking.

    The default dilation pattern [1, 2, 4, 8] with kernel_size=3 and two
    convolutions per residual block gives a receptive field of 61 time steps,
    which covers the current 60-day lookback while staying smaller than the
    existing GRU baseline.
    """

    def __init__(
        self,
        in_dim: int,
        channels: int = 64,
        levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
        use_attention: bool = False,
        attention_hidden_ratio: float = 0.5,
    ):
        super().__init__()
        if int(levels) < 1:
            raise ValueError("levels must be >= 1")
        if int(kernel_size) < 2:
            raise ValueError("kernel_size must be >= 2")

        self.in_dim = int(in_dim)
        self.channels = int(channels)
        self.levels = int(levels)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self.use_attention = bool(use_attention)

        blocks = []
        in_channels = self.in_dim
        for i in range(self.levels):
            dilation = 2**i
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=self.channels,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    dropout=self.dropout,
                )
            )
            in_channels = self.channels
        self.tcn = nn.Sequential(*blocks)
        self.attention = TemporalAttentionPooling(self.channels, hidden_ratio=attention_hidden_ratio)
        self.head = nn.Linear(self.channels, 1)

    @property
    def receptive_field(self) -> int:
        dilation_sum = sum(2**i for i in range(self.levels))
        return 1 + 2 * (self.kernel_size - 1) * dilation_sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, time, features] -> [batch, features, time]
        z = x.transpose(1, 2)
        z = self.tcn(z)
        z = z.transpose(1, 2)
        if self.use_attention:
            pooled, _ = self.attention(z)
        else:
            pooled = z[:, -1, :]
        return self.head(pooled).squeeze(-1)

