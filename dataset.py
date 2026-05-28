from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleIndex:
    ts_code: str
    t: int


class PanelSeqDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
        seq_len: int = 30,
        normalize: str = "window",
        min_finite_ratio: float = 1.0,
    ):
        self.seq_len = int(seq_len)
        self.feature_cols = list(feature_cols)
        self.label_col = label_col
        self.normalize = normalize
        self.min_finite_ratio = float(min_finite_ratio)

        df = df.sort_values(["ts_code", "trade_date"], kind="mergesort")
        self._store: dict[str, dict[str, object]] = {}
        self.samples: list[SampleIndex] = []

        for code, g in df.groupby("ts_code", sort=False):
            g = g.reset_index(drop=True)
            X = g[self.feature_cols].to_numpy(dtype=np.float32, copy=True)
            y = g[self.label_col].to_numpy(dtype=np.float32, copy=True)
            dates = g["trade_date"].astype(str).to_numpy(copy=True)

            valid_y = np.isfinite(y)
            finite_X = np.isfinite(X)
            finite_ratio = finite_X.mean(axis=1)
            valid_x_row = finite_ratio >= self.min_finite_ratio

            self._store[code] = {"X": X, "y": y, "dates": dates}

            for t in range(self.seq_len - 1, len(g)):
                if not valid_y[t]:
                    continue
                if not valid_x_row[t]:
                    continue
                win = X[t - self.seq_len + 1 : t + 1]
                if not np.isfinite(win).all():
                    continue
                self.samples.append(SampleIndex(code, t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        item = self._store[s.ts_code]
        X: np.ndarray = item["X"]  # type: ignore[assignment]
        y: np.ndarray = item["y"]  # type: ignore[assignment]
        dates: np.ndarray = item["dates"]  # type: ignore[assignment]

        win = X[s.t - self.seq_len + 1 : s.t + 1].astype(np.float32, copy=True)
        if self.normalize == "window":
            m = win.mean(axis=0, keepdims=True)
            sd = win.std(axis=0, keepdims=True) + 1e-6
            win = (win - m) / sd
        label = np.float32(y[s.t])
        date = str(dates[s.t])
        return torch.from_numpy(win), torch.tensor(label), s.ts_code, date

