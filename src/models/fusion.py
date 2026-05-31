from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


class DeepMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden)),
            nn.LayerNorm(int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(hidden)),
            nn.LayerNorm(int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


META_INPUT_COLUMNS = [
    "pred_lgb",
    "pred_xgb",
    "rank_lgb",
    "rank_xgb",
    "pred_mean",
    "rank_mean",
    "pred_diff",
    "rank_diff",
]


def add_meta_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rank_lgb"] = out.groupby("trade_date")["pred_lgb"].rank(method="average", pct=True).astype(np.float32)
    out["rank_xgb"] = out.groupby("trade_date")["pred_xgb"].rank(method="average", pct=True).astype(np.float32)
    out["pred_mean"] = 0.5 * (out["pred_lgb"] + out["pred_xgb"])
    out["rank_mean"] = 0.5 * (out["rank_lgb"] + out["rank_xgb"])
    out["pred_diff"] = out["pred_lgb"] - out["pred_xgb"]
    out["rank_diff"] = out["rank_lgb"] - out["rank_xgb"]
    return out


def merge_lgb_xgb_predictions(lgb: pd.DataFrame, xgb: pd.DataFrame, label_cols: list[str] | None = None) -> pd.DataFrame:
    label_cols = list(dict.fromkeys(c for c in (label_cols or []) if c in lgb.columns))
    keep = ["trade_date", "ts_code", "pred_lgb"] + label_cols
    df = lgb[keep].merge(xgb[["trade_date", "ts_code", "pred_xgb"]], on=["trade_date", "ts_code"], how="inner")
    df["trade_date"] = df["trade_date"].astype(str)
    df["ts_code"] = df["ts_code"].astype(str)
    return add_meta_prediction_features(df)


def standardize(train_x: np.ndarray, *xs: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], dict[str, list[float]]]:
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, [(x - mean) / std for x in xs], {"mean": mean.tolist(), "std": std.tolist()}


@dataclass
class ResidualRankFusionModel:
    model: DeepMLP
    input_columns: list[str]
    scaler: dict[str, list[float]]
    alpha: float = 1.5

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.input_columns].to_numpy(dtype=np.float32, copy=False)
        mean = np.asarray(self.scaler["mean"], dtype=np.float32)
        std = np.asarray(self.scaler["std"], dtype=np.float32)
        return ((x - mean) / std).astype(np.float32, copy=False)

    def predict_residual(self, df: pd.DataFrame, batch_size: int = 65536, device: str | torch.device | None = None) -> np.ndarray:
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(dev)
        self.model.eval()
        x = self.transform(df)
        preds = []
        with torch.no_grad():
            for i in range(0, len(x), int(batch_size)):
                xb = torch.from_numpy(x[i : i + int(batch_size)]).to(dev)
                preds.append(self.model(xb).detach().cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float32)

    def predict_frame(self, df: pd.DataFrame, batch_size: int = 65536, device: str | torch.device | None = None) -> pd.DataFrame:
        out = df.copy()
        residual = self.predict_residual(out, batch_size=batch_size, device=device)
        out["residual_rank_pred"] = residual
        out["alpha"] = np.float32(self.alpha)
        out["final_pred"] = out["pred_lgb"].to_numpy(dtype=np.float32) + np.float32(self.alpha) * residual
        out["pred"] = out["final_pred"].astype(np.float32)
        return out


def load_residual_rank_fusion(path: str | Path, alpha: float | None = None) -> ResidualRankFusionModel:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    input_columns = list(ckpt["input_columns"])
    model = DeepMLP(
        in_dim=len(input_columns),
        hidden=int(ckpt.get("mlp_hidden", 128)),
        dropout=float(ckpt.get("mlp_dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state"])
    if alpha is None:
        best_valid = ckpt.get("best_valid") or {}
        alpha = float(best_valid.get("alpha", 1.5))
    return ResidualRankFusionModel(
        model=model,
        input_columns=input_columns,
        scaler=ckpt["scaler"],
        alpha=float(alpha),
    )
