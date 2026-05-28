from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from data_utils import filter_stock_pool, load_daily_dir, load_panel_csv, merge_on_keys, normalize_date_str
from dataset import PanelSeqDataset
from features import add_basic_tech_features, add_future_return_label
from models import build_model
from strategies import build_strategy


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    if np.allclose(ar, ar[0]) or np.allclose(br, br[0]):
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def ic_by_day(preds: np.ndarray, labels: np.ndarray, codes: list[str], dates: list[str]) -> np.ndarray:
    df = pd.DataFrame({"pred": preds, "label": labels, "ts_code": codes, "trade_date": dates})
    ics = []
    for _, g in df.groupby("trade_date", sort=True):
        g = g.dropna(subset=["pred", "label"])
        if len(g) < 3:
            continue
        ic = _spearman_corr(g["pred"].to_numpy(), g["label"].to_numpy())
        if np.isfinite(ic):
            ics.append(ic)
    return np.asarray(ics, dtype=np.float32)


def _load_data(cfg: dict) -> pd.DataFrame:
    data_cfg = cfg.get("data", {})
    panel_csv = data_cfg.get("panel_csv")
    if panel_csv:
        df = load_panel_csv(panel_csv)
    else:
        daily_dir = data_cfg["daily_dir"]
        start_date = normalize_date_str(data_cfg["start_date"])
        end_date = normalize_date_str(data_cfg["end_date"])
        usecols = data_cfg.get(
            "usecols",
            [
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "pre_close",
                "vol",
                "amount",
                "vwap",
            ],
        )
        df = load_daily_dir(daily_dir, start_date, end_date, usecols=usecols, limit_files=data_cfg.get("limit_files"))

        for extra in data_cfg.get("extra_daily_dirs", []) or []:
            extra_dir = extra.get("dir")
            if not extra_dir:
                continue
            extra_usecols = extra.get("usecols")
            ex = load_daily_dir(extra_dir, start_date, end_date, usecols=extra_usecols, limit_files=data_cfg.get("limit_files"))
            df = merge_on_keys(df, ex)

    pool_cfg = data_cfg.get("pool", {})
    df = filter_stock_pool(
        df,
        basic_csv=pool_cfg.get("basic_csv"),
        stock_st_dir=pool_cfg.get("stock_st_dir"),
        exclude_bj=bool(pool_cfg.get("exclude_bj", True)),
        exclude_st=bool(pool_cfg.get("exclude_st", True)),
    )
    return df


def evaluate(cfg: dict):
    eval_cfg = cfg.get("eval", {})
    ckpt_path = Path(eval_cfg.get("ckpt", "ckpt.pt"))
    ckpt = torch.load(ckpt_path, map_location="cpu")

    feature_cols = list(ckpt["feature_cols"])
    label_col = str(ckpt["label_col"])
    seq_len = int(ckpt["seq_len"])
    normalize = str(ckpt.get("normalize", "window"))

    df = _load_data(cfg)
    df = add_basic_tech_features(df)
    df, computed_label_col = add_future_return_label(df, horizon=int(cfg.get("task", {}).get("horizon", 1)))
    if computed_label_col != label_col:
        label_col = computed_label_col
    keep_cols = ["ts_code", "trade_date"] + feature_cols + [label_col]
    df = df[keep_cols].copy()

    split_cfg = cfg.get("split", {})
    train_end = normalize_date_str(split_cfg["train_end"])
    val_end = normalize_date_str(split_cfg["val_end"])
    va_df = df[(df["trade_date"] > train_end) & (df["trade_date"] <= val_end)].copy()

    ds = PanelSeqDataset(va_df, feature_cols, label_col, seq_len=seq_len, normalize=normalize)
    bs = int(eval_cfg.get("batch_size", 512))
    num_workers = int(eval_cfg.get("num_workers", 0))
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)

    in_dim = seq_len * len(feature_cols)
    model = build_model(cfg, in_dim=in_dim)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loss_fn = nn.MSELoss(reduction="sum")
    preds: list[float] = []
    labels: list[float] = []
    codes: list[str] = []
    dates: list[str] = []
    loss_sum = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb, cb, db in dl:
            pb = model(xb).squeeze(-1)
            loss_sum += float(loss_fn(pb, yb).item())
            n += int(yb.shape[0])
            preds.extend(pb.cpu().numpy().tolist())
            labels.extend(yb.cpu().numpy().tolist())
            codes.extend(list(cb))
            dates.extend(list(db))
    mse = loss_sum / max(1, n)

    ics = ic_by_day(np.asarray(preds), np.asarray(labels), codes, dates)
    ic_mean = float(np.nanmean(ics)) if len(ics) else float("nan")
    ic_std = float(np.nanstd(ics)) if len(ics) else float("nan")
    icir = ic_mean / (ic_std + 1e-12) if np.isfinite(ic_mean) and np.isfinite(ic_std) else float("nan")

    out = {"val_mse": mse, "ic_mean": ic_mean, "icir": icir, "ic_days": int(len(ics)), "samples": int(len(ds))}
    print(json.dumps(out, ensure_ascii=False))

    pred_path = eval_cfg.get("pred_path")
    if pred_path:
        pred_df = pd.DataFrame({"ts_code": codes, "trade_date": dates, "pred": preds, "label": labels})
        Path(pred_path).parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_path, index=False)
        print(json.dumps({"saved_pred": str(pred_path)}, ensure_ascii=False))

    bt_cfg = cfg.get("backtest", {})
    if bool(bt_cfg.get("enabled", False)):
        pred_df = pd.DataFrame({"ts_code": codes, "trade_date": dates, "pred": preds, "label": labels})
        horizon = int(cfg.get("task", {}).get("horizon", 1))
        strategy = build_strategy(cfg)
        res = strategy(pred_df, horizon)
        print(json.dumps({"backtest": res.metrics}, ensure_ascii=False))

        curve_path = bt_cfg.get("curve_path")
        if curve_path:
            Path(curve_path).parent.mkdir(parents=True, exist_ok=True)
            res.equity_curve.to_csv(curve_path, index=False)
            print(json.dumps({"saved_curve": str(curve_path)}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    evaluate(cfg)


if __name__ == "__main__":
    main()

