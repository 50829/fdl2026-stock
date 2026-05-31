from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import ProcessedConfig, ProcessedSplit
from src.models.mlp import MLPModel
from src.evaluation import ic_metrics
from src.models.fusion import DeepMLP, standardize
from src.models.sdd.run_gbdt import evaluate_predictions, load_tabular_frame, predict_model, train_lightgbm
from src.models.sdd.run_gbdt_walkforward import resolve_features, year_split
from src.train import set_seed
from src.utils import write_json


def split_range(name: str, start: str, end: str) -> ProcessedSplit:
    return ProcessedSplit(name=name, start_date=start, end_date=end)


def load_frame_for_dates(
    pcfg: ProcessedConfig,
    feature_cols: list[str],
    args: argparse.Namespace,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    label_cols = [args.target, args.raw_return_col, args.daily_return_col]
    df = load_tabular_frame(
        pcfg,
        split_range("custom", start_date, end_date),
        feature_cols,
        label_cols,
        filter_in_universe=args.filter_in_universe,
    )
    return df.dropna(subset=[args.target]).reset_index(drop=True)


def train_lgb_for_split(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
):
    return train_lightgbm(train_df, valid_df, feature_cols, args.target, args)[0]


def add_base_pred(df: pd.DataFrame, model, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    out = df.copy()
    out["base_pred"] = predict_model(model, "lightgbm", out, feature_cols)
    out["residual"] = out[target_col].to_numpy(dtype=np.float32) - out["base_pred"].to_numpy(dtype=np.float32)
    return out


def make_mlp_matrix(df: pd.DataFrame, mlp_feature_cols: list[str]) -> np.ndarray:
    cols = list(mlp_feature_cols) + ["base_pred"]
    return df[cols].to_numpy(dtype=np.float32, copy=False)


def predict_mlp(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).to(device)
            preds.append(model(xb).detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float32)


def train_residual_mlp(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    mlp_feature_cols: list[str],
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[nn.Module, dict]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    x_train = make_mlp_matrix(train_df, mlp_feature_cols)
    x_valid = make_mlp_matrix(valid_df, mlp_feature_cols)
    x_train, [x_valid], scaler = standardize(x_train, x_valid)
    y_train = train_df["residual"].to_numpy(dtype=np.float32, copy=True)
    y_valid = valid_df["residual"].to_numpy(dtype=np.float32, copy=True)

    if args.mlp_arch == "deep_ln":
        model: nn.Module = DeepMLP(in_dim=x_train.shape[1], hidden=args.mlp_hidden, dropout=args.mlp_dropout).to(device)
    else:
        model = MLPModel(in_dim=x_train.shape[1], hidden=args.mlp_hidden, dropout=args.mlp_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_weight_decay)
    loss_fn = nn.SmoothL1Loss() if args.mlp_loss == "smooth_l1" else nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(x_train)), torch.from_numpy(y_train)),
        batch_size=args.mlp_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    best_loss = math.inf
    best_epoch = 0
    bad_epochs = 0
    history = []
    best_state = None
    t_all = time.perf_counter()

    for epoch in range(1, args.mlp_epochs + 1):
        model.train()
        train_sum = 0.0
        train_n = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            train_sum += float(loss.item()) * int(yb.shape[0])
            train_n += int(yb.shape[0])

        valid_pred = predict_mlp(model, x_valid, args.eval_batch_size, device)
        valid_loss = float(np.mean((valid_pred.astype(np.float64) - y_valid.astype(np.float64)) ** 2))
        row = {
            "epoch": epoch,
            "train_loss": train_sum / max(1, train_n),
            "valid_mse": valid_loss,
        }
        history.append(row)
        print(json.dumps({"mlp_train": row}, ensure_ascii=False), flush=True)
        if valid_loss < best_loss - args.min_delta:
            best_loss = valid_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    payload = {
        "model_state": model.state_dict(),
        "feature_cols": mlp_feature_cols,
        "input_cols": list(mlp_feature_cols) + ["base_pred"],
        "mlp_arch": args.mlp_arch,
        "scaler": scaler,
        "best_epoch": best_epoch,
        "best_valid_mse": best_loss,
        "history": history,
        "elapsed_sec": time.perf_counter() - t_all,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_dir / "residual_mlp.pt")
    write_json(out_dir / "mlp_history.json", {k: v for k, v in payload.items() if k != "model_state"})
    model.scaler_ = scaler  # type: ignore[attr-defined]
    return model, payload


def transform_with_payload(df: pd.DataFrame, mlp_feature_cols: list[str], payload: dict) -> np.ndarray:
    x = make_mlp_matrix(df, mlp_feature_cols)
    mean = np.asarray(payload["scaler"]["mean"], dtype=np.float32)
    std = np.asarray(payload["scaler"]["std"], dtype=np.float32)
    return ((x - mean) / std).astype(np.float32, copy=False)


def evaluate_residual_outputs(
    name: str,
    df: pd.DataFrame,
    residual_pred: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    key_cols = ["trade_date", "ts_code", args.target, args.raw_return_col, args.daily_return_col]
    base_df = df[key_cols].copy()
    base_df["pred"] = df["base_pred"].to_numpy(dtype=np.float32, copy=False)
    base_metrics = evaluate_predictions(base_df, args.target, args)

    residual_df = df[["trade_date", "ts_code", "residual"]].copy()
    residual_df["pred"] = residual_pred
    residual_metrics = {
        "samples": int(len(residual_df)),
        "residual_mse": float(np.mean((residual_pred.astype(np.float64) - residual_df["residual"].to_numpy(dtype=np.float64)) ** 2)),
    }
    residual_metrics.update(ic_metrics(residual_df.rename(columns={"residual": "residual_label"}), label_col="residual_label"))

    alpha_rows = []
    for alpha in args.alpha_grid:
        pred_df = df[key_cols].copy()
        pred_df["pred"] = df["base_pred"].to_numpy(dtype=np.float32, copy=False) + float(alpha) * residual_pred
        metrics = evaluate_predictions(pred_df, args.target, args)
        alpha_rows.append({"alpha": float(alpha), **metrics})
    alpha_df = pd.DataFrame(alpha_rows).sort_values(["icir", "ic_mean"], ascending=False, kind="mergesort")
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_df.to_csv(out_dir / f"{name}_alpha_grid.csv", index=False)
    summary = {
        "split": name,
        "base": base_metrics,
        "residual_model_as_score": residual_metrics,
        "best_alpha_by_icir": alpha_df.iloc[0].to_dict() if not alpha_df.empty else {},
    }
    write_json(out_dir / f"{name}_summary.json", summary)
    print(json.dumps({"split": name, "summary": summary}, ensure_ascii=False), flush=True)
    return summary


def run_frozen(
    args: argparse.Namespace,
    pcfg: ProcessedConfig,
    data_feature_cols: list[str],
    base_feature_cols: list[str],
    mlp_feature_cols: list[str],
) -> dict:
    out_dir = Path(args.out_root) / "frozen_lgbm_mlp"
    train_base = load_frame_for_dates(pcfg, data_feature_cols, args, "20160101", "20201231")
    valid_base = load_frame_for_dates(pcfg, data_feature_cols, args, "20210101", "20211231")
    base_model = train_lgb_for_split(train_base, valid_base, base_feature_cols, args)

    residual_train_raw = load_frame_for_dates(pcfg, data_feature_cols, args, "20210101", "20231231")
    residual_valid_raw = load_frame_for_dates(pcfg, data_feature_cols, args, "20240101", "20241231")
    residual_test_raw = load_frame_for_dates(pcfg, data_feature_cols, args, "20250101", "20260518")
    residual_train = add_base_pred(residual_train_raw, base_model, base_feature_cols, args.target)
    residual_valid = add_base_pred(residual_valid_raw, base_model, base_feature_cols, args.target)
    residual_test = add_base_pred(residual_test_raw, base_model, base_feature_cols, args.target)

    model, payload = train_residual_mlp(residual_train, residual_valid, mlp_feature_cols, out_dir, args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    summaries = {}
    for split, df in [("valid", residual_valid), ("test", residual_test)]:
        x = transform_with_payload(df, mlp_feature_cols, payload)
        residual_pred = predict_mlp(model, x, args.eval_batch_size, device)
        summaries[split] = evaluate_residual_outputs(split, df, residual_pred, args, out_dir / split)
    summary = {
        "experiment": "frozen_lgbm_mlp",
        "base_feature_count": len(base_feature_cols),
        "mlp_feature_count": len(mlp_feature_cols),
        "base_train": ["20160101", "20201231"],
        "residual_train": ["20210101", "20231231"],
        "valid": summaries["valid"],
        "test": summaries["test"],
        "mlp": {k: v for k, v in payload.items() if k != "model_state"},
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def train_oof_base_predictions(
    args: argparse.Namespace,
    pcfg: ProcessedConfig,
    data_feature_cols: list[str],
    base_feature_cols: list[str],
) -> pd.DataFrame:
    pieces = []
    for year in [2021, 2022, 2023]:
        train_df = load_frame_for_dates(pcfg, data_feature_cols, args, "20160101", f"{year - 1}1231")
        valid_df = load_frame_for_dates(pcfg, data_feature_cols, args, f"{year}0101", f"{year}1231")
        model = train_lgb_for_split(train_df, valid_df, base_feature_cols, args)
        pieces.append(add_base_pred(valid_df, model, base_feature_cols, args.target))
    return pd.concat(pieces, ignore_index=True)


def run_oof(
    args: argparse.Namespace,
    pcfg: ProcessedConfig,
    data_feature_cols: list[str],
    base_feature_cols: list[str],
    mlp_feature_cols: list[str],
) -> dict:
    out_dir = Path(args.out_root) / "oof_lgbm_mlp"
    residual_train = train_oof_base_predictions(args, pcfg, data_feature_cols, base_feature_cols)

    final_train = load_frame_for_dates(pcfg, data_feature_cols, args, "20160101", "20231231")
    final_valid_raw = load_frame_for_dates(pcfg, data_feature_cols, args, "20240101", "20241231")
    final_test_raw = load_frame_for_dates(pcfg, data_feature_cols, args, "20250101", "20260518")
    final_model = train_lgb_for_split(final_train, final_valid_raw, base_feature_cols, args)
    residual_valid = add_base_pred(final_valid_raw, final_model, base_feature_cols, args.target)
    residual_test = add_base_pred(final_test_raw, final_model, base_feature_cols, args.target)

    model, payload = train_residual_mlp(residual_train, residual_valid, mlp_feature_cols, out_dir, args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    summaries = {}
    for split, df in [("valid", residual_valid), ("test", residual_test)]:
        x = transform_with_payload(df, mlp_feature_cols, payload)
        residual_pred = predict_mlp(model, x, args.eval_batch_size, device)
        summaries[split] = evaluate_residual_outputs(split, df, residual_pred, args, out_dir / split)
    summary = {
        "experiment": "oof_lgbm_mlp",
        "base_feature_count": len(base_feature_cols),
        "mlp_feature_count": len(mlp_feature_cols),
        "oof_years": [2021, 2022, 2023],
        "final_base_train": ["20160101", "20231231"],
        "valid": summaries["valid"],
        "test": summaries["test"],
        "mlp": {k: v for k, v in payload.items() if k != "model_state"},
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["frozen", "oof", "both"], default="both")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--feature-list", default="outputs/sdd_feature_selection/features/lightgbm_top40.txt")
    parser.add_argument("--base-feature-list", default=None)
    parser.add_argument("--mlp-feature-list", default=None)
    parser.add_argument("--out-root", default="outputs/sdd_residual_mlp")
    parser.add_argument("--target", default="label_5d__cs_rank")
    parser.add_argument("--raw-return-col", default="label_5d")
    parser.add_argument("--daily-return-col", default="label_1d")
    parser.add_argument("--filter-in-universe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--num-boost-round", type=int, default=800)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--log-period", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-data-in-leaf", type=int, default=1000)
    parser.add_argument("--feature-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-l2", type=float, default=1.0)
    parser.add_argument("--max-bin", type=int, default=255)
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--mlp-arch", choices=["shallow", "deep_ln"], default="shallow")
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-loss", choices=["mse", "smooth_l1"], default="smooth_l1")
    parser.add_argument("--mlp-epochs", type=int, default=6)
    parser.add_argument("--mlp-batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=3.0)
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--n-hold", type=int, default=20)
    parser.add_argument("--k-rotate", type=int, default=5)
    parser.add_argument("--step-days", type=int, default=5)
    parser.add_argument("--tranche-size", type=int, default=4)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    args = parser.parse_args()

    pcfg = ProcessedConfig(processed_dir=args.processed_dir)
    base_feature_list = args.base_feature_list if args.base_feature_list is not None else args.feature_list
    mlp_feature_list = args.mlp_feature_list if args.mlp_feature_list is not None else args.feature_list
    base_feature_cols = resolve_features(pcfg, base_feature_list)
    mlp_feature_cols = resolve_features(pcfg, mlp_feature_list)
    all_feature_cols = resolve_features(pcfg, None)
    feature_union = set(base_feature_cols) | set(mlp_feature_cols)
    data_feature_cols = [col for col in all_feature_cols if col in feature_union]
    summaries = []
    if args.mode in {"frozen", "both"}:
        summaries.append(run_frozen(args, pcfg, data_feature_cols, base_feature_cols, mlp_feature_cols))
    if args.mode in {"oof", "both"}:
        summaries.append(run_oof(args, pcfg, data_feature_cols, base_feature_cols, mlp_feature_cols))
    write_json(Path(args.out_root) / "summary.json", {"experiments": summaries})
