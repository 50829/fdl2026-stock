from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import ProcessedConfig, ProcessedSplit
from src.evaluation import prediction_metrics
from src.model_experiments.run_gbdt import (
    feature_importance,
    load_tabular_frame,
    predict_model,
    save_model,
    train_lightgbm,
    train_xgboost,
)
from src.model_experiments.run_gbdt_walkforward import resolve_features
from src.models.fusion import DeepMLP, standardize
from src.strategy import StrategyBacktestConfig, run_strategy
from src.train import set_seed
from src.utils import make_run_dir, write_json, write_run_metadata


META_COLUMNS = [
    "pred_lgb",
    "pred_xgb",
    "rank_lgb",
    "rank_xgb",
    "base_rank",
    "pred_mean",
    "pred_diff",
    "rank_diff",
]


def _year_split(name: str, start_year: int, end_year: int) -> ProcessedSplit:
    return ProcessedSplit(name=name, start_date=f"{start_year}0101", end_date=f"{end_year}1231")


def _load_frame(
    pcfg: ProcessedConfig,
    split: ProcessedSplit,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    label_cols = [args.target, args.raw_return_col, args.daily_return_col]
    df = load_tabular_frame(pcfg, split, feature_cols, label_cols, args.filter_in_universe)
    return df.dropna(subset=[args.target]).reset_index(drop=True)


def _fit_tree_pair(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
    out_dir: Path,
    save_models_flag: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    lgb_model, lgb_params = train_lightgbm(train_df, valid_df, feature_cols, args.target, args)
    lgb_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    xgb_model, xgb_params = train_xgboost(train_df, valid_df, feature_cols, args.target, args)
    xgb_sec = time.perf_counter() - t0
    if save_models_flag:
        save_model(lgb_model, "lightgbm", out_dir / "lightgbm_model")
        save_model(xgb_model, "xgboost", out_dir / "xgboost_model")
        feature_importance(lgb_model, "lightgbm", feature_cols).to_csv(out_dir / "lightgbm_feature_importance.csv", index=False)
        feature_importance(xgb_model, "xgboost", feature_cols).to_csv(out_dir / "xgboost_feature_importance.csv", index=False)
    summary = {
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "lightgbm_train_sec": lgb_sec,
        "xgboost_train_sec": xgb_sec,
        "lightgbm_params": lgb_params,
        "xgboost_params": xgb_params,
    }
    write_json(out_dir / "tree_summary.json", summary)
    return lgb_model, xgb_model, summary


def _base_frame(
    source_df: pd.DataFrame,
    lgb_pred: np.ndarray,
    xgb_pred: np.ndarray,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    cols = ["trade_date", "ts_code", args.target, args.raw_return_col, args.daily_return_col, *feature_cols]
    cols = list(dict.fromkeys(c for c in cols if c in source_df.columns))
    out = source_df[cols].copy()
    out["pred_lgb"] = lgb_pred.astype(np.float32, copy=False)
    out["pred_xgb"] = xgb_pred.astype(np.float32, copy=False)
    out["rank_lgb"] = out.groupby("trade_date", sort=False)["pred_lgb"].rank(method="average", pct=True).astype(np.float32)
    out["rank_xgb"] = out.groupby("trade_date", sort=False)["pred_xgb"].rank(method="average", pct=True).astype(np.float32)
    out["base_rank"] = (out["rank_lgb"] + out["rank_xgb"]).astype(np.float32) - 1.0
    out["pred_mean"] = 0.5 * (out["pred_lgb"] + out["pred_xgb"])
    out["pred_diff"] = out["pred_lgb"] - out["pred_xgb"]
    out["rank_diff"] = out["rank_lgb"] - out["rank_xgb"]
    out["residual_target"] = out[args.target].astype(np.float32) - out["base_rank"].astype(np.float32)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).reset_index(drop=True)


def _prediction_frame(df: pd.DataFrame, pred: np.ndarray, args: argparse.Namespace) -> pd.DataFrame:
    cols = ["trade_date", "ts_code", args.target, args.raw_return_col, args.daily_return_col]
    cols = list(dict.fromkeys(c for c in cols if c in df.columns))
    out = df[cols].copy()
    out["pred"] = pred.astype(np.float32, copy=False)
    return out


def _write_prediction_metrics(pred_df: pd.DataFrame, split: str, model_name: str, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    metrics = prediction_metrics(pred_df, label_col=args.target, raw_return_col=args.raw_return_col if args.raw_return_col in pred_df.columns else None)
    metrics.update(
        {
            "model": model_name,
            "split": split,
            "start_date": str(pred_df["trade_date"].min()) if len(pred_df) else None,
            "end_date": str(pred_df["trade_date"].max()) if len(pred_df) else None,
            "n_dates": int(pred_df["trade_date"].nunique()) if len(pred_df) else 0,
            "metric_type": "prediction_quality",
        }
    )
    write_json(out_dir / f"{split}_metrics.json", metrics)
    return metrics


def _strategy_metrics(pred_df: pd.DataFrame, split: str, model_name: str, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    cfg = StrategyBacktestConfig(
        strategy="topk_drop",
        topk=args.topk,
        drop=args.drop,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
    )
    result = run_strategy(pred_df, cfg, name=f"topk{args.topk}_drop{args.drop}")
    metrics = dict(result["metrics"])
    metrics.update({"model": model_name, "split": split, "metric_type": "strategy_backtest"})
    write_json(out_dir / f"{split}_strategy_topk{args.topk}_drop{args.drop}_metrics.json", metrics)
    return metrics


def _matrix(df: pd.DataFrame, input_cols: list[str]) -> np.ndarray:
    return np.ascontiguousarray(df[input_cols].to_numpy(dtype=np.float32, copy=False))


def _predict_torch(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device, non_blocking=True)
            out.append(model(xb).detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(out) if out else np.empty((0,), dtype=np.float32)


def _train_residual_mlp(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    input_cols: list[str],
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover
        tqdm = None

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    x_train_raw = _matrix(train_df, input_cols)
    x_valid_raw = _matrix(valid_df, input_cols)
    x_train, [x_valid], scaler = standardize(x_train_raw, x_valid_raw)
    y_train = train_df["residual_target"].to_numpy(dtype=np.float32, copy=True)
    y_valid = valid_df["residual_target"].to_numpy(dtype=np.float32, copy=True)

    model = DeepMLP(in_dim=x_train.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    history: list[dict[str, float | int]] = []
    best_state = None
    best_loss = math.inf
    best_epoch = 0
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.0
        train_n = 0
        iterator = tqdm(loader, desc=f"oof_residual_mlp:epoch{epoch}", leave=False) if tqdm is not None else loader
        for xb, yb in iterator:
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
        valid_pred = _predict_torch(model, x_valid, args.eval_batch_size, device)
        valid_mse = float(np.mean((valid_pred.astype(np.float64) - y_valid.astype(np.float64)) ** 2))
        row = {"epoch": epoch, "train_loss": train_sum / max(1, train_n), "valid_mse": valid_mse}
        history.append(row)
        print(json.dumps({"oof_residual_mlp_train": row}, ensure_ascii=False), flush=True)
        if valid_mse < best_loss:
            best_loss = valid_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if args.patience > 0 and bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    payload = {
        "input_columns": input_cols,
        "scaler": scaler,
        "history": history,
        "best_epoch": best_epoch,
        "best_valid_mse": best_loss,
        "hidden": args.hidden,
        "dropout": args.dropout,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), **payload}, out_dir / "oof_residual_mlp.pt")
    write_json(out_dir / "train_history.json", payload)
    _plot_loss_curve(history, best_epoch, out_dir)
    return model, payload


def _plot_loss_curve(history: list[dict[str, float | int]], best_epoch: int, out_dir: Path) -> None:
    if not history:
        return
    epochs = [int(r["epoch"]) for r in history]
    train = [float(r["train_loss"]) for r in history]
    valid = [float(r["valid_mse"]) for r in history]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    color = "#4C78A8"
    ax.plot(epochs, train, color=color, linestyle="-", marker="o", linewidth=1.8, markersize=4, label="train SmoothL1")
    ax.plot(epochs, valid, color=color, linestyle="--", marker="s", linewidth=1.8, markersize=4, label="valid MSE")
    if best_epoch:
        ax.axvline(best_epoch, color="#777777", linestyle=":", linewidth=1.2, label=f"best epoch {best_epoch}")
    ax.set_title("OOF Residual MLP Training Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, color="#E6E6E6", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "oof_residual_mlp_loss_curve.png", bbox_inches="tight")
    fig.savefig(out_dir / "oof_residual_mlp_loss_curve.svg", bbox_inches="tight")
    plt.close(fig)


def _transform(df: pd.DataFrame, input_cols: list[str], scaler: dict[str, list[float]]) -> np.ndarray:
    x = _matrix(df, input_cols)
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    return np.ascontiguousarray((x - mean) / std)


def _alpha_suffix(alpha: float) -> str:
    return "a" + f"{float(alpha):g}".replace("-", "neg").replace(".", "_")


def _write_registry(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    payload = {"models": entries, "feature_sets": {"top40": {"path": "data/processed/features.parquet"}}}
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Strict OOF tree residual MLP for report-grade label1d experiments.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--feature-list", default="outputs/models/20260530_205006__feature_selection/features/lightgbm_top40.txt")
    parser.add_argument("--out-root", default="outputs/models")
    parser.add_argument("--run-name", default="report_label1d_oof_tree_residual_deep")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--target", default="label_1d__cs_rank")
    parser.add_argument("--raw-return-col", default="label_1d")
    parser.add_argument("--daily-return-col", default="label_1d")
    parser.add_argument("--oof-years", nargs="+", type=int, default=[2021, 2022, 2023])
    parser.add_argument("--min-year", type=int, default=2016)
    parser.add_argument("--filter-in-universe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-tree-models", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--num-boost-round", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--log-period", type=int, default=100)
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
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-min-child-weight", type=float, default=100.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--alpha-grid",
        nargs="+",
        type=float,
        default=[-0.10, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--drop", type=int, default=2)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    out_root = make_run_dir(args.out_root, args.run_name, timestamped=not args.no_timestamp)
    write_run_metadata(out_root, command="oof-tree-residual-deep", args=args, inputs={"feature_list": args.feature_list})
    pcfg = ProcessedConfig(processed_dir=args.processed_dir)
    feature_cols = resolve_features(pcfg, args.feature_list)

    oof_parts = []
    fold_summaries = []
    for year in args.oof_years:
        if year - 2 < args.min_year:
            raise ValueError(f"OOF year {year} is too early for one-year inner validation")
        fold_name = f"expanding_oof{year}"
        train_split = _year_split("train", args.min_year, year - 2)
        inner_valid_split = _year_split("inner_valid", year - 1, year - 1)
        oof_split = _year_split("oof", year, year)
        print(
            json.dumps(
                {
                    "stage": "oof_fold_start",
                    "fold": fold_name,
                    "train": [train_split.start_date, train_split.end_date],
                    "inner_valid": [inner_valid_split.start_date, inner_valid_split.end_date],
                    "oof": [oof_split.start_date, oof_split.end_date],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        train_df = _load_frame(pcfg, train_split, feature_cols, args)
        inner_valid_df = _load_frame(pcfg, inner_valid_split, feature_cols, args)
        oof_df = _load_frame(pcfg, oof_split, feature_cols, args)
        lgb_model, xgb_model, tree_summary = _fit_tree_pair(
            train_df,
            inner_valid_df,
            feature_cols,
            args,
            out_root / "trees" / fold_name,
            save_models_flag=args.save_tree_models,
        )
        lgb_pred = predict_model(lgb_model, "lightgbm", oof_df, feature_cols)
        xgb_pred = predict_model(xgb_model, "xgboost", oof_df, feature_cols)
        oof_base = _base_frame(oof_df, lgb_pred, xgb_pred, feature_cols, args)
        oof_path = out_root / "oof" / fold_name / "oof_base.parquet"
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        oof_base.to_parquet(oof_path, index=False)
        oof_parts.append(oof_base)
        fold_summary = {
            "fold": fold_name,
            "oof_year": year,
            "oof_rows": int(len(oof_base)),
            "oof_path": str(oof_path),
            **tree_summary,
        }
        fold_summaries.append(fold_summary)
        print(json.dumps({"stage": "oof_fold_done", **fold_summary}, ensure_ascii=False), flush=True)

    oof_train = pd.concat(oof_parts, ignore_index=True)
    oof_train_path = out_root / "oof" / "oof_train_base.parquet"
    oof_train.to_parquet(oof_train_path, index=False)

    final_train = _load_frame(pcfg, ProcessedSplit("train", "20160101", "20231231"), feature_cols, args)
    final_valid = _load_frame(pcfg, ProcessedSplit("valid", "20240101", "20241231"), feature_cols, args)
    final_test = _load_frame(pcfg, ProcessedSplit("test", "20250101", "20260518"), feature_cols, args)
    print(json.dumps({"stage": "final_tree_start", "train_rows": len(final_train), "valid_rows": len(final_valid), "test_rows": len(final_test)}, ensure_ascii=False), flush=True)
    lgb_final, xgb_final, final_tree_summary = _fit_tree_pair(
        final_train,
        final_valid,
        feature_cols,
        args,
        out_root / "trees" / "final_train2016_2023_valid2024",
        save_models_flag=True,
    )
    valid_base = _base_frame(
        final_valid,
        predict_model(lgb_final, "lightgbm", final_valid, feature_cols),
        predict_model(xgb_final, "xgboost", final_valid, feature_cols),
        feature_cols,
        args,
    )
    test_base = _base_frame(
        final_test,
        predict_model(lgb_final, "lightgbm", final_test, feature_cols),
        predict_model(xgb_final, "xgboost", final_test, feature_cols),
        feature_cols,
        args,
    )
    valid_base.to_parquet(out_root / "valid_base.parquet", index=False)
    test_base.to_parquet(out_root / "test_base.parquet", index=False)

    input_cols = [*META_COLUMNS, *feature_cols]
    residual_model, residual_payload = _train_residual_mlp(oof_train, valid_base, input_cols, args, out_root / "oof_residual_mlp")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    for frame in [valid_base, test_base]:
        x = _transform(frame, input_cols, residual_payload["scaler"])
        frame["residual_pred"] = _predict_torch(residual_model, x, args.eval_batch_size, device)

    entries: dict[str, dict[str, Any]] = {
        "oof_lgb_label1d": {"description": "严格 OOF residual 实验基线：最终 LightGBM label1d。", "predictions": {}, "metrics": {}},
        "oof_xgb_label1d": {"description": "严格 OOF residual 实验基线：最终 XGBoost label1d。", "predictions": {}, "metrics": {}},
        "oof_fusion_rank_equal_gbdt": {"description": "严格 OOF residual 实验基线：最终 LGB/XGB 等权 rank 融合。", "predictions": {}, "metrics": {}},
    }
    alpha_rows = []
    for alpha in args.alpha_grid:
        entries[f"oof_residual_mlp_{_alpha_suffix(alpha)}"] = {
            "description": f"严格 OOF residual MLP：base_rank + {float(alpha):g} * residual_pred。",
            "predictions": {},
            "metrics": {},
        }

    for split, frame in [("valid", valid_base), ("test", test_base)]:
        base_pred = frame["base_rank"].to_numpy(dtype=np.float32, copy=False)
        residual = frame["residual_pred"].to_numpy(dtype=np.float32, copy=False)
        predictions = [
            ("oof_lgb_label1d", frame["pred_lgb"].to_numpy(dtype=np.float32, copy=False), None),
            ("oof_xgb_label1d", frame["pred_xgb"].to_numpy(dtype=np.float32, copy=False), None),
            ("oof_fusion_rank_equal_gbdt", base_pred, None),
        ]
        predictions.extend((f"oof_residual_mlp_{_alpha_suffix(alpha)}", base_pred + np.float32(alpha) * residual, float(alpha)) for alpha in args.alpha_grid)
        for model_name, pred, alpha in predictions:
            split_dir = out_root / model_name / split
            split_dir.mkdir(parents=True, exist_ok=True)
            pred_df = _prediction_frame(frame, pred, args)
            pred_path = split_dir / f"{split}_pred.parquet"
            pred_df.to_parquet(pred_path, index=False)
            pred_metrics = _write_prediction_metrics(pred_df, split, model_name, args, split_dir)
            strat_metrics = _strategy_metrics(pred_df, split, model_name, args, split_dir)
            entries[model_name]["predictions"][split] = str(pred_path)
            entries[model_name]["metrics"][split] = str(split_dir / f"{split}_metrics.json")
            if alpha is not None:
                alpha_rows.append(
                    {
                        "alpha": alpha,
                        "model": model_name,
                        "split": split,
                        **pred_metrics,
                        **{f"strategy_{k}": v for k, v in strat_metrics.items() if isinstance(v, (int, float, np.integer, np.floating))},
                    }
                )
            print(
                json.dumps(
                    {
                        "model": model_name,
                        "split": split,
                        "alpha": alpha,
                        "prediction_ic": pred_metrics.get("ic_mean"),
                        "strategy_sharpe": strat_metrics.get("sharpe"),
                        "strategy_total_return": strat_metrics.get("total_return"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df.to_csv(out_root / "alpha_grid_metrics.csv", index=False)
    valid_alpha = alpha_df[alpha_df["split"] == "valid"].copy()
    valid_alpha = valid_alpha.sort_values(["strategy_sharpe", "strategy_total_return"], ascending=[False, False], kind="mergesort")
    best_alpha = float(valid_alpha.iloc[0]["alpha"]) if not valid_alpha.empty else 0.0
    best_name = f"oof_residual_mlp_{_alpha_suffix(best_alpha)}"
    entries["oof_residual_mlp"] = {
        "description": f"严格 OOF residual MLP，按 valid topk{args.topk}_drop{args.drop} Sharpe 选择 alpha={best_alpha:g}。",
        "predictions": {
            "valid": entries[best_name]["predictions"]["valid"],
            "test": entries[best_name]["predictions"]["test"],
        },
        "metrics": {
            "valid": entries[best_name]["metrics"]["valid"],
            "test": entries[best_name]["metrics"]["test"],
        },
    }
    registry_path = out_root / "models_report_label1d_oof_tree_residual.yaml"
    _write_registry(registry_path, entries)
    summary = {
        "out_root": str(out_root),
        "registry": str(registry_path),
        "feature_count": len(feature_cols),
        "input_count": len(input_cols),
        "oof_years": args.oof_years,
        "oof_rows": int(len(oof_train)),
        "valid_rows": int(len(valid_base)),
        "test_rows": int(len(test_base)),
        "oof_train_path": str(oof_train_path),
        "folds": fold_summaries,
        "final_tree_summary": final_tree_summary,
        "residual_mlp": str(out_root / "oof_residual_mlp" / "oof_residual_mlp.pt"),
        "best_alpha_by_valid_strategy_sharpe": best_alpha,
        "alpha_grid_metrics": str(out_root / "alpha_grid_metrics.csv"),
    }
    write_json(out_root / "summary.json", summary)
    print(json.dumps({"saved_summary": str(out_root / "summary.json"), "registry": str(registry_path), "best_alpha": best_alpha}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run_cli()
