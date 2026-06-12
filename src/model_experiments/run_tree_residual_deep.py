from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import prediction_metrics
from src.models.fusion import DeepMLP, standardize
from src.pipelines.run_strategy_backtest import load_model_registry, resolve_prediction_path
from src.strategy import StrategyBacktestConfig, run_strategy
from src.train import set_seed
from src.utils import make_run_dir, write_json, write_run_metadata


DEFAULT_MODEL_REGISTRY = "configs/registry/models_report_label1d.yaml"
BASE_MODELS = ("lgb_label1d", "xgb_label1d")
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


def _load_pred(registry: dict[str, Any], model: str, split: str, score_col: str) -> pd.DataFrame:
    df = pd.read_parquet(resolve_prediction_path(registry, model, split))
    required = {"trade_date", "ts_code", score_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"`{model}` split `{split}` missing columns: {missing}")
    return df.rename(columns={score_col: f"pred_{model.split('_')[0]}"})


def _base_frame(registry: dict[str, Any], split: str, score_col: str, label_col: str) -> pd.DataFrame:
    lgb = _load_pred(registry, "lgb_label1d", split, score_col)
    xgb = _load_pred(registry, "xgb_label1d", split, score_col)
    keep = ["trade_date", "ts_code", "pred_lgb", label_col, "label_1d"]
    keep = [col for col in keep if col in lgb.columns]
    df = lgb[keep].merge(xgb[["trade_date", "ts_code", "pred_xgb"]], on=["trade_date", "ts_code"], how="inner")
    df["rank_lgb"] = df.groupby("trade_date", sort=False)["pred_lgb"].rank(method="average", pct=True).astype(np.float32)
    df["rank_xgb"] = df.groupby("trade_date", sort=False)["pred_xgb"].rank(method="average", pct=True).astype(np.float32)
    df["base_rank"] = (df["rank_lgb"] + df["rank_xgb"]).astype(np.float32) - 1.0
    df["pred_mean"] = 0.5 * (df["pred_lgb"] + df["pred_xgb"])
    df["pred_diff"] = df["pred_lgb"] - df["pred_xgb"]
    df["rank_diff"] = df["rank_lgb"] - df["rank_xgb"]
    df["residual_target"] = df[label_col].astype(np.float32) - df["base_rank"].astype(np.float32)
    return df.dropna(subset=[label_col, "label_1d", "base_rank", "residual_target"]).reset_index(drop=True)


def _feature_columns(feature_path: Path, limit: int | None) -> list[str]:
    import pyarrow.parquet as pq

    columns = pq.ParquetFile(feature_path).schema.names
    features = [col for col in columns if col not in {"trade_date", "ts_code"}]
    return features[:limit] if limit else features


def _load_features_for_frames(feature_path: Path, frames: list[pd.DataFrame], feature_cols: list[str]) -> pd.DataFrame:
    dates = pd.concat([frame["trade_date"] for frame in frames], ignore_index=True)
    min_date = dates.min()
    max_date = dates.max()
    cols = ["trade_date", "ts_code", *feature_cols]
    try:
        return pd.read_parquet(feature_path, columns=cols, filters=[("trade_date", ">=", min_date), ("trade_date", "<=", max_date)])
    except Exception:
        feat = pd.read_parquet(feature_path, columns=cols)
        return feat[(feat["trade_date"] >= min_date) & (feat["trade_date"] <= max_date)].copy()


def _attach_features(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(features, on=["trade_date", "ts_code"], how="inner")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _split_calibration_dates(valid_df: pd.DataFrame, valid_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.asarray(sorted(valid_df["trade_date"].unique()))
    n_valid = max(1, int(round(len(dates) * valid_fraction)))
    cut = dates[-n_valid]
    train = valid_df[valid_df["trade_date"] < cut].copy()
    calib = valid_df[valid_df["trade_date"] >= cut].copy()
    if train.empty or calib.empty:
        raise ValueError("calibration split produced an empty train or validation segment")
    return train, calib


def _matrix(df: pd.DataFrame, input_cols: list[str]) -> np.ndarray:
    return np.ascontiguousarray(df[input_cols].to_numpy(dtype=np.float32, copy=False))


def _predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device, non_blocking=True)
            preds.append(model(xb).detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(preds) if preds else np.empty((0,), dtype=np.float32)


def _train_residual_mlp(
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    input_cols: list[str],
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - tqdm is optional.
        tqdm = None

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    x_train_raw = _matrix(train_df, input_cols)
    x_calib_raw = _matrix(calib_df, input_cols)
    x_train, [x_calib], scaler = standardize(x_train_raw, x_calib_raw)
    y_train = train_df["residual_target"].to_numpy(dtype=np.float32, copy=True)
    y_calib = calib_df["residual_target"].to_numpy(dtype=np.float32, copy=True)

    model = DeepMLP(in_dim=x_train.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    history = []
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.0
        train_n = 0
        iterator = loader
        if tqdm is not None:
            iterator = tqdm(loader, desc=f"residual_mlp:epoch{epoch}", leave=False)
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
        calib_pred = _predict(model, x_calib, args.eval_batch_size, device)
        calib_loss = float(np.mean((calib_pred.astype(np.float64) - y_calib.astype(np.float64)) ** 2))
        row = {"epoch": epoch, "train_loss": train_sum / max(1, train_n), "calib_mse": calib_loss}
        history.append(row)
        print(json.dumps({"residual_mlp_train": row}, ensure_ascii=False), flush=True)
        if calib_loss < best_loss:
            best_loss = calib_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    payload = {
        "input_columns": input_cols,
        "scaler": scaler,
        "history": history,
        "best_epoch": best_epoch,
        "best_calib_mse": best_loss,
        "hidden": args.hidden,
        "dropout": args.dropout,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), **payload}, out_dir / "residual_mlp.pt")
    (out_dir / "train_history.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return model, payload


def _transform(df: pd.DataFrame, input_cols: list[str], scaler: dict[str, list[float]]) -> np.ndarray:
    x = _matrix(df, input_cols)
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    return np.ascontiguousarray((x - mean) / std)


def _rank_ic(df: pd.DataFrame, score_col: str, label_col: str) -> float:
    values = []
    for _, g in df.groupby("trade_date", sort=False):
        if g[score_col].nunique(dropna=True) <= 1 or g[label_col].nunique(dropna=True) <= 1:
            continue
        values.append(g[score_col].corr(g[label_col], method="pearson"))
    return float(np.nanmean(values)) if values else float("nan")


def _alpha_grid(calib_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    cfg = StrategyBacktestConfig(strategy="topk_drop", topk=10, drop=2, transaction_cost_bps=args.transaction_cost_bps)
    for alpha in args.alpha_grid:
        df = calib_df[["trade_date", "ts_code", args.label_col, "label_1d", "base_rank", "residual_pred"]].copy()
        df["pred"] = df["base_rank"].astype(np.float32) + float(alpha) * df["residual_pred"].astype(np.float32)
        metrics = run_strategy(df, cfg, name="topk10_drop2")["metrics"]
        rows.append(
            {
                "alpha": float(alpha),
                "rank_ic": _rank_ic(df, "pred", args.label_col),
                "sharpe": float(metrics["sharpe"]),
                "total_return": float(metrics["total_return"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "avg_turnover": float(metrics["avg_turnover"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["sharpe", "max_drawdown", "rank_ic"], ascending=[False, False, False], kind="mergesort")


def _prediction_frame(df: pd.DataFrame, pred: np.ndarray, label_col: str) -> pd.DataFrame:
    cols = ["trade_date", "ts_code", label_col, "label_1d"]
    out = df[cols].copy()
    out["pred"] = pred.astype(np.float32, copy=False)
    return out


def _write_registry(path: Path, base_registry: dict[str, Any], entries: dict[str, dict[str, Any]]) -> None:
    payload = {"models": {**base_registry.get("models", {}), **entries}, "feature_sets": base_registry.get("feature_sets", {})}
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _write_prediction_metrics(pred_df: pd.DataFrame, split: str, label_col: str, out_dir: Path) -> dict[str, object]:
    metrics = prediction_metrics(pred_df, label_col=label_col, raw_return_col="label_1d" if "label_1d" in pred_df.columns else None)
    metrics.update(
        {
            "split": split,
            "start_date": str(pred_df["trade_date"].min()) if len(pred_df) else None,
            "end_date": str(pred_df["trade_date"].max()) if len(pred_df) else None,
            "n_dates": int(pred_df["trade_date"].nunique()) if len(pred_df) else 0,
            "metric_type": "prediction_quality",
        }
    )
    write_json(out_dir / f"{split}_metrics.json", metrics)
    return metrics


def _alpha_suffix(alpha: float) -> str:
    text = f"{float(alpha):g}".replace("-", "neg").replace(".", "_")
    return f"a{text}"


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Train a deep residual model behind the GBDT rank ensemble.")
    parser.add_argument("--model-registry", default=DEFAULT_MODEL_REGISTRY)
    parser.add_argument("--feature-path", default="data/processed/features.parquet")
    parser.add_argument("--out-root", default="outputs/models")
    parser.add_argument("--run-name", default="report_label1d_tree_residual_deep")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--score-col", default="pred")
    parser.add_argument("--label-col", default="label_1d__cs_rank")
    parser.add_argument("--calib-valid-fraction", type=float, default=0.30)
    parser.add_argument("--feature-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.0])
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    registry = load_model_registry(args.model_registry)
    for model in BASE_MODELS:
        if model not in registry["models"]:
            parser.error(f"model registry must contain `{model}`")

    out_root = make_run_dir(args.out_root, args.run_name, timestamped=not args.no_timestamp)
    write_run_metadata(
        out_root,
        command="tree-residual-deep",
        args=args,
        inputs={"model_registry": args.model_registry, "feature_path": args.feature_path, "base_models": BASE_MODELS},
        registry_paths=[args.model_registry],
    )

    valid_base = _base_frame(registry, "valid", args.score_col, args.label_col)
    test_base = _base_frame(registry, "test", args.score_col, args.label_col)
    feature_cols = _feature_columns(Path(args.feature_path), args.feature_limit)
    features = _load_features_for_frames(Path(args.feature_path), [valid_base, test_base], feature_cols)
    valid = _attach_features(valid_base, features)
    test = _attach_features(test_base, features)
    residual_train, residual_calib = _split_calibration_dates(valid, args.calib_valid_fraction)

    input_cols = [*META_COLUMNS, *feature_cols]
    model, payload = _train_residual_mlp(residual_train, residual_calib, input_cols, args, out_root / "residual_mlp")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    for frame in [residual_calib, valid, test]:
        x = _transform(frame, input_cols, payload["scaler"])
        frame["residual_pred"] = _predict(model, x, args.eval_batch_size, device)

    alpha_df = _alpha_grid(residual_calib, args)
    alpha_df.to_csv(out_root / "alpha_grid_calib.csv", index=False)
    best_alpha = float(alpha_df.iloc[0]["alpha"])

    entries: dict[str, dict[str, Any]] = {
        "fusion_rank_equal_gbdt": {
            "description": "报告主实验：LGB/XGB 等权 rank 融合基线。",
            "predictions": {},
        },
        "fusion_gbdt_residual_mlp": {
            "description": f"报告主实验：LGB/XGB rank 融合 + MLP 残差修正，alpha={best_alpha:g}。",
            "predictions": {},
        },
    }
    for alpha in args.alpha_grid:
        entries[f"fusion_gbdt_residual_mlp_{_alpha_suffix(alpha)}"] = {
            "description": f"报告主实验：LGB/XGB rank 融合 + MLP 残差修正，alpha={float(alpha):g}。",
            "predictions": {},
        }
    for split, frame in [("valid", valid), ("test", test)]:
        base_pred = frame["base_rank"].to_numpy(dtype=np.float32, copy=False)
        residual_signal = frame["residual_pred"].to_numpy(dtype=np.float32, copy=False)
        predictions = [
            ("fusion_rank_equal_gbdt", base_pred),
            ("fusion_gbdt_residual_mlp", base_pred + np.float32(best_alpha) * residual_signal),
        ]
        predictions.extend(
            (f"fusion_gbdt_residual_mlp_{_alpha_suffix(alpha)}", base_pred + np.float32(alpha) * residual_signal)
            for alpha in args.alpha_grid
        )
        for model_name, pred in predictions:
            split_dir = out_root / model_name / split
            split_dir.mkdir(parents=True, exist_ok=True)
            pred_path = split_dir / f"{split}_pred.parquet"
            pred_df = _prediction_frame(frame, pred, args.label_col)
            pred_df.to_parquet(pred_path, index=False)
            metrics = _write_prediction_metrics(pred_df, split, args.label_col, split_dir)
            entries[model_name]["predictions"][split] = str(pred_path)
            entries[model_name].setdefault("metrics", {})[split] = str(split_dir / f"{split}_metrics.json")
            print(
                json.dumps(
                    {"model": model_name, "split": split, "rows": int(len(frame)), "path": str(pred_path), "metrics": metrics},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    registry_path = out_root / "models_report_label1d_tree_residual.yaml"
    _write_registry(registry_path, registry, entries)
    summary = {
        "out_root": str(out_root),
        "registry": str(registry_path),
        "feature_count": len(feature_cols),
        "input_count": len(input_cols),
        "valid_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "residual_train_rows": int(len(residual_train)),
        "residual_calib_rows": int(len(residual_calib)),
        "best_alpha": best_alpha,
        "alpha_grid": str(out_root / "alpha_grid_calib.csv"),
        "model_path": str(out_root / "residual_mlp" / "residual_mlp.pt"),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved_summary": str(out_root / "summary.json"), "registry": str(registry_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run_cli()
