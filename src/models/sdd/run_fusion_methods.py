from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import ProcessedConfig
from src.models.mlp import MLPModel
from src.models.sdd.run_e0_e1 import write_json
from src.models.sdd.run_gbdt import evaluate_predictions, load_tabular_frame, predict_model, train_lightgbm
from src.models.sdd.run_gbdt_walkforward import resolve_features
from src.train import set_seed


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


def read_pred(path: str | Path, name: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["trade_date"] = df["trade_date"].astype(str)
    df["ts_code"] = df["ts_code"].astype(str)
    return df.rename(columns={"pred": f"pred_{name}"})


def concat_oof(root: str, model: str, years: list[int], name: str) -> pd.DataFrame:
    parts = []
    for y in years:
        p = Path(root) / model / f"expanding_valid{y}" / "valid_pred.parquet"
        parts.append(read_pred(p, name))
    return pd.concat(parts, ignore_index=True)


def merge_model_preds(lgb: pd.DataFrame, xgb: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    keep = ["trade_date", "ts_code", f"pred_lgb", args.target, args.raw_return_col, args.daily_return_col]
    df = lgb[keep].merge(xgb[["trade_date", "ts_code", "pred_xgb"]], on=["trade_date", "ts_code"], how="inner")
    df["rank_lgb"] = df.groupby("trade_date")["pred_lgb"].rank(method="average", pct=True)
    df["rank_xgb"] = df.groupby("trade_date")["pred_xgb"].rank(method="average", pct=True)
    df["pred_mean"] = 0.5 * (df["pred_lgb"] + df["pred_xgb"])
    df["rank_mean"] = 0.5 * (df["rank_lgb"] + df["rank_xgb"])
    df["pred_diff"] = df["pred_lgb"] - df["pred_xgb"]
    df["rank_diff"] = df["rank_lgb"] - df["rank_xgb"]
    return df.dropna(subset=[args.target]).reset_index(drop=True)


def feature_cols_for_mlp(df: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    cols = ["pred_lgb", "pred_xgb", "rank_lgb", "rank_xgb", "pred_mean", "rank_mean", "pred_diff", "rank_diff"]
    cols.extend(getattr(args, "_fusion_raw_cols", []))
    return cols


def meta_features(df: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    cols = feature_cols_for_mlp(df, args)
    return df[cols].to_numpy(dtype=np.float32, copy=False)


def pred_frame(df: pd.DataFrame, pred: np.ndarray, args: argparse.Namespace) -> pd.DataFrame:
    out = df[["trade_date", "ts_code", args.target, args.raw_return_col, args.daily_return_col]].copy()
    out["pred"] = pred.astype(np.float32, copy=False)
    return out


def eval_and_save(name: str, split: str, df: pd.DataFrame, pred: np.ndarray, args: argparse.Namespace, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pf = pred_frame(df, pred, args)
    metrics = evaluate_predictions(pf, args.target, args)
    metrics["experiment"] = name
    metrics["split"] = split
    write_json(out_dir / f"{split}_metrics.json", metrics)
    print(json.dumps({"experiment": name, "split": split, "metrics": metrics}, ensure_ascii=False), flush=True)
    return metrics


def run_ridge(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, args: argparse.Namespace, out_root: Path) -> dict:
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.asarray(args.ridge_alphas, dtype=np.float64)))
    model.fit(meta_features(train_df, args), train_df[args.target].to_numpy(dtype=np.float32))
    out_dir = out_root / "stacking_ridge"
    valid_pred = model.predict(meta_features(valid_df, args)).astype(np.float32)
    test_pred = model.predict(meta_features(test_df, args)).astype(np.float32)
    summary = {
        "valid": eval_and_save("stacking_ridge", "valid", valid_df, valid_pred, args, out_dir / "valid"),
        "test": eval_and_save("stacking_ridge", "test", test_df, test_pred, args, out_dir / "test"),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def standardize(train_x: np.ndarray, *xs: np.ndarray):
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, [(x - mean) / std for x in xs], {"mean": mean.tolist(), "std": std.tolist()}


def train_simple_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    args: argparse.Namespace,
    in_dim: int | None = None,
) -> tuple[nn.Module, dict]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    input_dim = int(in_dim or train_x.shape[1])
    if args.mlp_arch == "deep_ln":
        model: nn.Module = DeepMLP(in_dim=input_dim, hidden=args.mlp_hidden, dropout=args.mlp_dropout).to(device)
    else:
        model = MLPModel(in_dim=input_dim, hidden=args.mlp_hidden, dropout=args.mlp_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_weight_decay)
    loss_fn = nn.SmoothL1Loss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x.astype(np.float32)), torch.from_numpy(train_y.astype(np.float32))),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    best_loss = math.inf
    best_state = None
    bad = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_sum = 0.0
        tr_n = 0
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
            tr_sum += float(loss.item()) * int(yb.shape[0])
            tr_n += int(yb.shape[0])
        valid_pred = predict_torch(model, valid_x, args.eval_batch_size, device)
        valid_loss = float(np.mean((valid_pred.astype(np.float64) - valid_y.astype(np.float64)) ** 2))
        row = {"epoch": epoch, "train_loss": tr_sum / max(1, tr_n), "valid_mse": valid_loss}
        history.append(row)
        print(json.dumps({"mlp_train": row}, ensure_ascii=False), flush=True)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, {"history": history, "best_valid_mse": best_loss}


def predict_torch(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i : i + batch_size].astype(np.float32, copy=False)).to(device)
            out.append(model(xb).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0) if out else np.empty((0,), dtype=np.float32)


def run_stacking_mlp(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, args: argparse.Namespace, out_root: Path) -> dict:
    train_x, [valid_x, test_x], scaler = standardize(meta_features(train_df, args), meta_features(valid_df, args), meta_features(test_df, args))
    train_y = train_df[args.target].to_numpy(dtype=np.float32)
    valid_y = valid_df[args.target].to_numpy(dtype=np.float32)
    model, hist = train_simple_mlp(train_x, train_y, valid_x.astype(np.float32), valid_y, args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = out_root / "stacking_mlp"
    valid_pred = predict_torch(model, valid_x.astype(np.float32), args.eval_batch_size, device)
    test_pred = predict_torch(model, test_x.astype(np.float32), args.eval_batch_size, device)
    summary = {
        "input_columns": feature_cols_for_mlp(train_df, args),
        "mlp_arch": args.mlp_arch,
        "history": hist,
        "scaler": scaler,
        "valid": eval_and_save("stacking_mlp", "valid", valid_df, valid_pred, args, out_dir / "valid"),
        "test": eval_and_save("stacking_mlp", "test", test_df, test_pred, args, out_dir / "test"),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def add_residual_rank(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out["residual"] = out[args.target] - out["pred_lgb"]
    out["residual_rank"] = out.groupby("trade_date")["residual"].rank(method="average", pct=True).astype(np.float32) - 0.5
    return out


def run_residual_rank_mlp(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    out_root: Path,
) -> dict:
    train_df = add_residual_rank(train_df, args)
    valid_df = add_residual_rank(valid_df, args)
    test_df = add_residual_rank(test_df, args)
    train_x, [valid_x, test_x], scaler = standardize(meta_features(train_df, args), meta_features(valid_df, args), meta_features(test_df, args))
    model, hist = train_simple_mlp(
        train_x.astype(np.float32),
        train_df["residual_rank"].to_numpy(dtype=np.float32),
        valid_x.astype(np.float32),
        valid_df["residual_rank"].to_numpy(dtype=np.float32),
        args,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = out_root / "residual_rank_mlp"
    valid_resid = predict_torch(model, valid_x.astype(np.float32), args.eval_batch_size, device)
    test_resid = predict_torch(model, test_x.astype(np.float32), args.eval_batch_size, device)
    summary = {"history": hist, "scaler": scaler, "input_columns": feature_cols_for_mlp(train_df, args), "mlp_arch": args.mlp_arch}
    for split, df, resid in [("valid", valid_df, valid_resid), ("test", test_df, test_resid)]:
        split_rows = []
        for alpha in args.alpha_grid:
            pred = df["pred_lgb"].to_numpy(dtype=np.float32) + float(alpha) * resid
            m = evaluate_predictions(pred_frame(df, pred, args), args.target, args)
            split_rows.append({"alpha": float(alpha), **m})
        grid = pd.DataFrame(split_rows).sort_values(["icir", "ic_mean"], ascending=False, kind="mergesort")
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        grid.to_csv(out_dir / split / f"{split}_alpha_grid.csv", index=False)
        best = grid.iloc[0].to_dict()
        write_json(out_dir / split / f"{split}_metrics.json", best)
        summary[split] = best
        print(json.dumps({"experiment": "residual_rank_mlp", "split": split, "best": best}, ensure_ascii=False), flush=True)
    write_json(out_dir / "summary.json", summary)
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_columns": summary["input_columns"],
            "scaler": scaler,
            "mlp_arch": args.mlp_arch,
            "mlp_hidden": args.mlp_hidden,
            "mlp_dropout": args.mlp_dropout,
            "mlp_lr": args.mlp_lr,
            "mlp_weight_decay": args.mlp_weight_decay,
            "alpha_grid": [float(x) for x in args.alpha_grid],
            "best_valid": summary.get("valid", {}),
            "best_test": summary.get("test", {}),
            "history": hist,
            "target": args.target,
            "base_model": "lightgbm_top40_pred_lgb",
        },
        out_dir / "residual_rank_mlp.pt",
    )
    return summary


class LeafMLP(nn.Module):
    def __init__(self, n_leaf_tokens: int, n_trees: int, emb_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(n_leaf_tokens, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(n_trees * emb_dim + 1, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, leaf_ids: torch.Tensor, base_pred: torch.Tensor) -> torch.Tensor:
        x = self.emb(leaf_ids).flatten(1)
        x = torch.cat([x, base_pred[:, None]], dim=1)
        return self.net(x).squeeze(-1)


def train_leaf_mlp(
    train_leaf: np.ndarray,
    train_base: np.ndarray,
    train_y: np.ndarray,
    valid_leaf: np.ndarray,
    valid_base: np.ndarray,
    valid_y: np.ndarray,
    args: argparse.Namespace,
) -> tuple[LeafMLP, dict]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    n_trees = int(train_leaf.shape[1])
    n_tokens = int(max(train_leaf.max(), valid_leaf.max()) + 1)
    model = LeafMLP(n_tokens, n_trees, args.leaf_emb_dim, args.mlp_hidden, args.mlp_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_weight_decay)
    loss_fn = nn.SmoothL1Loss()
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_leaf.astype(np.int64)),
            torch.from_numpy(train_base.astype(np.float32)),
            torch.from_numpy(train_y.astype(np.float32)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    best_loss = math.inf
    best_state = None
    bad = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_sum = 0.0
        tr_n = 0
        for leaf, base, yb in loader:
            leaf = leaf.to(device, non_blocking=True)
            base = base.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(leaf, base)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_sum += float(loss.item()) * int(yb.shape[0])
            tr_n += int(yb.shape[0])
        valid_pred = predict_leaf(model, valid_leaf, valid_base, args.eval_batch_size, device)
        valid_loss = float(np.mean((valid_pred.astype(np.float64) - valid_y.astype(np.float64)) ** 2))
        row = {"epoch": epoch, "train_loss": tr_sum / max(1, tr_n), "valid_mse": valid_loss}
        history.append(row)
        print(json.dumps({"leaf_mlp_train": row}, ensure_ascii=False), flush=True)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, {"history": history, "best_valid_mse": best_loss, "n_trees": n_trees, "n_tokens": n_tokens}


def predict_leaf(model: LeafMLP, leaf_ids: np.ndarray, base_pred: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(leaf_ids), batch_size):
            leaf = torch.from_numpy(leaf_ids[i : i + batch_size].astype(np.int64, copy=False)).to(device)
            base = torch.from_numpy(base_pred[i : i + batch_size].astype(np.float32, copy=False)).to(device)
            out.append(model(leaf, base).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0) if out else np.empty((0,), dtype=np.float32)


def split_range(name: str, start: str, end: str):
    from src.data import ProcessedSplit

    return ProcessedSplit(name=name, start_date=start, end_date=end)


def attach_raw_features(
    df: pd.DataFrame,
    pcfg: ProcessedConfig,
    raw_cols: list[str],
    args: argparse.Namespace,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    if not raw_cols:
        return df
    label_cols = [args.target, args.raw_return_col, args.daily_return_col]
    raw = load_tabular_frame(pcfg, split_range("raw", start_date, end_date), raw_cols, label_cols, True)
    raw = raw[["trade_date", "ts_code"] + raw_cols].copy()
    raw["trade_date"] = raw["trade_date"].astype(str)
    raw["ts_code"] = raw["ts_code"].astype(str)
    return df.merge(raw, on=["trade_date", "ts_code"], how="inner")


def offset_leaf_ids(leaf: np.ndarray, num_leaves: int = 64) -> np.ndarray:
    offsets = np.arange(leaf.shape[1], dtype=np.int64) * int(num_leaves)
    return leaf.astype(np.int64) + offsets[None, :]


def run_leaf_embedding(args: argparse.Namespace, out_root: Path) -> dict:
    pcfg = ProcessedConfig(args.processed_dir)
    feature_cols = resolve_features(pcfg, args.feature_list)
    label_cols = [args.target, args.raw_return_col, args.daily_return_col]
    train_base = load_tabular_frame(pcfg, split_range("train", "20160101", "20201231"), feature_cols, label_cols, True)
    valid_base = load_tabular_frame(pcfg, split_range("valid", "20210101", "20211231"), feature_cols, label_cols, True)
    booster = train_lightgbm(train_base, valid_base, feature_cols, args.target, args)[0]
    train_df = load_tabular_frame(pcfg, split_range("leaf_train", "20210101", "20231231"), feature_cols, label_cols, True)
    valid_df = load_tabular_frame(pcfg, split_range("leaf_valid", "20240101", "20241231"), feature_cols, label_cols, True)
    test_df = load_tabular_frame(pcfg, split_range("leaf_test", "20250101", "20260518"), feature_cols, label_cols, True)
    def leaves_and_base(df):
        x = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        base = booster.predict(x).astype(np.float32)
        leaf = booster.predict(x, pred_leaf=True)
        return offset_leaf_ids(np.asarray(leaf), num_leaves=max(64, args.num_leaves + 1)), base
    tr_leaf, tr_base = leaves_and_base(train_df)
    va_leaf, va_base = leaves_and_base(valid_df)
    te_leaf, te_base = leaves_and_base(test_df)
    model, hist = train_leaf_mlp(
        tr_leaf,
        tr_base,
        train_df[args.target].to_numpy(dtype=np.float32),
        va_leaf,
        va_base,
        valid_df[args.target].to_numpy(dtype=np.float32),
        args,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    valid_pred = predict_leaf(model, va_leaf, va_base, args.eval_batch_size, device)
    test_pred = predict_leaf(model, te_leaf, te_base, args.eval_batch_size, device)
    out_dir = out_root / "leaf_embedding_mlp"
    summary = {
        "history": hist,
        "valid": eval_and_save("leaf_embedding_mlp", "valid", valid_df, valid_pred, args, out_dir / "valid"),
        "test": eval_and_save("leaf_embedding_mlp", "test", test_df, test_pred, args, out_dir / "test"),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-root", default="outputs/sdd_fusion_methods")
    parser.add_argument("--feature-list", default="outputs/sdd_feature_selection/features/lightgbm_top40.txt")
    parser.add_argument("--fusion-feature-mode", choices=["meta", "meta_top", "meta_full"], default="meta")
    parser.add_argument("--target", default="label_5d__cs_rank")
    parser.add_argument("--raw-return-col", default="label_5d")
    parser.add_argument("--daily-return-col", default="label_1d")
    parser.add_argument("--lgb-oof-root", default="outputs/sdd_oof_preds_lgb_top40")
    parser.add_argument("--xgb-oof-root", default="outputs/sdd_oof_preds_xgb_top40")
    parser.add_argument("--valid-lgb", default="outputs/sdd_feature_selection/lightgbm_top40/lightgbm/valid/valid_pred.parquet")
    parser.add_argument("--test-lgb", default="outputs/sdd_feature_selection/lightgbm_top40/lightgbm/test/test_pred.parquet")
    parser.add_argument("--valid-xgb", default="outputs/sdd_feature_selection/xgboost_top40/xgboost/valid/valid_pred.parquet")
    parser.add_argument("--test-xgb", default="outputs/sdd_feature_selection/xgboost_top40/xgboost/test/test_pred.parquet")
    parser.add_argument("--experiments", nargs="+", default=["ridge", "stacking_mlp", "residual_rank", "leaf_embedding"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--mlp-arch", choices=["shallow", "deep_ln"], default="shallow")
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--grad-clip", type=float, default=3.0)
    parser.add_argument("--ridge-alphas", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--leaf-emb-dim", type=int, default=4)
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
    parser.add_argument("--n-hold", type=int, default=20)
    parser.add_argument("--k-rotate", type=int, default=5)
    parser.add_argument("--step-days", type=int, default=5)
    parser.add_argument("--tranche-size", type=int, default=4)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    summaries = {}
    pcfg = ProcessedConfig(args.processed_dir)
    top_feature_cols = resolve_features(pcfg, args.feature_list)
    if args.fusion_feature_mode == "meta_top":
        raw_cols = top_feature_cols
    elif args.fusion_feature_mode == "meta_full":
        raw_cols = resolve_features(pcfg, None)
    else:
        raw_cols = []
    args._fusion_raw_cols = raw_cols
    oof_lgb = concat_oof(args.lgb_oof_root, "lightgbm", [2021, 2022, 2023], "lgb")
    oof_xgb = concat_oof(args.xgb_oof_root, "xgboost", [2021, 2022, 2023], "xgb")
    train_df = merge_model_preds(oof_lgb, oof_xgb, args)
    valid_df = merge_model_preds(read_pred(args.valid_lgb, "lgb"), read_pred(args.valid_xgb, "xgb"), args)
    test_df = merge_model_preds(read_pred(args.test_lgb, "lgb"), read_pred(args.test_xgb, "xgb"), args)
    train_df = attach_raw_features(train_df, pcfg, raw_cols, args, "20210101", "20231231")
    valid_df = attach_raw_features(valid_df, pcfg, raw_cols, args, "20240101", "20241231")
    test_df = attach_raw_features(test_df, pcfg, raw_cols, args, "20250101", "20260518")

    if "ridge" in args.experiments:
        summaries["ridge"] = run_ridge(train_df, valid_df, test_df, args, out_root)
    if "stacking_mlp" in args.experiments:
        summaries["stacking_mlp"] = run_stacking_mlp(train_df, valid_df, test_df, args, out_root)
    if "residual_rank" in args.experiments:
        summaries["residual_rank"] = run_residual_rank_mlp(train_df, valid_df, test_df, args, out_root)
    if "leaf_embedding" in args.experiments:
        summaries["leaf_embedding"] = run_leaf_embedding(args, out_root)
    write_json(out_root / "summary.json", summaries)


if __name__ == "__main__":
    main()
