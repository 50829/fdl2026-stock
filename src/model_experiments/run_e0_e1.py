from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from src.data import (
    ProcessedConfig,
    ProcessedSplit,
    build_processed_splits,
    iter_processed_batches,
    iter_processed_sequence_feature_batches,
)
from src.data.processed import _load_cached_feature_frame
from src.evaluation import backtest_config_from_cfg, backtest_rolling_tranche, backtest_topk, ic_metrics
from src.models import build_model
from src.train import train as train_model
from src.utils import make_run_dir, read_yaml, write_json, write_run_metadata


EXPERIMENTS = {
    "e0": {
        "config": "configs/exp_e0_mlp_5d_rank_pilot.yaml",
        "name": "e0_mlp_5d_rank_pilot",
        "raw_return_col": "label_5d",
    },
    "e1": {
        "config": "configs/exp_e1_gru_5d_rank_pilot.yaml",
        "name": "e1_gru_5d_rank_pilot",
        "raw_return_col": "label_5d",
    },
    "e0_full": {
        "config": "configs/exp_e0_mlp_5d_rank.yaml",
        "name": "e0_mlp_5d_rank",
        "raw_return_col": "label_5d",
    },
    "e1_full": {
        "config": "configs/exp_e1_gru_5d_rank.yaml",
        "name": "e1_gru_5d_rank",
        "raw_return_col": "label_5d",
    },
    "e1_daily": {
        "config": "configs/exp_e1_gru_1d_rank_daily_pilot.yaml",
        "name": "e1_gru_1d_rank_daily_pilot",
        "raw_return_col": "label_1d",
    },
    "e1_daily_full": {
        "config": "configs/exp_e1_gru_1d_rank_daily.yaml",
        "name": "e1_gru_1d_rank_daily",
        "raw_return_col": "label_1d",
    },
}


def is_sequence_model(cfg: dict) -> bool:
    name = str(cfg.get("model", {}).get("name", "mlp")).strip().lower()
    return name in {
        "lstm",
        "transformer",
        "tf",
        "alstm",
        "tcn",
        "temporal_conv",
        "temporal_convolution",
        "patchtst",
        "patch_tst",
        "itransformer",
        "i_transformer",
    }


def load_checkpoint_model(cfg: dict, ckpt_path: str | Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    feature_cols = list(ckpt["feature_cols"])
    model = build_model(cfg, in_dim=len(feature_cols))
    model.load_state_dict(ckpt["model_state"])
    return ckpt, model, feature_cols


def resolve_warmup_start(pcfg: ProcessedConfig, start_date: str, seq_len: int) -> str:
    import pyarrow.dataset as ds

    key_trade, _ = pcfg.key_cols
    proc = Path(pcfg.processed_dir)
    dates = set()
    scan = ds.dataset(str(proc / pcfg.features_path), format="parquet").scanner(columns=[key_trade], batch_size=1 << 20)
    for batch in scan.to_reader():
        dates.update(str(x) for x in batch.column(0).to_pylist())
    ordered = sorted(dates)
    if not ordered:
        return str(start_date)
    idx = 0
    for i, d in enumerate(ordered):
        if d >= str(start_date):
            idx = i
            break
    warm_idx = max(0, idx - int(seq_len) + 1)
    return ordered[warm_idx]


def infer_device(device_text: str | None = None) -> torch.device:
    if device_text:
        return torch.device(device_text)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_split(
    cfg: dict,
    split_name: str,
    out_dir: Path,
    raw_return_col: str,
    device_text: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    pred_cfg = cfg.get("predict", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    processed_dir = str(data_cfg.get("processed_dir", "data/processed"))
    pcfg = ProcessedConfig(processed_dir=processed_dir)
    splits = build_processed_splits(pcfg, fallback=cfg)
    if split_name not in splits:
        raise ValueError(f"Unknown split: {split_name}")

    ckpt_path = pred_cfg.get("ckpt", train_cfg.get("save_path"))
    ckpt, model, feature_cols = load_checkpoint_model(cfg, ckpt_path)
    label_col = str(ckpt["label_col"])
    device = infer_device(device_text)
    model.to(device)
    model.eval()

    split = splits[split_name]
    batch_size = int(pred_cfg.get("batch_size", train_cfg.get("batch_size", 4096)))
    filter_in_universe = bool(pred_cfg.get("filter_in_universe", train_cfg.get("filter_in_universe", True)))
    cache_data = bool(pred_cfg.get("cache_data", train_cfg.get("cache_data", False)))

    if is_sequence_model(cfg):
        seq_len = int(cfg.get("model", {}).get("seq_len", cfg.get("sample", {}).get("lookback", 60)))
        warmup_start = resolve_warmup_start(pcfg, split.start_date, seq_len)
        iterator = iter_processed_sequence_feature_batches(
            pcfg,
            start_date=warmup_start,
            end_date=split.end_date,
            feature_cols=feature_cols,
            seq_len=seq_len,
            batch_size=batch_size,
            filter_in_universe=filter_in_universe,
            return_keys=True,
            use_tqdm=False,
            emit_start_date=split.start_date,
            cache_in_memory=cache_data,
        )
    else:
        iterator = iter_processed_batches(
            pcfg,
            split,
            feature_cols=feature_cols,
            label_col=label_col,
            batch_size=batch_size,
            filter_in_universe=filter_in_universe,
            return_keys=True,
            use_tqdm=False,
            cache_in_memory=cache_data,
        )

    rows = []
    with torch.no_grad():
        for batch in iterator:
            xb = torch.from_numpy(batch["X"]).to(device, non_blocking=True)
            pred = model(xb)
            row = {
                "trade_date": np.asarray(batch["trade_date"]).astype(str),
                "ts_code": np.asarray(batch["ts_code"]).astype(str),
                "pred": pred.detach().cpu().numpy().astype(np.float32, copy=False),
            }
            rows.append(pd.DataFrame(row))

    pred_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    bt_cfg = backtest_config_from_cfg(cfg)
    extra_label_cols = [bt_cfg.daily_return_col] if bt_cfg.mode == "rolling_tranche" else None
    if not pred_df.empty:
        pred_df = attach_labels(
            pcfg,
            pred_df,
            split,
            label_col=label_col,
            raw_return_col=raw_return_col,
            extra_label_cols=extra_label_cols,
        )
        pred_df = pred_df.dropna(subset=[label_col])

    n = int(len(pred_df))
    if n:
        diff = pred_df["pred"].to_numpy(dtype=np.float64) - pred_df[label_col].to_numpy(dtype=np.float64)
        mse = float(np.mean(diff * diff))
    else:
        mse = math.nan

    metrics = {
        "split": split_name,
        "samples": int(n),
        "mse": mse,
        "label_col": label_col,
        "raw_return_col": raw_return_col,
    }
    metrics.update(ic_metrics(pred_df, label_col=label_col))
    if bt_cfg.mode == "rolling_tranche":
        metrics.update(backtest_rolling_tranche(pred_df, cfg=bt_cfg))
    else:
        metrics.update(backtest_topk(pred_df, return_col=raw_return_col, cfg=bt_cfg))

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{split_name}_pred.parquet"
    pred_df.to_parquet(pred_path, index=False)
    write_json(out_dir / f"{split_name}_metrics.json", metrics)
    return pred_df, metrics


def attach_labels(
    pcfg: ProcessedConfig,
    pred_df: pd.DataFrame,
    split: ProcessedSplit,
    label_col: str,
    raw_return_col: str,
    extra_label_cols: list[str] | None = None,
) -> pd.DataFrame:
    import pyarrow.dataset as ds

    proc = Path(pcfg.processed_dir)
    key_trade, key_code = pcfg.key_cols
    l_path = proc / pcfg.labels_path
    date_filter = (ds.field(key_trade) >= split.start_date) & (ds.field(key_trade) <= split.end_date)
    label_cols = [key_trade, key_code, label_col, raw_return_col]
    for col in extra_label_cols or []:
        if col not in label_cols:
            label_cols.append(col)
    labels = (
        ds.dataset(str(l_path), format="parquet")
        .to_table(columns=label_cols, filter=date_filter)
        .to_pandas()
    )
    labels[key_trade] = labels[key_trade].astype(str)
    labels[key_code] = labels[key_code].astype(str)
    return pred_df.merge(labels, on=[key_trade, key_code], how="left")


def predict_split_no_label(
    cfg: dict,
    split_name: str,
    out_dir: Path,
    device_text: str | None = None,
) -> Path:
    pred_cfg = cfg.get("predict", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    processed_dir = str(data_cfg.get("processed_dir", "data/processed"))
    pcfg = ProcessedConfig(processed_dir=processed_dir)
    splits = build_processed_splits(pcfg, fallback=cfg)
    split = splits[split_name]

    ckpt_path = pred_cfg.get("ckpt", train_cfg.get("save_path"))
    _, model, feature_cols = load_checkpoint_model(cfg, ckpt_path)
    device = infer_device(device_text)
    model.to(device)
    model.eval()

    batch_size = int(pred_cfg.get("batch_size", train_cfg.get("batch_size", 4096)))
    filter_in_universe = bool(pred_cfg.get("filter_in_universe", train_cfg.get("filter_in_universe", True)))
    cache_data = bool(pred_cfg.get("cache_data", train_cfg.get("cache_data", False)))

    rows = []
    with torch.no_grad():
        if is_sequence_model(cfg):
            seq_len = int(cfg.get("model", {}).get("seq_len", cfg.get("sample", {}).get("lookback", 60)))
            start_date = str(pred_cfg.get("warmup_start_date") or resolve_warmup_start(pcfg, split.start_date, seq_len))
            iterator = iter_processed_sequence_feature_batches(
                pcfg,
                start_date=start_date,
                end_date=split.end_date,
                feature_cols=feature_cols,
                seq_len=seq_len,
                batch_size=batch_size,
                filter_in_universe=filter_in_universe,
                return_keys=True,
                use_tqdm=False,
                emit_start_date=split.start_date,
                cache_in_memory=cache_data,
            )
            for batch in iterator:
                xb = torch.from_numpy(batch["X"]).to(device, non_blocking=True)
                pred = model(xb).detach().cpu().numpy().astype(np.float32, copy=False)
                rows.append(
                    pd.DataFrame(
                        {
                            "trade_date": np.asarray(batch["trade_date"]).astype(str),
                            "ts_code": np.asarray(batch["ts_code"]).astype(str),
                            "pred": pred,
                        }
                    )
                )
        else:
            m = _load_cached_feature_frame(pcfg, split.start_date, split.end_date, feature_cols, filter_in_universe)
            for i in range(0, len(m), batch_size):
                part = m.iloc[i : i + batch_size]
                xb = torch.from_numpy(part[feature_cols].to_numpy(dtype=np.float32, copy=False)).to(device)
                pred = model(xb).detach().cpu().numpy().astype(np.float32, copy=False)
                rows.append(part[["trade_date", "ts_code"]].assign(pred=pred))

    pred_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["trade_date", "ts_code", "pred"])
    pred_df["rank"] = pred_df.groupby("trade_date")["pred"].rank(method="first", ascending=False).astype("int32")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split_name}_scores.parquet"
    pred_df.to_parquet(out_path, index=False)
    return out_path


def run_experiment(exp_key: str, config_path: str, out_root: Path, stages: Iterable[str], device: str | None) -> dict:
    cfg = read_yaml(config_path)
    name = EXPERIMENTS.get(exp_key, {}).get("name", Path(config_path).stem)
    raw_return_col = cfg.get("backtest", {}).get("return_col", EXPERIMENTS.get(exp_key, {}).get("raw_return_col", "label_5d"))
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {"experiment": exp_key, "config": config_path, "out_dir": str(out_dir)}

    if "train" in stages:
        t0 = time.perf_counter()
        train_model(cfg)
        summary["train_sec"] = time.perf_counter() - t0

    if "eval" in stages:
        eval_summary = {}
        for split_name in ["valid", "test"]:
            _, metrics = evaluate_split(cfg, split_name, out_dir / split_name, raw_return_col=raw_return_col, device_text=device)
            eval_summary[split_name] = metrics
            print(json.dumps({"experiment": exp_key, "split": split_name, "metrics": metrics}, ensure_ascii=False))
        summary["eval"] = eval_summary

    if "predict" in stages:
        score_paths = {}
        for split_name in ["test"]:
            path = predict_split_no_label(cfg, split_name, out_dir / split_name, device_text=device)
            score_paths[split_name] = str(path)
            print(json.dumps({"experiment": exp_key, "split": split_name, "scores": str(path)}, ensure_ascii=False))
        summary["scores"] = score_paths

    write_json(out_dir / "summary.json", summary)
    return summary


def run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=["e0", "e1"], choices=sorted(EXPERIMENTS))
    parser.add_argument("--stage", nargs="+", default=["train", "eval", "predict"], choices=["train", "eval", "predict"])
    parser.add_argument("--out-root", default="outputs/models")
    parser.add_argument("--run-name", default="sequence_e0_e1")
    parser.add_argument("--no-timestamp", action="store_true", help="Write to <out-root>/<run-name> instead of timestamping the run directory.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out_root = make_run_dir(args.out_root, args.run_name, timestamped=not args.no_timestamp)
    write_run_metadata(out_root, command="gru", args=args)
    summaries = []
    for exp_key in args.experiments:
        summaries.append(
            run_experiment(
                exp_key,
                config_path=EXPERIMENTS[exp_key]["config"],
                out_root=out_root,
                stages=args.stage,
                device=args.device,
            )
        )
    write_json(out_root / "e0_e1_summary.json", {"experiments": summaries})


if __name__ == "__main__":
    run_cli()
