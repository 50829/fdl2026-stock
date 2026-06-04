from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn, optim

from src.data import ProcessedConfig, build_processed_splits, iter_processed_sequence_labeled_feature_batches, load_feature_columns
from src.data.feature_meta import read_feature_meta, resolve_feature_columns, read_parquet_feature_columns
from src.models import build_model
from src.train import set_seed
from src.models.sdd.run_e0_e1 import evaluate_split, resolve_warmup_start
from src.utils import write_json


BASE_CFG = {
    "data": {"processed_dir": "data/processed_pilot"},
    "task": {"label": "label_5d__cs_rank"},
    "model": {
        "name": "alstm",
        "seq_len": 60,
        "input_dim": 112,
        "hidden_size": 128,
        "num_layers": 2,
        "rnn_type": "GRU",
        "dropout": 0.2,
        "attention_hidden_ratio": 0.5,
        "use_attention": True,
    },
    "train": {
        "seed": 2026,
        "batch_size": 4096,
        "epochs": 8,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "grad_clip": 3.0,
        "filter_in_universe": True,
        "cache_data": True,
        "loss": "smooth_l1",
        "patience": 2,
        "min_delta": 0.0,
        "use_tqdm": False,
    },
    "predict": {"batch_size": 4096, "use_tqdm": False},
}


EXPERIMENTS = {
    "base_attn_h128_l2": {},
    "hidden64": {"model": {"hidden_size": 64, "input_dim": 112}},
    "layer1": {"model": {"num_layers": 1}},
    "layer1_input_layernorm": {"model": {"num_layers": 1, "input_layernorm": True}},
    "layer1_hidden_layernorm": {"model": {"num_layers": 1, "hidden_layernorm": True}},
    "layer1_input_hidden_layernorm": {
        "model": {"num_layers": 1, "input_layernorm": True, "hidden_layernorm": True}
    },
    "no_attention": {"model": {"use_attention": False}},
    "core_features": {
        "features": {
            "mode": "groups",
            "groups": ["core_price", "volume_liquidity", "momentum_ma", "volatility", "ts_zscore"],
        }
    },
    "loss_layer1_smooth_l1": {"model": {"num_layers": 1}},
    "loss_layer1_smooth_l1_corr": {
        "model": {"num_layers": 1},
        "train": {"loss": "smooth_l1_corr", "corr_lambda": 0.05},
    },
    "loss_layer1_mse_corr": {
        "model": {"num_layers": 1},
        "train": {"loss": "mse_corr", "corr_lambda": 0.05},
    },
    "layer1_lb20": {"model": {"num_layers": 1, "seq_len": 20}},
    "layer1_lb30": {"model": {"num_layers": 1, "seq_len": 30}},
    "layer1_core_features": {
        "model": {"num_layers": 1},
        "features": {
            "mode": "groups",
            "groups": ["core_price", "volume_liquidity", "momentum_ma", "volatility", "ts_zscore"],
        },
    },
    "tcn_lb20": {
        "model": {
            "name": "tcn",
            "seq_len": 20,
            "channels": 64,
            "levels": 3,
            "kernel_size": 3,
            "dropout": 0.2,
            "use_attention": False,
        }
    },
    "tcn_lb30": {
        "model": {
            "name": "tcn",
            "seq_len": 30,
            "channels": 64,
            "levels": 3,
            "kernel_size": 3,
            "dropout": 0.2,
            "use_attention": False,
        }
    },
    "tcn_lb60": {
        "model": {
            "name": "tcn",
            "seq_len": 60,
            "channels": 64,
            "levels": 4,
            "kernel_size": 3,
            "dropout": 0.2,
            "use_attention": False,
        }
    },
    "tcn_light_lb30": {
        "model": {
            "name": "tcn",
            "seq_len": 30,
            "channels": 32,
            "levels": 3,
            "kernel_size": 3,
            "dropout": 0.2,
            "use_attention": False,
        }
    },
    "tcn_light_attn_lb30": {
        "model": {
            "name": "tcn",
            "seq_len": 30,
            "channels": 32,
            "levels": 3,
            "kernel_size": 3,
            "dropout": 0.2,
            "use_attention": True,
        }
    },
    "tcn_light_lb60": {
        "model": {
            "name": "tcn",
            "seq_len": 60,
            "channels": 32,
            "levels": 4,
            "kernel_size": 3,
            "dropout": 0.2,
            "use_attention": False,
        }
    },
}


def deep_update(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def select_feature_columns(pcfg: ProcessedConfig, cfg: dict) -> list[str]:
    feature_cfg = cfg.get("features", {})
    if not feature_cfg:
        return load_feature_columns(pcfg)

    meta_path = Path(pcfg.processed_dir) / pcfg.feature_meta_path
    feature_path = Path(pcfg.processed_dir) / pcfg.features_path
    meta = read_feature_meta(meta_path)
    parquet_cols = read_parquet_feature_columns(feature_path)
    cols = resolve_feature_columns(
        meta,
        parquet_cols,
        mode=str(feature_cfg.get("mode", "default")),
        groups=feature_cfg.get("groups"),
        columns=feature_cfg.get("columns"),
    )
    return cols


def build_loss(name: str):
    name = str(name).lower()
    if name in {"smooth_l1", "huber"}:
        return nn.SmoothL1Loss()
    if name in {"mse", "l2"}:
        return nn.MSELoss()
    if name in {"smooth_l1_corr", "huber_corr"}:
        return nn.SmoothL1Loss()
    if name in {"mse_corr", "l2_corr"}:
        return nn.MSELoss()
    raise ValueError(f"unknown loss: {name}")


def pearson_corr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    pred_std = torch.sqrt(torch.mean(pred_centered * pred_centered) + 1e-8)
    target_std = torch.sqrt(torch.mean(target_centered * target_centered) + 1e-8)
    corr = torch.mean(pred_centered * target_centered) / (pred_std * target_std)
    return 1.0 - corr


def compute_loss(pred: torch.Tensor, target: torch.Tensor, base_loss, train_cfg: dict) -> torch.Tensor:
    name = str(train_cfg.get("loss", "mse")).lower()
    loss = base_loss(pred, target)
    if name in {"smooth_l1_corr", "huber_corr", "mse_corr", "l2_corr"}:
        loss = loss + float(train_cfg.get("corr_lambda", 0.05)) * pearson_corr_loss(pred, target)
    return loss


def train_one(cfg: dict, out_dir: Path) -> dict:
    train_cfg = cfg["train"]
    set_seed(int(train_cfg.get("seed", 2026)))

    pcfg = ProcessedConfig(processed_dir=str(cfg["data"]["processed_dir"]))
    splits = build_processed_splits(pcfg, fallback=cfg)
    feature_cols = select_feature_columns(pcfg, cfg)
    cfg["model"]["input_dim"] = len(feature_cols)

    label_col = str(cfg["task"]["label"])
    model = build_model(cfg, in_dim=len(feature_cols))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    loss_fn = build_loss(str(train_cfg.get("loss", "mse")))

    epochs = int(train_cfg.get("epochs", 8))
    patience = int(train_cfg.get("patience", 2))
    min_delta = float(train_cfg.get("min_delta", 0.0))
    batch_size = int(train_cfg.get("batch_size", 4096))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    filter_in_universe = bool(train_cfg.get("filter_in_universe", True))
    cache_data = bool(train_cfg.get("cache_data", True))
    seq_len = int(cfg["model"].get("seq_len", 60))

    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    best_path = out_dir / "best.pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        tr_sum = 0.0
        tr_n = 0
        train_iter = iter_processed_sequence_labeled_feature_batches(
            pcfg,
            start_date=splits["train"].start_date,
            end_date=splits["train"].end_date,
            emit_start_date=splits["train"].start_date,
            feature_cols=feature_cols,
            label_col=label_col,
            seq_len=seq_len,
            batch_size=batch_size,
            filter_in_universe=filter_in_universe,
            return_keys=False,
            cache_in_memory=cache_data,
        )
        for batch in train_iter:
            xb = torch.from_numpy(batch["X"]).to(device, non_blocking=True)
            yb = torch.from_numpy(batch["y"]).to(device, non_blocking=True)
            pred = model(xb)
            loss = compute_loss(pred, yb, loss_fn, train_cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_sum += float(loss.item()) * int(yb.shape[0])
            tr_n += int(yb.shape[0])

        model.eval()
        va_sum = 0.0
        va_n = 0
        valid_warmup_start = resolve_warmup_start(pcfg, splits["valid"].start_date, seq_len)
        valid_iter = iter_processed_sequence_labeled_feature_batches(
            pcfg,
            start_date=valid_warmup_start,
            end_date=splits["valid"].end_date,
            emit_start_date=splits["valid"].start_date,
            feature_cols=feature_cols,
            label_col=label_col,
            seq_len=seq_len,
            batch_size=batch_size,
            filter_in_universe=filter_in_universe,
            return_keys=False,
            cache_in_memory=cache_data,
        )
        with torch.no_grad():
            for batch in valid_iter:
                xb = torch.from_numpy(batch["X"]).to(device, non_blocking=True)
                yb = torch.from_numpy(batch["y"]).to(device, non_blocking=True)
                pred = model(xb)
                loss = compute_loss(pred, yb, loss_fn, train_cfg)
                va_sum += float(loss.item()) * int(yb.shape[0])
                va_n += int(yb.shape[0])

        tr_loss = tr_sum / max(1, tr_n)
        va_loss = va_sum / max(1, va_n)
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "valid_loss": va_loss,
            "sec": time.perf_counter() - t0,
        }
        history.append(row)
        print(json.dumps({"train": row, "model": cfg["model"]}, ensure_ascii=False), flush=True)

        if va_loss < best_loss - min_delta:
            best_loss = va_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_cols": feature_cols,
                    "label_col": label_col,
                    "cfg": cfg,
                    "best_epoch": best_epoch,
                    "best_valid_loss": best_loss,
                    "history": history,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    cfg.setdefault("predict", {})["ckpt"] = str(best_path)
    cfg["train"]["save_path"] = str(best_path)
    write_json(out_dir / "train_history.json", {"history": history, "best_epoch": best_epoch, "best_valid_loss": best_loss})
    return {"best_path": str(best_path), "best_epoch": best_epoch, "best_valid_loss": best_loss, "history": history}


def run_ablation(experiment: str, out_root: Path, processed_dir: str | None = None, epochs: int | None = None) -> dict:
    cfg = deep_update(BASE_CFG, EXPERIMENTS[experiment])
    if processed_dir is not None:
        cfg["data"]["processed_dir"] = str(processed_dir)
    if epochs is not None:
        cfg["train"]["epochs"] = int(epochs)
    out_dir = out_root / experiment
    train_summary = train_one(cfg, out_dir)
    _, metrics = evaluate_split(cfg, "valid", out_dir / "valid", raw_return_col="label_5d")
    summary = {"experiment": experiment, "train": train_summary, "valid": metrics, "config": cfg}
    write_json(out_dir / "summary.json", summary)
    print(json.dumps({"experiment": experiment, "valid": metrics, "best_epoch": train_summary["best_epoch"]}, ensure_ascii=False), flush=True)
    return summary


def run_feature_list_ablation(
    base_experiment: str,
    feature_list: str | Path,
    custom_name: str,
    out_root: Path,
    processed_dir: str | None = None,
    epochs: int | None = None,
) -> dict:
    cfg = deep_update(BASE_CFG, EXPERIMENTS[base_experiment])
    columns = [line.strip() for line in Path(feature_list).read_text(encoding="utf-8").splitlines() if line.strip()]
    cfg["features"] = {"mode": "explicit", "columns": columns}
    if processed_dir is not None:
        cfg["data"]["processed_dir"] = str(processed_dir)
    if epochs is not None:
        cfg["train"]["epochs"] = int(epochs)
    out_dir = out_root / custom_name
    train_summary = train_one(cfg, out_dir)
    eval_summary = {}
    for split in ["valid", "test"]:
        _, metrics = evaluate_split(cfg, split, out_dir / split, raw_return_col="label_5d")
        eval_summary[split] = metrics
    summary = {
        "experiment": custom_name,
        "base_experiment": base_experiment,
        "feature_list": str(feature_list),
        "train": train_summary,
        "eval": eval_summary,
        "config": cfg,
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps({"experiment": custom_name, "eval": eval_summary, "best_epoch": train_summary["best_epoch"]}, ensure_ascii=False), flush=True)
    return summary


def run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=sorted(EXPERIMENTS))
    parser.add_argument("--out-root", default="outputs/sdd_ablation")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--feature-list", default=None)
    parser.add_argument("--custom-name", default=None)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    if args.feature_list:
        base_experiment = args.experiments[0]
        custom_name = args.custom_name or f"{base_experiment}_feature_list"
        summaries = [
            run_feature_list_ablation(
                base_experiment,
                args.feature_list,
                custom_name,
                out_root,
                processed_dir=args.processed_dir,
                epochs=args.epochs,
            )
        ]
    else:
        summaries = [run_ablation(exp, out_root, processed_dir=args.processed_dir, epochs=args.epochs) for exp in args.experiments]
    write_json(out_root / "summary.json", {"experiments": summaries})


if __name__ == "__main__":
    run_cli()
