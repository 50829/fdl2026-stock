"""Training entry point."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn, optim

from src.data import (
    ProcessedConfig,
    build_processed_splits,
    iter_processed_batches,
    iter_processed_sequence_batches,
    load_feature_columns,
)
from src.models import build_model


def _get_tqdm(enabled: bool):
    if not enabled:
        return None
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


def _pwrite(tqdm_mod, msg: str):
    if tqdm_mod is not None:
        tqdm_mod.write(msg)
    else:
        print(msg)


def set_seed(seed: int):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _infer_label_col(cfg: dict) -> str:
    task = cfg.get("task", {})
    if "label" in task:
        return str(task["label"])
    sample = cfg.get("sample", {})
    horizon = int(sample.get("horizon", 1))
    return f"label_{horizon}d"


def train(cfg: dict):
    train_cfg = cfg.get("train", {})
    seed = int(train_cfg.get("seed", cfg.get("seed", 2026)))
    set_seed(seed)

    use_tqdm = bool(train_cfg.get("use_tqdm", True))
    tqdm_mod = _get_tqdm(use_tqdm)

    data_cfg = cfg.get("data", {})
    processed_dir = str(data_cfg.get("processed_dir", "data/processed"))
    pcfg = ProcessedConfig(processed_dir=processed_dir)

    _pwrite(tqdm_mod, json.dumps({"stage": "load_meta", "processed_dir": processed_dir}, ensure_ascii=False))
    feature_cols = load_feature_columns(pcfg)
    label_col = _infer_label_col(cfg)
    splits = build_processed_splits(pcfg, fallback=cfg)
    if "train" not in splits or "valid" not in splits:
        raise ValueError("splits must contain train and valid")

    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", "mlp")).strip().lower()
    is_sequence_model = model_name in {"lstm", "transformer", "tf", "alstm"}

    in_dim = int(len(feature_cols))
    model = build_model(cfg, in_dim=in_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _pwrite(tqdm_mod, json.dumps({"stage": "device", "device": str(device)}, ensure_ascii=False))

    lr = float(train_cfg.get("learning_rate", train_cfg.get("lr", 1e-3)))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    epochs = int(train_cfg.get("epochs", 20))
    batch_size = int(train_cfg.get("batch_size", 256))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    filter_in_universe = bool(train_cfg.get("filter_in_universe", True))

    seq_len = int(model_cfg.get("seq_len", cfg.get("sample", {}).get("lookback", 60)))

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        tr_loss_sum = 0.0
        tr_n = 0

        if is_sequence_model:
            train_iter = iter_processed_sequence_batches(
                pcfg,
                splits["train"],
                feature_cols=feature_cols,
                label_col=label_col,
                seq_len=seq_len,
                batch_size=batch_size,
                filter_in_universe=filter_in_universe,
                return_keys=False,
                use_tqdm=use_tqdm,
                stage_desc="train_seq",
            )
        else:
            train_iter = iter_processed_batches(
                pcfg,
                splits["train"],
                feature_cols=feature_cols,
                label_col=label_col,
                batch_size=batch_size,
                filter_in_universe=filter_in_universe,
                return_keys=False,
                use_tqdm=use_tqdm,
                stage_desc="train_tab",
            )

        for batch in train_iter:
            xb = torch.from_numpy(batch["X"]).to(device, non_blocking=True)
            yb = torch.from_numpy(batch["y"]).to(device, non_blocking=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_loss_sum += float(loss.item()) * int(yb.shape[0])
            tr_n += int(yb.shape[0])

        tr_loss = tr_loss_sum / max(1, tr_n)

        model.eval()
        va_loss_sum = 0.0
        va_n = 0
        if is_sequence_model:
            valid_iter = iter_processed_sequence_batches(
                pcfg,
                splits["valid"],
                feature_cols=feature_cols,
                label_col=label_col,
                seq_len=seq_len,
                batch_size=batch_size,
                filter_in_universe=filter_in_universe,
                return_keys=False,
                use_tqdm=use_tqdm,
                stage_desc="valid_seq",
            )
        else:
            valid_iter = iter_processed_batches(
                pcfg,
                splits["valid"],
                feature_cols=feature_cols,
                label_col=label_col,
                batch_size=batch_size,
                filter_in_universe=filter_in_universe,
                return_keys=False,
                use_tqdm=use_tqdm,
                stage_desc="valid_tab",
            )
        with torch.no_grad():
            for batch in valid_iter:
                xb = torch.from_numpy(batch["X"]).to(device, non_blocking=True)
                yb = torch.from_numpy(batch["y"]).to(device, non_blocking=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_loss_sum += float(loss.item()) * int(yb.shape[0])
                va_n += int(yb.shape[0])
        va_loss = va_loss_sum / max(1, va_n)
        elapsed = time.perf_counter() - t0

        _pwrite(
            tqdm_mod,
            json.dumps({"epoch": epoch, "epochs": epochs, "train_loss": tr_loss, "val_loss": va_loss, "sec": elapsed}, ensure_ascii=False),
        )

    save_path = Path(train_cfg.get("save_path", "outputs/models/ckpt.pt"))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "feature_cols": feature_cols,
        "label_col": label_col,
        "cfg": cfg,
    }
    torch.save(ckpt, save_path)
    _pwrite(tqdm_mod, json.dumps({"saved": str(save_path)}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
