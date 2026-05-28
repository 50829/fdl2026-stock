from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader

from data_utils import filter_stock_pool, load_daily_dir, load_panel_csv, merge_on_keys, normalize_date_str, split_by_date
from dataset import PanelSeqDataset
from features import add_basic_tech_features, add_future_return_label, select_feature_and_label_cols
from models import build_model


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


def _load_data(cfg: dict) -> torch.utils.data.Dataset:
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    use_tqdm = bool(train_cfg.get("use_tqdm", False))
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
        df = load_daily_dir(
            daily_dir,
            start_date,
            end_date,
            usecols=usecols,
            limit_files=data_cfg.get("limit_files"),
            use_tqdm=use_tqdm,
            tqdm_desc="read daily",
        )

        for extra in data_cfg.get("extra_daily_dirs", []) or []:
            extra_dir = extra.get("dir")
            if not extra_dir:
                continue
            extra_usecols = extra.get("usecols")
            ex = load_daily_dir(
                extra_dir,
                start_date,
                end_date,
                usecols=extra_usecols,
                limit_files=data_cfg.get("limit_files"),
                use_tqdm=use_tqdm,
                tqdm_desc=f"read extra {Path(extra_dir).name}",
            )
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


def train(cfg: dict):
    set_seed(int(cfg.get("seed", 42)))

    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    use_tqdm = bool(train_cfg.get("use_tqdm", False))
    tqdm_mod = _get_tqdm(use_tqdm)

    _pwrite(tqdm_mod, json.dumps({"stage": "load_data"}, ensure_ascii=False))
    df = _load_data(cfg)
    _pwrite(tqdm_mod, json.dumps({"stage": "loaded", "rows": int(len(df))}, ensure_ascii=False))

    _pwrite(tqdm_mod, json.dumps({"stage": "feature_engineering"}, ensure_ascii=False))
    df = add_basic_tech_features(df)
    df, label_col = add_future_return_label(df, horizon=int(cfg.get("task", {}).get("horizon", 1)))
    feature_cols, label_col = select_feature_and_label_cols(df, label_col, extra_feature_cols=cfg.get("data", {}).get("extra_feature_cols"))
    df = df[["ts_code", "trade_date"] + feature_cols + [label_col]].copy()

    split_cfg = cfg.get("split", {})
    tr_df, va_df = split_by_date(df, split_cfg["train_end"], split_cfg["val_end"], date_col="trade_date")

    _pwrite(
        tqdm_mod,
        json.dumps({"stage": "split", "train_rows": int(len(tr_df)), "val_rows": int(len(va_df))}, ensure_ascii=False),
    )

    model_cfg = cfg.get("model", {})
    seq_len = int(model_cfg.get("seq_len", 30))
    normalize = str(model_cfg.get("normalize", "window"))

    tr_ds = PanelSeqDataset(tr_df, feature_cols, label_col, seq_len=seq_len, normalize=normalize)
    va_ds = PanelSeqDataset(va_df, feature_cols, label_col, seq_len=seq_len, normalize=normalize)

    if len(tr_ds) == 0:
        raise ValueError(
            "训练集样本数为 0：请扩大 data.start_date~data.end_date 范围、减小 model.seq_len、"
            "或调整 split.train_end / pool 过滤条件。"
        )
    if len(va_ds) == 0:
        raise ValueError(
            "验证集样本数为 0：请扩大 data.start_date~data.end_date 范围、减小 model.seq_len、"
            "或调整 split.val_end / pool 过滤条件。"
        )

    bs = int(train_cfg.get("batch_size", 256))
    num_workers = int(train_cfg.get("num_workers", 0))
    shuffle = bool(train_cfg.get("shuffle", True))
    log_every = int(train_cfg.get("log_every", 0))
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)

    in_dim = seq_len * len(feature_cols)
    model = build_model(cfg, in_dim=in_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    _pwrite(tqdm_mod, json.dumps({"stage": "device", "device": str(device)}, ensure_ascii=False))

    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    epochs = int(train_cfg.get("epochs", 10))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        steps = len(tr_dl)
        tr_loss_sum = 0.0
        tr_n = 0
        it = tr_dl
        if tqdm_mod is not None:
            it = tqdm_mod(tr_dl, desc=f"train {epoch}/{epochs}", total=steps)
        for step_idx, (xb, yb, _, _) in enumerate(it, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_loss_sum += float(loss.item()) * int(yb.shape[0])
            tr_n += int(yb.shape[0])
            if log_every > 0 and (step_idx % log_every == 0 or step_idx == steps):
                elapsed = time.perf_counter() - t0
                it_s = elapsed / max(1, step_idx)
                eta_s = it_s * max(0, steps - step_idx)
                tr_loss_so_far = tr_loss_sum / max(1, tr_n)
                _pwrite(
                    tqdm_mod,
                    json.dumps(
                        {"epoch": epoch, "epochs": epochs, "step": step_idx, "steps": steps, "train_loss": tr_loss_so_far, "eta_sec": eta_s},
                        ensure_ascii=False,
                    ),
                )
        tr_loss = tr_loss_sum / max(1, tr_n)

        model.eval()
        va_loss_sum = 0.0
        va_n = 0
        with torch.no_grad():
            it2 = va_dl
            if tqdm_mod is not None:
                it2 = tqdm_mod(va_dl, desc=f"val {epoch}/{epochs}", total=len(va_dl))
            for xb, yb, _, _ in it2:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_loss_sum += float(loss.item()) * int(yb.shape[0])
                va_n += int(yb.shape[0])
        va_loss = va_loss_sum / max(1, va_n)

        print(json.dumps({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss}, ensure_ascii=False))

    save_path = Path(train_cfg.get("save_path", "ckpt.pt"))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "feature_cols": feature_cols,
        "label_col": label_col,
        "seq_len": seq_len,
        "normalize": normalize,
        "cfg": cfg,
    }
    torch.save(ckpt, save_path)
    print(json.dumps({"saved": str(save_path)}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()

