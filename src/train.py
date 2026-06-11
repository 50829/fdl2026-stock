"""Training entry point."""

from __future__ import annotations

import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from src.data import (
    ProcessedConfig,
    build_processed_splits,
    iter_processed_batches,
    iter_processed_sequence_batches,
    iter_processed_sequence_labeled_feature_batches,
    load_feature_columns,
)
from src.models import build_model
from src.utils import read_yaml


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


def _write_train_history(out_dir: Path, history: list[dict], best_epoch: int, best_loss: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"history": history, "best_epoch": int(best_epoch), "best_valid_loss": float(best_loss)}
    (out_dir / "train_history.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_lines = ["epoch,epochs,train_loss,val_loss,sec"]
    for row in history:
        csv_lines.append(
            ",".join(
                [
                    str(row.get("epoch", "")),
                    str(row.get("epochs", "")),
                    str(row.get("train_loss", "")),
                    str(row.get("val_loss", "")),
                    str(row.get("sec", "")),
                ]
            )
        )
    (out_dir / "train_history.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        epochs = [int(row["epoch"]) for row in history]
        train_loss = [float(row["train_loss"]) for row in history]
        val_loss = [float(row["val_loss"]) for row in history]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(epochs, train_loss, color="#2563eb", marker="o", linewidth=2, label="train loss")
        ax.plot(epochs, val_loss, color="#f97316", marker="s", linewidth=2, label="valid loss")
        if epochs:
            ax.annotate(
                "train",
                xy=(epochs[-1], train_loss[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                color="#2563eb",
                va="center",
                fontsize=9,
            )
            ax.annotate(
                "valid",
                xy=(epochs[-1], val_loss[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                color="#f97316",
                va="center",
                fontsize=9,
            )
        if best_epoch > 0 and np.isfinite(best_loss):
            ax.axvline(best_epoch, color="#6b7280", linewidth=1, alpha=0.45)
            ax.annotate(
                f"best valid epoch {best_epoch}",
                xy=(best_epoch, best_loss),
                xytext=(8, 10),
                textcoords="offset points",
                color="#374151",
                fontsize=8,
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "train_history.svg")
        plt.close(fig)
    except Exception:
        pass


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


def _build_loss(name: str) -> nn.Module:
    name = str(name).lower()
    if name in {"mse", "l2"}:
        return nn.MSELoss()
    if name in {"smooth_l1", "huber"}:
        return nn.SmoothL1Loss()
    raise ValueError(f"unsupported train.loss for src.train: {name}")


def _resolve_warmup_start(pcfg: ProcessedConfig, start_date: str, seq_len: int) -> str:
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
    return ordered[max(0, idx - int(seq_len) + 1)]


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
    is_sequence_model = model_name in {"lstm", "transformer", "tf", "alstm", "tcn", "temporal_conv", "temporal_convolution"}

    in_dim = int(len(feature_cols))
    model = build_model(cfg, in_dim=in_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _pwrite(tqdm_mod, json.dumps({"stage": "device", "device": str(device)}, ensure_ascii=False))

    lr = float(train_cfg.get("learning_rate", train_cfg.get("lr", 1e-3)))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = _build_loss(str(train_cfg.get("loss", "mse")))

    epochs = int(train_cfg.get("epochs", 20))
    batch_size = int(train_cfg.get("batch_size", 256))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    filter_in_universe = bool(train_cfg.get("filter_in_universe", True))
    cache_data = bool(train_cfg.get("cache_data", False))
    patience_raw = train_cfg.get("patience")
    patience = int(patience_raw) if patience_raw is not None else None
    min_delta = float(train_cfg.get("min_delta", 0.0))
    save_path = Path(train_cfg.get("save_path", "outputs/models/ckpt.pt"))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    seq_len = int(model_cfg.get("seq_len", cfg.get("sample", {}).get("lookback", 60)))
    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        tr_loss_sum = 0.0
        tr_n = 0

        if is_sequence_model:
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
                use_tqdm=use_tqdm,
                stage_desc="train_seq",
                cache_in_memory=cache_data,
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
                cache_in_memory=cache_data,
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
            valid_warmup_start = _resolve_warmup_start(pcfg, splits["valid"].start_date, seq_len)
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
                use_tqdm=use_tqdm,
                stage_desc="valid_seq",
                cache_in_memory=cache_data,
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
                cache_in_memory=cache_data,
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
        row = {"epoch": epoch, "epochs": epochs, "train_loss": tr_loss, "val_loss": va_loss, "sec": elapsed}
        history.append(row)

        _pwrite(
            tqdm_mod,
            json.dumps(row, ensure_ascii=False),
        )

        if va_loss < best_loss - min_delta:
            best_loss = va_loss
            best_epoch = epoch
            bad_epochs = 0
            ckpt = {
                "model_state": model.state_dict(),
                "feature_cols": feature_cols,
                "label_col": label_col,
                "cfg": cfg,
                "best_epoch": best_epoch,
                "best_valid_loss": best_loss,
                "history": history,
            }
            torch.save(ckpt, save_path)
            _pwrite(tqdm_mod, json.dumps({"saved_best": str(save_path), "best_epoch": best_epoch, "best_valid_loss": best_loss}, ensure_ascii=False))
        else:
            bad_epochs += 1
        _write_train_history(save_path.parent, history, best_epoch, best_loss)
        torch.save(
            {
                "feature_cols": feature_cols,
                "label_col": label_col,
                "cfg": cfg,
                "best_epoch": best_epoch,
                "best_valid_loss": best_loss,
                "history": history,
            },
            save_path.parent / "last_history.pt",
        )
        if not (va_loss < best_loss - min_delta):
            if patience is not None and bad_epochs >= patience:
                _pwrite(tqdm_mod, json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}, ensure_ascii=False))
                break

    _pwrite(tqdm_mod, json.dumps({"saved": str(save_path), "best_epoch": best_epoch, "best_valid_loss": best_loss}, ensure_ascii=False))
    _write_train_history(save_path.parent, history, best_epoch, best_loss)


def run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(read_yaml(args.config))


if __name__ == "__main__":
    run_cli()
