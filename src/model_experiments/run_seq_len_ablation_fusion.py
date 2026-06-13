from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation import BacktestConfig, evaluate_prediction_scores
from src.model_experiments.run_e0_e1 import evaluate_split
from src.train import train as train_torch_model
from src.utils import make_run_dir, read_yaml, write_json, write_run_metadata


LABEL_COL = "label_1d__cs_rank"
RETURN_COL = "label_1d"
DEFAULT_TREE_VALID = "outputs/models/20260612_151735__nsntk_inspired_label1d/main_model_stability/valid/valid_pred.parquet"
DEFAULT_TREE_TEST = "outputs/models/20260612_151735__nsntk_inspired_label1d/main_model_stability/test/test_pred.parquet"
DEFAULT_PREVIOUS_SEQ60_ROOT = "outputs/models/20260612_172058__nsntk_ema_grid_label1d"


def _ema_tag(decay: float) -> str:
    return str(float(decay)).rstrip("0").rstrip(".").replace(".", "_")


def _topk10_drop2_metrics(df: pd.DataFrame) -> dict[str, object]:
    return evaluate_prediction_scores(
        df,
        label_col=LABEL_COL,
        raw_return_col=RETURN_COL,
        daily_return_col=RETURN_COL,
        topk_cfg=BacktestConfig(mode="topk", n_hold=10, k_rotate=2, step_days=1, transaction_cost_bps=5.0),
        rolling_cfg=BacktestConfig(
            mode="rolling_tranche",
            tranche_size=2,
            hold_days=5,
            daily_return_col=RETURN_COL,
            transaction_cost_bps=5.0,
        ),
    )


def _deep_cfg(base_cfg: dict[str, Any], *, model_name: str, seq_len: int, seed: int, decay: float, out_dir: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg.setdefault("predict", {})
    cfg["model"]["seq_len"] = int(seq_len)
    cfg["train"]["seed"] = int(seed)
    cfg["train"]["save_path"] = str(out_dir / "best.pt")
    cfg["train"]["ema_decays"] = [float(decay)]
    cfg["train"]["use_tqdm"] = True
    cfg["train"]["progress_log"] = True
    cfg["predict"]["ckpt"] = str(out_dir / "best.pt")
    cfg["predict"]["use_tqdm"] = True
    cfg["backtest"] = {
        "mode": "topk",
        "return_col": RETURN_COL,
        "n_hold": 10,
        "k_rotate": 2,
        "step_days": 1,
        "transaction_cost_bps": 5.0,
    }
    cfg["experiment_name"] = f"{model_name}_seq{seq_len}_seed{seed}_ema{_ema_tag(decay)}"
    return cfg


def _evaluate_checkpoint(cfg: dict[str, Any], ckpt_path: Path, out_dir: Path, device: str | None) -> dict[str, dict[str, Any]]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("predict", {})
    cfg["predict"]["ckpt"] = str(ckpt_path)
    out: dict[str, dict[str, Any]] = {}
    for split in ["valid", "test"]:
        _, metrics = evaluate_split(cfg, split, out_dir / split, raw_return_col=RETURN_COL, device_text=device)
        out[split] = metrics
    return out


def _merge_rank_ensemble(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("ensemble paths cannot be empty")
    merged = pd.read_parquet(paths[0]).rename(columns={"pred": "pred_0"})
    keep = ["trade_date", "ts_code", LABEL_COL, RETURN_COL, "pred_0"]
    merged = merged[[c for c in keep if c in merged.columns]]
    pred_cols = ["pred_0"]
    for idx, path in enumerate(paths[1:], start=1):
        part = pd.read_parquet(path)[["trade_date", "ts_code", "pred"]].rename(columns={"pred": f"pred_{idx}"})
        merged = merged.merge(part, on=["trade_date", "ts_code"], how="inner")
        pred_cols.append(f"pred_{idx}")
    ranks = [merged.groupby("trade_date", sort=False)[col].rank(method="average", pct=True) for col in pred_cols]
    out = merged[[c for c in ["trade_date", "ts_code", LABEL_COL, RETURN_COL] if c in merged.columns]].copy()
    out["pred"] = np.mean(np.vstack([r.to_numpy(dtype=np.float32) for r in ranks]), axis=0).astype(np.float32)
    return out.dropna(subset=[LABEL_COL]).reset_index(drop=True)


def _save_pred_metrics(df: pd.DataFrame, out_dir: Path, split: str, name: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{split}_pred.parquet"
    df.to_parquet(pred_path, index=False)
    metrics = _topk10_drop2_metrics(df)
    metrics.update({"split": split, "name": name})
    write_json(out_dir / f"{split}_metrics.json", metrics)
    return {"pred_path": str(pred_path), "metrics": metrics}


def _coverage(tree_path: Path, deep_path: Path) -> dict[str, Any]:
    tree = pd.read_parquet(tree_path, columns=["trade_date", "ts_code"])
    deep = pd.read_parquet(deep_path, columns=["trade_date", "ts_code"])
    merged = tree.merge(deep.assign(has_deep=True), on=["trade_date", "ts_code"], how="left")
    has_deep = merged["has_deep"].fillna(False).astype(bool)
    return {
        "tree_rows": int(len(merged)),
        "tree_dates": int(merged["trade_date"].nunique()),
        "deep_rows": int(has_deep.sum()),
        "deep_coverage": float(has_deep.mean()) if len(merged) else 0.0,
        "fallback_rows": int((~has_deep).sum()),
    }


def _load_tree_deep_frame(tree_path: Path, deep_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    tree = pd.read_parquet(tree_path)
    deep = pd.read_parquet(deep_path)
    required = {"trade_date", "ts_code", "pred", LABEL_COL, RETURN_COL}
    missing = sorted(required - set(tree.columns))
    if missing:
        raise ValueError(f"tree prediction file missing required columns: {missing}")
    merged = tree[["trade_date", "ts_code", LABEL_COL, RETURN_COL, "pred"]].rename(columns={"pred": "pred_tree"}).merge(
        deep[["trade_date", "ts_code", "pred"]].rename(columns={"pred": "pred_deep"}),
        on=["trade_date", "ts_code"],
        how="left",
    )
    merged = merged.dropna(subset=[LABEL_COL, RETURN_COL, "pred_tree"]).reset_index(drop=True)
    has_deep = merged["pred_deep"].notna()
    merged["rank_tree"] = merged.groupby("trade_date", sort=False)["pred_tree"].rank(method="average", pct=True).astype(np.float32)
    merged["rank_deep"] = merged.groupby("trade_date", sort=False)["pred_deep"].rank(method="average", pct=True).astype(np.float32)
    merged["rank_deep"] = merged["rank_deep"].fillna(merged["rank_tree"]).astype(np.float32)
    coverage = {
        "tree_rows": int(len(merged)),
        "tree_dates": int(merged["trade_date"].nunique()),
        "deep_rows": int(has_deep.sum()),
        "deep_coverage": float(has_deep.mean()) if len(merged) else 0.0,
        "fallback_rows": int((~has_deep).sum()),
    }
    return merged, coverage


def _eval_tree_deep_alpha(df: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    pred = df[["trade_date", "ts_code", LABEL_COL, RETURN_COL]].copy()
    pred["pred"] = ((1.0 - alpha) * df["rank_tree"].to_numpy(dtype=np.float32) + alpha * df["rank_deep"].to_numpy(dtype=np.float32)).astype(np.float32)
    return pred, _topk10_drop2_metrics(pred)


def _run_fusion_grid(
    candidates: list[dict[str, Any]],
    out_root: Path,
    *,
    tree_valid: Path,
    tree_test: Path,
    alpha_grid: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    tree_paths = {"valid": tree_valid, "test": tree_test}
    for candidate in candidates:
        name = str(candidate["candidate"])
        exp_dir = out_root / "fusion" / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        split_grids: dict[str, pd.DataFrame] = {}
        split_frames: dict[str, pd.DataFrame] = {}
        for split in ["valid", "test"]:
            frame, coverage = _load_tree_deep_frame(tree_paths[split], Path(candidate[f"{split}_pred_path"]))
            split_frames[split] = frame
            rows = []
            for alpha in alpha_grid:
                pred, metrics = _eval_tree_deep_alpha(frame, float(alpha))
                row = {
                    "candidate": name,
                    "model": candidate["model"],
                    "seq_len": candidate["seq_len"],
                    "split": split,
                    "alpha_deep": float(alpha),
                    "alpha_tree": float(1.0 - alpha),
                    **coverage,
                    **metrics,
                }
                rows.append(row)
                all_rows.append(row)
            grid = pd.DataFrame(rows).sort_values(["bt_sharpe", "icir", "ic_mean"], ascending=False, kind="mergesort").reset_index(drop=True)
            grid.to_csv(exp_dir / f"{split}_alpha_grid.csv", index=False)
            split_grids[split] = grid
        valid_best = split_grids["valid"].iloc[0].to_dict()
        best_alpha = float(valid_best["alpha_deep"])
        test_match = split_grids["test"][split_grids["test"]["alpha_deep"].eq(best_alpha)].iloc[0].to_dict()
        selected_rows.extend(
            [
                {"selection": "valid_best_by_topk10_drop2_sharpe", **valid_best},
                {"selection": "valid_selected_test_once", **test_match},
            ]
        )
        for split in ["valid", "test"]:
            pred, metrics = _eval_tree_deep_alpha(split_frames[split], best_alpha)
            split_dir = exp_dir / f"{split}_selected_alpha_{str(best_alpha).replace('.', '_')}"
            _save_pred_metrics(pred, split_dir, split, f"{name}_alpha_{best_alpha:g}")
            write_json(split_dir / f"{split}_metrics.json", metrics)
        write_json(exp_dir / "summary.json", {"valid_best_by_sharpe": valid_best, "test_at_valid_best_alpha": test_match})
    return all_rows, selected_rows


def _previous_seq60_candidate(model: str, previous_root: Path) -> dict[str, Any]:
    if model == "gru":
        base = previous_root / "seed_ensemble/gru_ema_0_995_3seed_rank_mean"
        candidate = "gru_seq60_ema0995_3seed"
    elif model == "tcn":
        base = previous_root / "deep_ema_seed/tcn/seed_2026/ema_0_995"
        candidate = "tcn_seq60_ema0995"
    else:
        raise ValueError(model)
    return {
        "candidate": candidate,
        "model": model,
        "seq_len": 60,
        "variant": "reused_previous_seq60",
        "valid_pred_path": str(base / "valid/valid_pred.parquet"),
        "test_pred_path": str(base / "test/test_pred.parquet"),
    }


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Run GRU/TCN seq_len ablation and tree fusion backtests.")
    parser.add_argument("--out-root", default="outputs/models")
    parser.add_argument("--run-name", default="seq_len_ablation_fusion_label1d")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[20, 30, 60])
    parser.add_argument("--train-seq-lens", nargs="+", type=int, default=[20, 30])
    parser.add_argument("--models", nargs="+", choices=["gru", "tcn"], default=["gru", "tcn"])
    parser.add_argument("--gru-config", default="configs/report_label1d_gru_rerun4.yaml")
    parser.add_argument("--tcn-config", default="configs/report_label1d_tcn_rerun4.yaml")
    parser.add_argument("--gru-seeds", nargs="+", type=int, default=[2024, 2025, 2026])
    parser.add_argument("--tcn-seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--tree-valid-pred", default=DEFAULT_TREE_VALID)
    parser.add_argument("--tree-test-pred", default=DEFAULT_TREE_TEST)
    parser.add_argument("--previous-seq60-root", default=DEFAULT_PREVIOUS_SEQ60_ROOT)
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30])
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    out_root = make_run_dir(args.out_root, args.run_name, timestamped=not args.no_timestamp)
    out_root.mkdir(parents=True, exist_ok=True)
    write_run_metadata(out_root, command="seq-len-fusion", args=args)

    configs = {"gru": read_yaml(args.gru_config), "tcn": read_yaml(args.tcn_config)}
    seeds_by_model = {"gru": args.gru_seeds, "tcn": args.tcn_seeds}
    train_seq_lens = sorted(set(int(x) for x in args.train_seq_lens if int(x) in set(args.seq_lens)))
    candidates: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    ensemble_rows: list[dict[str, Any]] = []

    for model in args.models:
        base_cfg = configs[model]
        for seq_len in train_seq_lens:
            seed_pred_paths: dict[str, list[Path]] = {"valid": [], "test": []}
            for seed in seeds_by_model[model]:
                model_dir = out_root / "deep_seq_len" / model / f"seq{seq_len}" / f"seed_{seed}"
                cfg = _deep_cfg(base_cfg, model_name=model, seq_len=seq_len, seed=seed, decay=args.ema_decay, out_dir=model_dir)
                write_json(model_dir / "config.json", cfg)
                if not args.skip_train:
                    print(json.dumps({"stage": "train", "model": model, "seq_len": seq_len, "seed": seed, "ema_decay": args.ema_decay}, ensure_ascii=False), flush=True)
                    train_torch_model(cfg)
                ckpt = model_dir / f"best_ema_{_ema_tag(args.ema_decay)}.pt"
                if not ckpt.exists():
                    raise FileNotFoundError(f"missing EMA checkpoint: {ckpt}")
                eval_dir = model_dir / f"ema_{_ema_tag(args.ema_decay)}"
                metrics_by_split = _evaluate_checkpoint(cfg, ckpt, eval_dir, args.device)
                for split, metrics in metrics_by_split.items():
                    pred_path = eval_dir / split / f"{split}_pred.parquet"
                    seed_pred_paths[split].append(pred_path)
                    cov = _coverage(Path(args.tree_valid_pred if split == "valid" else args.tree_test_pred), pred_path)
                    seed_rows.append(
                        {
                            "section": "seed",
                            "candidate": f"{model}_seq{seq_len}_seed{seed}_ema{_ema_tag(args.ema_decay)}",
                            "model": model,
                            "seq_len": int(seq_len),
                            "seed": int(seed),
                            "ema_decay": float(args.ema_decay),
                            "split": split,
                            "pred_path": str(pred_path),
                            **cov,
                            **metrics,
                        }
                    )
            if len(seed_pred_paths["valid"]) > 1:
                candidate_name = f"{model}_seq{seq_len}_ema{_ema_tag(args.ema_decay)}_{len(seed_pred_paths['valid'])}seed"
                candidate: dict[str, Any] = {
                    "candidate": candidate_name,
                    "model": model,
                    "seq_len": int(seq_len),
                    "variant": "seed_rank_mean",
                    "seed_count": len(seed_pred_paths["valid"]),
                }
                for split in ["valid", "test"]:
                    df = _merge_rank_ensemble(seed_pred_paths[split])
                    result = _save_pred_metrics(df, out_root / "deep_seq_len_ensemble" / candidate_name / split, split, candidate_name)
                    pred_path = Path(result["pred_path"])
                    candidate[f"{split}_pred_path"] = str(pred_path)
                    cov = _coverage(Path(args.tree_valid_pred if split == "valid" else args.tree_test_pred), pred_path)
                    ensemble_rows.append(
                        {
                            "section": "ensemble",
                            "candidate": candidate_name,
                            "model": model,
                            "seq_len": int(seq_len),
                            "seed_count": len(seed_pred_paths["valid"]),
                            "ema_decay": float(args.ema_decay),
                            "split": split,
                            **cov,
                            **result["metrics"],
                        }
                    )
                candidates.append(candidate)
            else:
                candidate_name = f"{model}_seq{seq_len}_ema{_ema_tag(args.ema_decay)}"
                candidate = {
                    "candidate": candidate_name,
                    "model": model,
                    "seq_len": int(seq_len),
                    "variant": "single_seed",
                    "seed_count": 1,
                    "valid_pred_path": str(seed_pred_paths["valid"][0]),
                    "test_pred_path": str(seed_pred_paths["test"][0]),
                }
                candidates.append(candidate)

    previous_root = Path(args.previous_seq60_root)
    for model in args.models:
        if 60 in set(args.seq_lens):
            candidate = _previous_seq60_candidate(model, previous_root)
            for split in ["valid", "test"]:
                pred_path = Path(candidate[f"{split}_pred_path"])
                if not pred_path.exists():
                    raise FileNotFoundError(f"missing reused seq60 prediction: {pred_path}")
                df = pd.read_parquet(pred_path)
                metrics = _topk10_drop2_metrics(df)
                cov = _coverage(Path(args.tree_valid_pred if split == "valid" else args.tree_test_pred), pred_path)
                ensemble_rows.append(
                    {
                        "section": "reused_previous_seq60",
                        "candidate": candidate["candidate"],
                        "model": model,
                        "seq_len": 60,
                        "ema_decay": float(args.ema_decay),
                        "split": split,
                        "pred_path": str(pred_path),
                        **cov,
                        **metrics,
                    }
                )
            candidates.append(candidate)

    pd.DataFrame(seed_rows).to_csv(out_root / "seq_len_seed_summary.csv", index=False)
    pd.DataFrame(ensemble_rows).to_csv(out_root / "seq_len_candidate_summary.csv", index=False)
    pd.DataFrame(candidates).to_csv(out_root / "seq_len_candidates.csv", index=False)

    fusion_rows, selected_rows = _run_fusion_grid(
        candidates,
        out_root,
        tree_valid=Path(args.tree_valid_pred),
        tree_test=Path(args.tree_test_pred),
        alpha_grid=[float(x) for x in args.alpha_grid],
    )
    pd.DataFrame(fusion_rows).to_csv(out_root / "fusion_alpha_grid.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(out_root / "fusion_selected_by_valid_sharpe.csv", index=False)

    summary = {
        "out_root": str(out_root),
        "seq_lens": [int(x) for x in args.seq_lens],
        "trained_seq_lens": train_seq_lens,
        "ema_decay": float(args.ema_decay),
        "selection_rule": "select alpha_deep by valid topk10_drop2 bt_sharpe; evaluate test once",
        "candidate_summary": str(out_root / "seq_len_candidate_summary.csv"),
        "fusion_grid": str(out_root / "fusion_alpha_grid.csv"),
        "fusion_selected": str(out_root / "fusion_selected_by_valid_sharpe.csv"),
    }
    write_json(out_root / "summary.json", summary)
    print(json.dumps({"saved": str(out_root), "candidates": len(candidates)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run_cli()
