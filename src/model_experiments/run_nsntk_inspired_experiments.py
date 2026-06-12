from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation import BacktestConfig, evaluate_prediction_scores
from src.model_experiments import run_gbdt
from src.model_experiments.run_e0_e1 import evaluate_split
from src.train import train as train_torch_model
from src.utils import (
    DEFAULT_ARTIFACT_REGISTRY,
    artifact_path,
    load_registry,
    make_run_dir,
    read_yaml,
    write_json,
    write_run_metadata,
)


LABEL_COL = "label_1d__cs_rank"
RETURN_COL = "label_1d"


def _progress(items, desc: str):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(list(items), desc=desc)
    except Exception:
        return items


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


def _rank_ic_by_day(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trade_date, g in df.groupby("trade_date", sort=True):
        g = g.dropna(subset=["pred", LABEL_COL])
        if len(g) < 3:
            continue
        pr = g["pred"].rank(method="average").to_numpy(dtype=np.float64)
        yr = g[LABEL_COL].rank(method="average").to_numpy(dtype=np.float64)
        if np.allclose(pr, pr[0]) or np.allclose(yr, yr[0]):
            continue
        ic = float(np.corrcoef(pr, yr)[0, 1])
        if np.isfinite(ic):
            rows.append({"trade_date": str(trade_date), "rank_ic": ic, "samples": int(len(g))})
    return pd.DataFrame(rows)


def _period_ic(ic_daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    if ic_daily.empty:
        return pd.DataFrame(columns=["period", "ic_mean", "ic_std", "icir", "ic_days", "samples"])
    out = ic_daily.copy()
    out["period"] = pd.to_datetime(out["trade_date"]).dt.to_period(freq).astype(str)
    rows = []
    for period, g in out.groupby("period", sort=True):
        mean = float(g["rank_ic"].mean())
        std = float(g["rank_ic"].std(ddof=0))
        rows.append(
            {
                "period": period,
                "ic_mean": mean,
                "ic_std": std,
                "icir": float(mean / (std + 1e-12)),
                "ic_days": int(len(g)),
                "samples": int(g["samples"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _plot_stability(monthly: pd.DataFrame, yearly: pd.DataFrame, out_dir: Path, name: str) -> dict[str, str]:
    plots: dict[str, str] = {}
    try:
        import matplotlib.pyplot as plt

        out_dir.mkdir(parents=True, exist_ok=True)
        if not monthly.empty:
            fig, ax = plt.subplots(figsize=(11, 4.2))
            ax.plot(monthly["period"], monthly["ic_mean"], color="#4b6475", marker="o", linewidth=1.8)
            ax.axhline(0.0, color="#8a8f93", linewidth=1, alpha=0.6)
            ax.set_title(f"{name} Monthly Rank IC")
            ax.set_xlabel("Month")
            ax.set_ylabel("Rank IC")
            ax.tick_params(axis="x", rotation=60, labelsize=8)
            ax.grid(True, alpha=0.22)
            fig.tight_layout()
            path = out_dir / f"{name}_monthly_ic.svg"
            fig.savefig(path)
            plt.close(fig)
            plots["monthly_ic"] = str(path)
        if not yearly.empty:
            fig, ax = plt.subplots(figsize=(7.2, 4.0))
            ax.bar(yearly["period"], yearly["icir"], color="#708c7f")
            ax.axhline(0.0, color="#8a8f93", linewidth=1, alpha=0.6)
            ax.set_title(f"{name} Yearly ICIR")
            ax.set_xlabel("Year")
            ax.set_ylabel("ICIR")
            ax.grid(True, axis="y", alpha=0.22)
            fig.tight_layout()
            path = out_dir / f"{name}_yearly_icir.svg"
            fig.savefig(path)
            plt.close(fig)
            plots["yearly_icir"] = str(path)
    except Exception as exc:
        plots["plot_error"] = str(exc)
    return plots


def _save_prediction_outputs(df: pd.DataFrame, out_dir: Path, split: str, name: str) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{split}_pred.parquet"
    df.to_parquet(pred_path, index=False)
    metrics = _topk10_drop2_metrics(df)
    metrics["split"] = split
    metrics["name"] = name
    write_json(out_dir / f"{split}_metrics.json", metrics)

    ic_daily = _rank_ic_by_day(df)
    monthly = _period_ic(ic_daily, "M")
    yearly = _period_ic(ic_daily, "Y")
    ic_daily.to_csv(out_dir / f"{split}_daily_ic.csv", index=False)
    monthly.to_csv(out_dir / f"{split}_monthly_ic.csv", index=False)
    yearly.to_csv(out_dir / f"{split}_yearly_ic.csv", index=False)
    plots = _plot_stability(monthly, yearly, out_dir / "plots", f"{name}_{split}")
    return {"pred_path": str(pred_path), "metrics": metrics, "plots": plots}


def _ema_tag(decay: float) -> str:
    return str(float(decay)).rstrip("0").rstrip(".").replace(".", "_")


def _deep_cfg(base_cfg: dict, model_name: str, seed: int, out_dir: Path, ema_decays: list[float]) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("train", {})
    cfg.setdefault("predict", {})
    cfg["train"]["seed"] = int(seed)
    cfg["train"]["save_path"] = str(out_dir / "best.pt")
    cfg["train"]["ema_decays"] = [float(x) for x in ema_decays]
    cfg["train"]["use_tqdm"] = True
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
    cfg["experiment_name"] = f"{model_name}_seed{seed}_ema_grid"
    return cfg


def _eval_checkpoint(cfg: dict, ckpt_path: Path, out_dir: Path, device: str | None) -> dict[str, dict]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("predict", {})
    cfg["predict"]["ckpt"] = str(ckpt_path)
    out = {}
    for split in ["valid", "test"]:
        _, metrics = evaluate_split(cfg, split, out_dir / split, raw_return_col=RETURN_COL, device_text=device)
        out[split] = metrics
    return out


def _merge_rank_ensemble(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("ensemble paths cannot be empty")
    merged = pd.read_parquet(paths[0]).copy()
    merged = merged.rename(columns={"pred": "pred_0"})
    base_cols = ["trade_date", "ts_code", LABEL_COL, RETURN_COL]
    keep = [c for c in base_cols if c in merged.columns] + ["pred_0"]
    merged = merged[keep]
    pred_cols = ["pred_0"]
    for i, path in enumerate(paths[1:], start=1):
        part = pd.read_parquet(path)[["trade_date", "ts_code", "pred"]].rename(columns={"pred": f"pred_{i}"})
        merged = merged.merge(part, on=["trade_date", "ts_code"], how="inner")
        pred_cols.append(f"pred_{i}")
    ranks = [merged.groupby("trade_date")[col].rank(method="average", pct=True) for col in pred_cols]
    out = merged[[c for c in base_cols if c in merged.columns]].copy()
    out["pred"] = np.mean(np.vstack([r.to_numpy(dtype=np.float32) for r in ranks]), axis=0).astype(np.float32)
    return out.dropna(subset=[LABEL_COL]).reset_index(drop=True)


def _smooth_scores(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    out = df.sort_values(["ts_code", "trade_date"], kind="mergesort").copy()
    out["pred"] = (
        out.groupby("ts_code", sort=False)["pred"]
        .transform(lambda s: s.ewm(alpha=float(alpha), adjust=False).mean())
        .astype(np.float32)
    )
    return out.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)


def _gbdt_namespace(args: argparse.Namespace, model: str, variant: str, half_life: float, out_root: Path, feature_list: str) -> SimpleNamespace:
    return SimpleNamespace(
        processed_dir=args.processed_dir,
        out_root=str(out_root / variant),
        model=model,
        target=LABEL_COL,
        raw_return_col=RETURN_COL,
        daily_return_col=RETURN_COL,
        feature_list=feature_list,
        filter_in_universe=True,
        max_train_rows=0,
        sample_weight_mode="uniform" if half_life <= 0 else "exp_decay",
        half_life_days=float(half_life),
        seed=args.seed,
        num_threads=args.num_threads,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        log_period=args.log_period,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_data_in_leaf=1000,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l1=0.0,
        lambda_l2=1.0,
        max_bin=255,
        xgb_max_depth=6,
        xgb_min_child_weight=100.0,
        n_hold=10,
        k_rotate=2,
        step_days=1,
        tranche_size=2,
        hold_days=5,
        transaction_cost_bps=5.0,
    )


def run_time_decay(args: argparse.Namespace, out_root: Path) -> list[dict[str, object]]:
    feature_list = args.feature_list or artifact_path(
        load_registry(args.artifact_registry),
        args.feature_list_artifact,
        source=args.artifact_registry,
    )
    variants = [("uniform", 0.0), ("half_life_1y", 252.0), ("half_life_2y", 504.0), ("half_life_3y", 756.0)]
    rows = []
    for variant, half_life in _progress(variants, "time_decay_variants"):
        for model in _progress(args.tree_models, f"time_decay_{variant}"):
            print(json.dumps({"stage": "tree_decay", "variant": variant, "model": model}, ensure_ascii=False))
            ns = _gbdt_namespace(args, model, variant, half_life, out_root / "time_decay", feature_list)
            summary = run_gbdt.run(ns)
            for split in ["valid", "test"]:
                m = summary.get(split, {})
                if isinstance(m, dict):
                    rows.append(
                        {
                            "section": "time_decay",
                            "variant": variant,
                            "model": model,
                            "split": split,
                            "half_life_days": half_life,
                            **m,
                        }
                    )
    pd.DataFrame(rows).to_csv(out_root / "time_decay_summary.csv", index=False)
    return rows


def run_deep_ema_and_seeds(args: argparse.Namespace, out_root: Path) -> tuple[list[dict[str, object]], dict[str, dict[str, list[Path]]]]:
    configs = {
        "mlp": "configs/report_label1d_mlp.yaml",
        "gru": "configs/report_label1d_gru_rerun4.yaml",
        "tcn": "configs/report_label1d_tcn_rerun4.yaml",
    }
    seeds_by_model = {"mlp": args.deep_seeds, "gru": args.deep_seeds, "tcn": [args.seed]}
    ema_decays = [float(x) for x in args.ema_decays]
    pred_paths: dict[str, dict[str, list[Path]]] = {}
    rows: list[dict[str, object]] = []
    for model_name, config_path in _progress(configs.items(), "deep_models"):
        base_cfg = read_yaml(config_path)
        variants = ["raw"] + [f"ema_{_ema_tag(decay)}" for decay in ema_decays]
        pred_paths[model_name] = {f"{variant}_{split}": [] for variant in variants for split in ["valid", "test"]}
        for seed in _progress(seeds_by_model[model_name], f"{model_name}_seeds"):
            model_dir = out_root / "deep_ema_seed" / model_name / f"seed_{seed}"
            cfg = _deep_cfg(base_cfg, model_name, seed, model_dir, ema_decays)
            write_json(model_dir / "config.json", cfg)
            print(json.dumps({"stage": "deep_train", "model": model_name, "seed": seed, "ema_decays": ema_decays}, ensure_ascii=False))
            train_torch_model(cfg)

            ckpts = [("raw", "best.pt")] + [(f"ema_{_ema_tag(decay)}", f"best_ema_{_ema_tag(decay)}.pt") for decay in ema_decays]
            for variant, ckpt_name in ckpts:
                ckpt_path = model_dir / ckpt_name
                if not ckpt_path.exists():
                    continue
                eval_dir = model_dir / variant
                metrics_by_split = _eval_checkpoint(cfg, ckpt_path, eval_dir, args.device)
                for split, metrics in metrics_by_split.items():
                    pred_paths[model_name][f"{variant}_{split}"].append(eval_dir / split / f"{split}_pred.parquet")
                    rows.append(
                        {
                            "section": "deep_ema_seed",
                            "model": model_name,
                            "seed": seed,
                            "variant": variant,
                            "ema_decay": float(variant.replace("ema_", "").replace("_", ".")) if variant.startswith("ema_") else None,
                            "split": split,
                            **metrics,
                        }
                    )
    pd.DataFrame(rows).to_csv(out_root / "deep_ema_seed_summary.csv", index=False)
    return rows, pred_paths


def run_seed_ensembles(out_root: Path, pred_paths: dict[str, dict[str, list[Path]]]) -> list[dict[str, object]]:
    rows = []
    for model_name in _progress(["mlp", "gru"], "seed_ensemble_models"):
        variants = sorted({key.rsplit("_", 1)[0] for key in pred_paths.get(model_name, {})})
        for variant in _progress(variants, f"{model_name}_ensemble_variants"):
            for split in ["valid", "test"]:
                paths = pred_paths.get(model_name, {}).get(f"{variant}_{split}", [])
                if len(paths) < 2:
                    continue
                df = _merge_rank_ensemble(paths)
                name = f"{model_name}_{variant}_{len(paths)}seed_rank_mean"
                result = _save_prediction_outputs(df, out_root / "seed_ensemble" / name / split, split, name)
                rows.append(
                    {
                        "section": "seed_ensemble",
                        "model": model_name,
                        "variant": variant,
                        "seed_count": len(paths),
                        "split": split,
                        **result["metrics"],
                    }
                )
    pd.DataFrame(rows).to_csv(out_root / "seed_ensemble_summary.csv", index=False)
    return rows


def run_score_smoothing(args: argparse.Namespace, out_root: Path) -> list[dict[str, object]]:
    rows = []
    for split in ["valid", "test"]:
        source = Path(args.smoothing_valid_pred if split == "valid" else args.smoothing_test_pred)
        base = pd.read_parquet(source)
        for alpha in _progress(args.smoothing_alpha, f"score_smoothing_{split}"):
            df = base.copy() if float(alpha) >= 0.999999 else _smooth_scores(base, float(alpha))
            name = f"gbdt_rank_mean_score_smooth_alpha_{str(alpha).replace('.', '_')}"
            result = _save_prediction_outputs(df, out_root / "score_smoothing" / name / split, split, name)
            rows.append(
                {
                    "section": "score_smoothing",
                    "alpha": float(alpha),
                    "split": split,
                    "source": str(source),
                    **result["metrics"],
                }
            )
    pd.DataFrame(rows).to_csv(out_root / "score_smoothing_summary.csv", index=False)
    return rows


def run_main_stability(args: argparse.Namespace, out_root: Path) -> list[dict[str, object]]:
    rows = []
    for split in _progress(["valid", "test"], "main_stability"):
        source = Path(args.smoothing_valid_pred if split == "valid" else args.smoothing_test_pred)
        df = pd.read_parquet(source)
        result = _save_prediction_outputs(df, out_root / "main_model_stability" / split, split, "main_gbdt_rank_mean")
        rows.append({"section": "main_model_stability", "split": split, "source": str(source), **result["metrics"]})
    pd.DataFrame(rows).to_csv(out_root / "main_model_stability_summary.csv", index=False)
    return rows


def _write_master_summary(out_root: Path, parts: Iterable[dict[str, object]]) -> None:
    rows = list(parts)
    pd.DataFrame(rows).to_csv(out_root / "all_experiment_metrics.csv", index=False)
    compact_cols = [
        "section",
        "model",
        "variant",
        "ema_decay",
        "seed",
        "seed_count",
        "alpha",
        "split",
        "ic_mean",
        "icir",
        "bt_total_return",
        "bt_sharpe",
        "bt_max_drawdown",
        "bt_avg_turnover",
        "rolling_bt_total_return",
        "rolling_bt_sharpe",
        "rolling_bt_max_drawdown",
        "rolling_bt_avg_turnover",
    ]
    compact = pd.DataFrame(rows)
    keep = [c for c in compact_cols if c in compact.columns]
    if keep:
        compact[keep].to_csv(out_root / "all_experiment_metrics_compact.csv", index=False)


def run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-registry", default=DEFAULT_ARTIFACT_REGISTRY)
    parser.add_argument("--feature-list-artifact", default="feature_list.lightgbm_top40")
    parser.add_argument("--feature-list", default=None)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-root", default="outputs/models")
    parser.add_argument("--run-name", default="nsntk_inspired_label1d")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--deep-seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    parser.add_argument("--ema-decay", type=float, default=None, help="Backward-compatible single EMA decay. Prefer --ema-decays.")
    parser.add_argument("--ema-decays", type=float, nargs="+", default=[0.99, 0.995, 0.999])
    parser.add_argument("--tree-models", nargs="+", choices=["lightgbm", "xgboost"], default=["lightgbm", "xgboost"])
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--num-boost-round", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--log-period", type=int, default=100)
    parser.add_argument("--smoothing-alpha", type=float, nargs="+", default=[1.0, 0.8, 0.6, 0.4])
    parser.add_argument("--smoothing-valid-pred", default="outputs/models/20260611_191641__report_label1d_fusion/fusion_rank_equal_gbdt/valid/valid_pred.parquet")
    parser.add_argument("--smoothing-test-pred", default="outputs/models/20260611_191641__report_label1d_fusion/fusion_rank_equal_gbdt/test/test_pred.parquet")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-tree", action="store_true")
    parser.add_argument("--skip-deep", action="store_true")
    parser.add_argument("--skip-smoothing", action="store_true")
    parser.add_argument("--skip-stability", action="store_true")
    args = parser.parse_args()
    if args.ema_decay is not None:
        args.ema_decays = [float(args.ema_decay)]

    out_root = make_run_dir(args.out_root, args.run_name, timestamped=not args.no_timestamp)
    out_root.mkdir(parents=True, exist_ok=True)
    write_run_metadata(out_root, command="nsntk-inspired", args=args, registry_paths=[args.artifact_registry])

    all_rows: list[dict[str, object]] = []
    pred_paths: dict[str, dict[str, list[Path]]] = {}
    if not args.skip_tree:
        all_rows.extend(run_time_decay(args, out_root))
    if not args.skip_deep:
        deep_rows, pred_paths = run_deep_ema_and_seeds(args, out_root)
        all_rows.extend(deep_rows)
        all_rows.extend(run_seed_ensembles(out_root, pred_paths))
    if not args.skip_smoothing:
        all_rows.extend(run_score_smoothing(args, out_root))
    if not args.skip_stability:
        all_rows.extend(run_main_stability(args, out_root))
    _write_master_summary(out_root, all_rows)
    write_json(
        out_root / "summary.json",
        {
            "out_root": str(out_root),
            "rows": len(all_rows),
            "sections": sorted(set(str(r.get("section")) for r in all_rows)),
        },
    )
    print(json.dumps({"saved": str(out_root), "rows": len(all_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    run_cli()
