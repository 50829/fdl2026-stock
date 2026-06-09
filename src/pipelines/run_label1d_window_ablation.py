from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.model_experiments import run_gbdt
from src.utils import make_run_dir, write_json, write_run_metadata


WINDOWS = (5, 10, 20, 60)
DEFAULT_VARIANTS = ("all_windows", "short_5_10", "mid_20_60", "no_5d", "no_10d", "no_20d", "no_60d")


def read_feature_list(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def feature_windows(feature: str) -> set[int]:
    """Return explicit rolling-window dependencies encoded in a feature name.

    This intentionally keys off the generated feature names, not model
    importance. For example, `momentum_5__ts_z60` depends on both a 5-day
    source signal and a 60-day time-series normalization.
    """
    found: set[int] = set()
    for window in WINDOWS:
        patterns = [
            rf"_{window}(?:__|_|$)",
            rf"z{window}(?:$|_)",
        ]
        if any(re.search(pattern, feature) for pattern in patterns):
            found.add(window)
    return found


def select_features(features: list[str], variant: str) -> list[str]:
    if variant == "all_windows":
        return list(features)
    if variant == "short_5_10":
        allowed = {5, 10}
        return [feature for feature in features if feature_windows(feature).issubset(allowed)]
    if variant == "mid_20_60":
        allowed = {20, 60}
        return [feature for feature in features if feature_windows(feature).issubset(allowed)]
    if variant.startswith("no_") and variant.endswith("d"):
        window = int(variant.removeprefix("no_").removesuffix("d"))
        return [feature for feature in features if window not in feature_windows(feature)]
    raise ValueError(f"Unknown window ablation variant: {variant}")


def write_feature_lists(root: Path, base_features: list[str], variants: list[str]) -> dict[str, Path]:
    feature_dir = root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    records = []
    for variant in variants:
        selected = select_features(base_features, variant)
        if not selected:
            raise ValueError(f"Variant {variant} produced an empty feature list")
        path = feature_dir / f"{variant}.txt"
        path.write_text("\n".join(selected) + "\n", encoding="utf-8")
        paths[variant] = path
        records.append(
            {
                "variant": variant,
                "feature_count": len(selected),
                "removed_count": len(base_features) - len(selected),
                "features": selected,
            }
        )
    write_json(feature_dir / "feature_lists_summary.json", {"variants": records})
    pd.DataFrame(
        [{k: v for k, v in record.items() if k != "features"} for record in records]
    ).to_csv(feature_dir / "feature_lists_summary.csv", index=False)
    return paths


def build_gbdt_args(args: argparse.Namespace, variant_root: Path, feature_list: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model=args.model,
        processed_dir=args.processed_dir,
        out_root=str(variant_root),
        target=args.target,
        raw_return_col=args.raw_return_col,
        daily_return_col=args.daily_return_col,
        feature_list=str(feature_list),
        filter_in_universe=args.filter_in_universe,
        max_train_rows=args.max_train_rows,
        seed=args.seed,
        num_threads=args.num_threads,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        log_period=args.log_period,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_data_in_leaf=args.min_data_in_leaf,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
        max_bin=args.max_bin,
        xgb_max_depth=args.xgb_max_depth,
        xgb_min_child_weight=args.xgb_min_child_weight,
        n_hold=args.n_hold,
        k_rotate=args.k_rotate,
        step_days=args.step_days,
        tranche_size=args.tranche_size,
        hold_days=args.hold_days,
        transaction_cost_bps=args.transaction_cost_bps,
    )


def flatten_summary(variant: str, summary: dict) -> dict:
    row = {
        "variant": variant,
        "model": summary.get("model"),
        "feature_count": summary.get("feature_count"),
        "train_rows": summary.get("train_rows"),
        "valid_rows": summary.get("valid_rows"),
        "train_sec": summary.get("train_sec"),
    }
    for split in ["valid", "test"]:
        metrics = summary.get(split, {})
        if isinstance(metrics, dict):
            for key in [
                "ic_mean",
                "ic_std",
                "icir",
                "mse",
                "bt_total_return",
                "bt_annual_return",
                "bt_sharpe",
                "bt_max_drawdown",
                "bt_avg_turnover",
                "rolling_bt_total_return",
                "rolling_bt_annual_return",
                "rolling_bt_sharpe",
                "rolling_bt_max_drawdown",
                "rolling_bt_avg_turnover",
                "rolling_bt_avg_active_positions",
            ]:
                row[f"{split}_{key}"] = metrics.get(key)
    return row


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    text_df = df.copy()
    for col in text_df.columns:
        if pd.api.types.is_float_dtype(text_df[col]):
            text_df[col] = text_df[col].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
        else:
            text_df[col] = text_df[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(text_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text_df.astype(str).to_numpy()]
    return "\n".join([header, sep, *rows])


def write_report(root: Path, metrics: pd.DataFrame, args: argparse.Namespace) -> None:
    sort_col = "valid_icir" if "valid_icir" in metrics else "test_icir"
    ordered = metrics.sort_values(sort_col, ascending=False, kind="mergesort")
    report = [
        "# label1d 窗口消融实验",
        "",
        f"输出目录：`{root}`",
        "",
        "## 实验口径",
        "",
        f"- 模型：`{args.model}`",
        f"- 目标：`{args.target}`",
        f"- 原始收益列：`{args.raw_return_col}`",
        f"- 日收益列：`{args.daily_return_col}`",
        f"- 股票池过滤：`filter_in_universe={args.filter_in_universe}`",
        f"- 基础特征列表：`{args.base_feature_list}`",
        "",
        "## 变体说明",
        "",
        "- `all_windows`：保留基础列表中所有显式窗口特征。",
        "- `short_5_10`：只保留无显式窗口依赖、5 日和 10 日窗口特征。",
        "- `mid_20_60`：只保留无显式窗口依赖、20 日和 60 日窗口特征。",
        "- `no_5d/no_10d/no_20d/no_60d`：删除对应显式窗口依赖的特征。",
        "",
        "注意：`momentum_5__ts_z60` 这类特征同时依赖 5 日源信号和 60 日时序标准化，因此在 `no_5d` 和 `no_60d` 中都会被删除。",
        "",
        "## 验证集排序",
        "",
        markdown_table(
            ordered[
            [
                "variant",
                "feature_count",
                "valid_ic_mean",
                "valid_icir",
                "valid_rolling_bt_sharpe",
                "valid_rolling_bt_max_drawdown",
                "test_ic_mean",
                "test_icir",
                "test_rolling_bt_sharpe",
                "test_rolling_bt_max_drawdown",
            ]
            ]
        ),
        "",
    ]
    (root / "window_ablation_report.md").write_text("\n".join(report), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    root = make_run_dir(args.out_root, args.run_name, timestamped=not args.no_timestamp)
    root.mkdir(parents=True, exist_ok=True)
    variants = list(args.variants or DEFAULT_VARIANTS)
    base_features = read_feature_list(args.base_feature_list)
    feature_paths = write_feature_lists(root, base_features, variants)
    write_run_metadata(
        root,
        command="label1d-window-ablation",
        args=args,
        inputs={"base_feature_list": args.base_feature_list, "variants": variants},
    )

    summaries = []
    for variant in variants:
        print(json.dumps({"stage": "variant_start", "variant": variant}, ensure_ascii=False), flush=True)
        gbdt_args = build_gbdt_args(args, root / variant, feature_paths[variant])
        summary = run_gbdt.run(gbdt_args)
        summaries.append({"variant": variant, "summary": summary})
        print(json.dumps({"stage": "variant_done", "variant": variant}, ensure_ascii=False), flush=True)

    write_json(root / "window_ablation_summary.json", {"experiments": summaries})
    metrics = pd.DataFrame([flatten_summary(item["variant"], item["summary"]) for item in summaries])
    metrics.to_csv(root / "window_ablation_metrics.csv", index=False)
    write_report(root, metrics, args)
    return root


def run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lightgbm", "xgboost"], default="lightgbm")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-root", default="outputs/models")
    parser.add_argument("--run-name", default="label1d_window_ablation")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--base-feature-list", default="outputs/models/20260530_205006__feature_selection/features/lightgbm_top40.txt")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--target", default="label_1d__cs_rank")
    parser.add_argument("--raw-return-col", default="label_1d")
    parser.add_argument("--daily-return-col", default="label_1d")
    parser.add_argument("--filter-in-universe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--num-boost-round", type=int, default=800)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
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
    parser.add_argument("--n-hold", type=int, default=20)
    parser.add_argument("--k-rotate", type=int, default=5)
    parser.add_argument("--step-days", type=int, default=5)
    parser.add_argument("--tranche-size", type=int, default=4)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    root = run(parser.parse_args())
    print(json.dumps({"out_dir": str(root)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run_cli()
