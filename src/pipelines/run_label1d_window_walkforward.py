from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import ProcessedConfig
from src.evaluation import BacktestConfig, backtest_topk
from src.model_experiments.run_gbdt_walkforward import build_folds, resolve_features, run_fold
from src.pipelines.run_label1d_window_ablation import markdown_table
from src.utils import make_run_dir, write_json, write_run_metadata


DEFAULT_VARIANTS = ("all_windows", "no_20d", "short_5_10")


def feature_list_path(feature_root: Path, variant: str) -> Path:
    path = feature_root / f"{variant}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature list for variant {variant}: {path}")
    return path


def jsonable_fold(fold: dict) -> dict:
    train = fold["train"]
    valid = fold["valid"]
    return {
        "scheme": fold["scheme"],
        "valid_year": fold["valid_year"],
        "train": [train.start_date, train.end_date],
        "valid": [valid.start_date, valid.end_date],
    }


def build_fold_args(args: argparse.Namespace, root: Path, variant: str, feature_list: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model=args.model,
        processed_dir=args.processed_dir,
        out_root=str(root / variant),
        run_name=args.run_name,
        no_timestamp=True,
        feature_list=str(feature_list),
        target=args.target,
        raw_return_col=args.raw_return_col,
        daily_return_col=args.daily_return_col,
        valid_years=args.valid_years,
        schemes=["expanding"],
        min_year=args.min_year,
        filter_in_universe=args.filter_in_universe,
        save_models=args.save_models,
        save_predictions=True,
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
        step_days=1,
        tranche_size=args.tranche_size,
        hold_days=args.hold_days,
        transaction_cost_bps=args.transaction_cost_bps,
    )


def flatten_fold_summary(variant: str, summary: dict) -> dict:
    row = {
        "variant": variant,
        "fold": summary.get("fold"),
        "model": summary.get("model"),
        "scheme": summary.get("scheme"),
        "valid_year": summary.get("valid_year"),
        "train_start": summary.get("train_start"),
        "train_end": summary.get("train_end"),
        "valid_start": summary.get("valid_start"),
        "valid_end": summary.get("valid_end"),
        "feature_count": summary.get("feature_count"),
        "train_rows": summary.get("train_rows"),
        "valid_rows": summary.get("valid_rows"),
        "train_sec": summary.get("train_sec"),
    }
    metrics = summary.get("metrics", {})
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
        ]:
            row[key] = metrics.get(key)
    return row


def evaluate_step1_topk(pred_path: Path, args: argparse.Namespace) -> list[dict]:
    pred = pd.read_parquet(pred_path)
    rows = []
    for drop in args.topk_drops:
        cfg = BacktestConfig(
            mode="topk",
            n_hold=int(args.n_hold),
            k_rotate=int(drop),
            step_days=1,
            transaction_cost_bps=float(args.transaction_cost_bps),
        )
        metrics = backtest_topk(pred, return_col=args.raw_return_col, cfg=cfg)
        rows.append(
            {
                "topk": int(args.n_hold),
                "drop": int(drop),
                "step_days": 1,
                "topk_total_return": metrics.get("bt_total_return"),
                "topk_annual_return": metrics.get("bt_annual_return"),
                "topk_sharpe": metrics.get("bt_sharpe"),
                "topk_max_drawdown": metrics.get("bt_max_drawdown"),
                "topk_avg_turnover": metrics.get("bt_avg_turnover"),
                "topk_periods": metrics.get("bt_periods"),
            }
        )
    return rows


def summarize_by_variant(fold_metrics: pd.DataFrame, topk_metrics: pd.DataFrame) -> pd.DataFrame:
    base = (
        fold_metrics.groupby("variant", as_index=False)
        .agg(
            folds=("fold", "count"),
            feature_count=("feature_count", "first"),
            ic_mean_avg=("ic_mean", "mean"),
            ic_mean_min=("ic_mean", "min"),
            icir_avg=("icir", "mean"),
            icir_min=("icir", "min"),
            ic_positive_years=("ic_mean", lambda s: int((s > 0).sum())),
            rolling_sharpe_avg=("rolling_bt_sharpe", "mean"),
            rolling_max_drawdown_min=("rolling_bt_max_drawdown", "min"),
        )
    )
    topk = (
        topk_metrics.groupby(["variant", "drop"], as_index=False)
        .agg(
            topk_sharpe_avg=("topk_sharpe", "mean"),
            topk_sharpe_min=("topk_sharpe", "min"),
            topk_max_drawdown_min=("topk_max_drawdown", "min"),
            topk_avg_turnover_avg=("topk_avg_turnover", "mean"),
        )
    )
    pieces = [base]
    for drop, group in topk.groupby("drop", sort=True):
        renamed = group.drop(columns=["drop"]).rename(
            columns={
                "topk_sharpe_avg": f"topk_drop{drop}_sharpe_avg",
                "topk_sharpe_min": f"topk_drop{drop}_sharpe_min",
                "topk_max_drawdown_min": f"topk_drop{drop}_max_drawdown_min",
                "topk_avg_turnover_avg": f"topk_drop{drop}_avg_turnover_avg",
            }
        )
        pieces.append(renamed)
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on="variant", how="left")
    return out.sort_values("icir_avg", ascending=False, kind="mergesort").reset_index(drop=True)


def write_report(root: Path, by_variant: pd.DataFrame, fold_metrics: pd.DataFrame, topk_metrics: pd.DataFrame, args: argparse.Namespace) -> None:
    report = [
        "# label1d 窗口 expanding walk-forward 验证",
        "",
        f"输出目录：`{root}`",
        "",
        "## 实验口径",
        "",
        f"- 模型：`{args.model}`",
        f"- 目标：`{args.target}`",
        f"- 股票池过滤：`filter_in_universe={args.filter_in_universe}`",
        f"- 验证年份：`{' '.join(str(y) for y in args.valid_years)}`",
        f"- 训练方式：`expanding`，即训练区间从 `{args.min_year}` 年开始，逐年扩展到验证年前一年。",
        f"- 日频 topk 评估：`topk={args.n_hold}`，`step_days=1`，`drop={list(args.topk_drops)}`。",
        "",
        "## 变体汇总",
        "",
        markdown_table(by_variant),
        "",
        "## 每年 ICIR",
        "",
        markdown_table(
            fold_metrics.pivot(index="valid_year", columns="variant", values="icir")
            .reset_index()
            .sort_values("valid_year")
        ),
        "",
        "## 每年 topk Sharpe",
        "",
    ]
    for drop in args.topk_drops:
        sub = topk_metrics[topk_metrics["drop"].eq(int(drop))]
        report.extend(
            [
                f"### topk20_drop{drop}",
                "",
                markdown_table(
                    sub.pivot(index="valid_year", columns="variant", values="topk_sharpe")
                    .reset_index()
                    .sort_values("valid_year")
                ),
                "",
            ]
        )
    report.extend(
        [
            "## 解读原则",
            "",
            "优先选择每年 IC 为正、平均 ICIR 高、topk Sharpe 跨年份稳定、最大回撤不过大、换手不过高的窗口组合。",
            "",
            "如果某个变体只在一两个年份明显领先，但其他年份波动较大，应视为研究候选，而不是直接替换 live 模型。",
            "",
        ]
    )
    (root / "walkforward_report.md").write_text("\n".join(report), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    root = make_run_dir(args.out_root, args.run_name, timestamped=not args.no_timestamp)
    root.mkdir(parents=True, exist_ok=True)
    feature_root = Path(args.feature_root)
    variants = list(args.variants or DEFAULT_VARIANTS)
    pcfg = ProcessedConfig(processed_dir=args.processed_dir)
    folds = build_folds(args.valid_years, ["expanding"], args.min_year)
    write_run_metadata(
        root,
        command="label1d-window-walkforward",
        args=args,
        inputs={"feature_root": str(feature_root), "variants": variants, "folds": [jsonable_fold(fold) for fold in folds]},
    )

    summaries = []
    fold_rows = []
    topk_rows = []
    for variant in variants:
        feature_list = feature_list_path(feature_root, variant)
        fold_args = build_fold_args(args, root, variant, feature_list)
        feature_cols = resolve_features(pcfg, str(feature_list))
        for fold in folds:
            print(
                json.dumps({"stage": "fold_start", "variant": variant, "fold": jsonable_fold(fold)}, ensure_ascii=False),
                flush=True,
            )
            summary = run_fold(fold_args, pcfg, feature_cols, fold)
            summaries.append({"variant": variant, "summary": summary})
            fold_row = flatten_fold_summary(variant, summary)
            fold_rows.append(fold_row)
            pred_path = root / variant / args.model / str(summary["fold"]) / "valid_pred.parquet"
            for topk_row in evaluate_step1_topk(pred_path, args):
                topk_rows.append({**fold_row, **topk_row})
            print(json.dumps({"stage": "fold_done", "variant": variant, "fold": summary["fold"]}, ensure_ascii=False), flush=True)

    fold_metrics = pd.DataFrame(fold_rows)
    topk_metrics = pd.DataFrame(topk_rows)
    by_variant = summarize_by_variant(fold_metrics, topk_metrics)
    fold_metrics.to_csv(root / "walkforward_fold_metrics.csv", index=False)
    topk_metrics.to_csv(root / "walkforward_topk_step1_metrics.csv", index=False)
    by_variant.to_csv(root / "walkforward_by_variant.csv", index=False)
    write_json(root / "walkforward_summary.json", {"experiments": summaries})
    write_report(root, by_variant, fold_metrics, topk_metrics, args)
    return root


def run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lightgbm", "xgboost"], default="lightgbm")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-root", default="outputs/models")
    parser.add_argument("--run-name", default="label1d_window_walkforward")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--feature-root", default="outputs/models/20260609_152416__label1d_window_ablation/features")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--target", default="label_1d__cs_rank")
    parser.add_argument("--raw-return-col", default="label_1d")
    parser.add_argument("--daily-return-col", default="label_1d")
    parser.add_argument("--valid-years", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025, 2026])
    parser.add_argument("--min-year", type=int, default=2016)
    parser.add_argument("--filter-in-universe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-models", action="store_true")
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
    parser.add_argument("--k-rotate", type=int, default=3)
    parser.add_argument("--topk-drops", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--tranche-size", type=int, default=4)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    root = run(parser.parse_args())
    print(json.dumps({"out_dir": str(root)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run_cli()
