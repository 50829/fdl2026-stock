from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.strategy import StrategyBacktestConfig, run_strategy
from src.utils import make_run_dir, write_json, write_run_metadata


LABEL1D_ROOT = Path("outputs/models/20260612_151735__nsntk_inspired_label1d")
EMA_ROOT = Path("outputs/models/20260612_172058__nsntk_ema_grid_label1d")
TREE_DEEP_ROOT = Path("outputs/models/20260613_012152__tree_deep_ema_fusion_backtest_leftjoin")
LABEL5D_LGB = Path("outputs/models/20260611_210718__report_label5d_lgb_top40_rerun/lightgbm/test/test_metrics.json")
LABEL5D_XGB = Path("outputs/models/20260611_210936__report_label5d_xgb_top40_rerun/xgboost/test/test_metrics.json")
LABEL5D_RIDGE = Path("outputs/models/20260611_212000__report_label5d_residual_rank_rerun/stacking_ridge/test/test_metrics.json")
LABEL_COL = "label_1d__cs_rank"
RETURN_COL = "label_1d"

COLORS = {
    "tree": "#4c78a8",
    "deep": "#72b7b2",
    "fusion": "#8f7aa8",
    "warn": "#b2795b",
    "muted": "#bab0ac",
    "green": "#59a14f",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_row(name: str, path: Path, group: str) -> dict[str, Any]:
    data = _read_json(path)
    keys = ["ic_mean", "icir", "bt_total_return", "bt_sharpe", "bt_max_drawdown", "bt_avg_turnover"]
    return {"name": name, "group": group, "path": str(path), **{k: data.get(k) for k in keys}}


def _save_bar(df: pd.DataFrame, *, metric: str, title: str, path: Path, color_col: str = "group") -> None:
    fig, ax = plt.subplots(figsize=(max(7.0, len(df) * 0.85), 4.2))
    colors = [COLORS.get(str(g), COLORS["muted"]) for g in df[color_col]]
    ax.bar(df["name"], df[metric], color=colors, width=0.72)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=28)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _strategy_curve(pred_path: Path, out_dir: Path, name: str) -> pd.DataFrame:
    df = pd.read_parquet(pred_path)
    cfg = StrategyBacktestConfig(
        strategy="topk_drop",
        score_col="pred",
        return_col=RETURN_COL,
        topk=10,
        drop=2,
        transaction_cost_bps=5.0,
        slippage_bps=0.0,
    )
    result = run_strategy(df, cfg, name=name)
    curve = result["curve"].copy()
    curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1.0
    curve["name"] = name
    out_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out_dir / f"{name}_curve.csv", index=False)
    pd.DataFrame([result["metrics"]]).to_csv(out_dir / f"{name}_strategy_metrics.csv", index=False)
    return curve


def _plot_curves(curves: list[pd.DataFrame], out_dir: Path) -> None:
    palette = ["#4c78a8", "#72b7b2", "#8f7aa8", "#b2795b", "#59a14f"]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for idx, curve in enumerate(curves):
        ax.plot(pd.to_datetime(curve["trade_date"]), curve["equity"], label=curve["name"].iloc[0], color=palette[idx % len(palette)], linewidth=1.8)
    ax.set_title("TopK10 Drop2 equity")
    ax.set_ylabel("equity")
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "topk10_drop2_equity.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for idx, curve in enumerate(curves):
        ax.plot(pd.to_datetime(curve["trade_date"]), curve["drawdown"], label=curve["name"].iloc[0], color=palette[idx % len(palette)], linewidth=1.6)
    ax.set_title("Drawdown")
    ax.set_ylabel("drawdown")
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "topk10_drop2_drawdown.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for idx, curve in enumerate(curves):
        turnover = curve["turnover"].rolling(20, min_periods=1).mean()
        ax.plot(pd.to_datetime(curve["trade_date"]), turnover, label=curve["name"].iloc[0], color=palette[idx % len(palette)], linewidth=1.5)
    ax.set_title("20-day average turnover")
    ax.set_ylabel("turnover")
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "topk10_drop2_turnover.svg")
    plt.close(fig)


def _plot_monthly_yearly(out_dir: Path) -> None:
    monthly = pd.read_csv(LABEL1D_ROOT / "main_model_stability/test/test_monthly_ic.csv")
    yearly = pd.read_csv(LABEL1D_ROOT / "main_model_stability/test/test_yearly_ic.csv")

    month_col = "month" if "month" in monthly.columns else monthly.columns[0]
    ic_col = "ic_mean" if "ic_mean" in monthly.columns else [c for c in monthly.columns if "ic" in c.lower()][0]
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    ax.plot(pd.to_datetime(monthly[month_col].astype(str)), monthly[ic_col], color=COLORS["tree"], linewidth=1.6)
    ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_title("Main model monthly IC")
    ax.set_ylabel("IC")
    ax.grid(alpha=0.25, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "main_model_monthly_ic.svg")
    plt.close(fig)

    year_col = "year" if "year" in yearly.columns else yearly.columns[0]
    icir_col = "icir" if "icir" in yearly.columns else [c for c in yearly.columns if "icir" in c.lower()][0]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(yearly[year_col].astype(str), yearly[icir_col], color=COLORS["tree"])
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_title("Main model yearly ICIR")
    ax.set_ylabel("ICIR")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "main_model_yearly_icir.svg")
    plt.close(fig)


def _selected_prediction_path(candidate: str, alpha: float, seq_len_run: Path | None) -> Path | None:
    alpha_tag = str(float(alpha)).replace(".", "_")
    candidates = [
        TREE_DEEP_ROOT / candidate / f"test_selected_alpha_{alpha_tag}/test_pred.parquet",
    ]
    if seq_len_run:
        candidates.append(seq_len_run / "fusion" / candidate / f"test_selected_alpha_{alpha_tag}/test_pred.parquet")
    for path in candidates:
        if path.exists() and path.suffix == ".parquet":
            return path
    return None


def _deep_raw_vs_ema() -> pd.DataFrame:
    rows = []
    summary = pd.read_csv(EMA_ROOT / "seed_ensemble_summary.csv")
    for model in ["mlp", "gru"]:
        sub = summary[(summary["model"].eq(model)) & (summary["split"].eq("test"))]
        for variant in ["raw", "ema_0_99", "ema_0_995", "ema_0_999"]:
            hit = sub[sub["variant"].eq(variant)]
            if not hit.empty:
                row = hit.iloc[0].to_dict()
                rows.append(
                    {
                        "name": f"{model.upper()} {variant}",
                        "model": model,
                        "variant": variant,
                        "group": "deep",
                        "ic_mean": row.get("ic_mean"),
                        "icir": row.get("icir"),
                        "bt_sharpe": row.get("bt_sharpe"),
                        "bt_max_drawdown": row.get("bt_max_drawdown"),
                    }
                )
    seed = pd.read_csv(EMA_ROOT / "deep_ema_seed_summary.csv")
    tcn = seed[(seed["model"].eq("tcn")) & (seed["split"].eq("test"))]
    for variant in ["raw", "ema_0_99", "ema_0_995", "ema_0_999"]:
        hit = tcn[tcn["variant"].eq(variant)]
        if not hit.empty:
            row = hit.iloc[0].to_dict()
            rows.append(
                {
                    "name": f"TCN {variant}",
                    "model": "tcn",
                    "variant": variant,
                    "group": "deep",
                    "ic_mean": row.get("ic_mean"),
                    "icir": row.get("icir"),
                    "bt_sharpe": row.get("bt_sharpe"),
                    "bt_max_drawdown": row.get("bt_max_drawdown"),
                }
            )
    return pd.DataFrame(rows)


def _existing_fusion_rows() -> pd.DataFrame:
    path = TREE_DEEP_ROOT / "selected_by_valid_sharpe.csv"
    if not path.exists():
        return pd.DataFrame()
    selected = pd.read_csv(path)
    rows = selected[selected["selection"].eq("valid_selected_test_once")].copy()
    name_col = "candidate" if "candidate" in rows.columns else "experiment"
    rows["candidate"] = rows[name_col].astype(str)
    rows["name"] = rows["candidate"].astype(str) + " a=" + rows["alpha_deep"].astype(str)
    rows["group"] = "fusion"
    return rows


def _optional_seq_len_rows(seq_len_run: Path | None) -> pd.DataFrame:
    if not seq_len_run:
        return pd.DataFrame()
    path = seq_len_run / "fusion_selected_by_valid_sharpe.csv"
    if not path.exists():
        return pd.DataFrame()
    selected = pd.read_csv(path)
    rows = selected[selected["selection"].eq("valid_selected_test_once")].copy()
    rows["name"] = rows["candidate"].astype(str) + " a=" + rows["alpha_deep"].astype(str)
    rows["group"] = "fusion"
    return rows


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Plot final report figures from completed experiment outputs.")
    parser.add_argument("--out-root", default="outputs/report_figures")
    parser.add_argument("--run-name", default="final_report_figures")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--seq-len-run", default=None)
    args = parser.parse_args()

    out_dir = make_run_dir(args.out_root, args.run_name, timestamped=not args.no_timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(out_dir, command="final-report-figures", args=args)

    rows = [
        _metric_row("LGB uniform", LABEL1D_ROOT / "time_decay/uniform/lightgbm/test/test_metrics.json", "tree"),
        _metric_row("XGB uniform", LABEL1D_ROOT / "time_decay/uniform/xgboost/test/test_metrics.json", "tree"),
        _metric_row("LGB+XGB rank", LABEL1D_ROOT / "main_model_stability/test/test_metrics.json", "tree"),
        _metric_row("label5d LGB", LABEL5D_LGB, "tree"),
        _metric_row("label5d XGB", LABEL5D_XGB, "tree"),
        _metric_row("label5d Ridge stack", LABEL5D_RIDGE, "fusion"),
    ]
    main_metrics = pd.DataFrame(rows)
    main_metrics.to_csv(out_dir / "main_metrics_summary.csv", index=False)
    _save_bar(main_metrics.iloc[:3], metric="icir", title="Label1d tree model ICIR", path=out_dir / "label1d_tree_model_icir.svg")
    _save_bar(main_metrics.iloc[:3], metric="bt_sharpe", title="Label1d tree model TopK10 Drop2 Sharpe", path=out_dir / "label1d_tree_model_sharpe.svg")
    _save_bar(main_metrics.iloc[2:], metric="icir", title="Label1d vs Label5d ICIR", path=out_dir / "label1d_vs_label5d_icir.svg")

    deep = _deep_raw_vs_ema()
    deep.to_csv(out_dir / "deep_raw_vs_ema_summary.csv", index=False)
    _save_bar(deep, metric="icir", title="Deep raw vs EMA ICIR", path=out_dir / "deep_raw_vs_ema_icir.svg", color_col="model")
    _save_bar(deep, metric="bt_sharpe", title="Deep raw vs EMA Sharpe", path=out_dir / "deep_raw_vs_ema_sharpe.svg", color_col="model")

    fusion = pd.concat([_existing_fusion_rows(), _optional_seq_len_rows(Path(args.seq_len_run) if args.seq_len_run else None)], ignore_index=True)
    if not fusion.empty:
        fusion.to_csv(out_dir / "tree_deep_fusion_summary.csv", index=False)
        plot_cols = ["name", "group", "icir", "bt_sharpe", "bt_max_drawdown", "bt_avg_turnover"]
        _save_bar(fusion[plot_cols], metric="bt_sharpe", title="Tree + deep fusion Sharpe", path=out_dir / "tree_deep_fusion_sharpe.svg")
        _save_bar(fusion[plot_cols], metric="icir", title="Tree + deep fusion ICIR", path=out_dir / "tree_deep_fusion_icir.svg")

    _plot_monthly_yearly(out_dir)
    curve_dir = out_dir / "curves"
    curve_specs = [
        ("tree_rank_mean", LABEL1D_ROOT / "main_model_stability/test/test_pred.parquet"),
        ("score_smooth_0_6", LABEL1D_ROOT / "score_smoothing/gbdt_rank_mean_score_smooth_alpha_0_6/test/test_pred.parquet"),
    ]
    if fusion.empty:
        curve_specs.append(("gru_ema_0_995", EMA_ROOT / "seed_ensemble/gru_ema_0_995_3seed_rank_mean/test/test_pred.parquet"))
    else:
        candidate = fusion.sort_values(["bt_sharpe", "icir"], ascending=False).iloc[0]
        pred_text = str(candidate.get("pred_path", "") or "")
        pred_file = Path(pred_text) if pred_text else Path("__missing_prediction_file__")
        if not pred_file.exists() or pred_file.suffix != ".parquet":
            inferred = _selected_prediction_path(
                str(candidate["candidate"]),
                float(candidate["alpha_deep"]),
                Path(args.seq_len_run) if args.seq_len_run else None,
            )
            if inferred is not None:
                pred_file = inferred
        if pred_file.exists() and pred_file.suffix == ".parquet":
            curve_specs.append(("best_tree_deep_fusion", pred_file))
    curves = [_strategy_curve(path, curve_dir, name) for name, path in curve_specs if path.exists()]
    if curves:
        _plot_curves(curves, out_dir)
        pd.concat(curves, ignore_index=True).to_csv(out_dir / "topk10_drop2_curves.csv", index=False)

    write_json(
        out_dir / "summary.json",
        {
            "main_metrics": str(out_dir / "main_metrics_summary.csv"),
            "deep_raw_vs_ema": str(out_dir / "deep_raw_vs_ema_summary.csv"),
            "tree_deep_fusion": str(out_dir / "tree_deep_fusion_summary.csv") if not fusion.empty else None,
            "curves": str(out_dir / "topk10_drop2_curves.csv") if curves else None,
        },
    )
    print(json.dumps({"saved": str(out_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run_cli()
