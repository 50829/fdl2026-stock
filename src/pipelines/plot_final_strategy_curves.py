from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import make_run_dir, write_json, write_run_metadata


GRID_ROOT = Path("outputs/strategy/20260613_162348__report_strategy_grid_final")
REALISTIC_ROOT = Path("outputs/strategy/20260613_162611__report_strategy_realistic_final")

CURVE_LABELS = {
    "benchmark_000300_sh_weight": "CSI300 weighted",
    "benchmark_equal_weight_universe": "Universe equal weight",
    "label1d_lgb_xgb_rank__topk10_drop2": "label1d main TopK10",
    "label1d_lgb_xgb_rank__topk20_drop3": "label1d main TopK20",
    "label1d_lgb_xgb_rank__rankbuf_p20_b50_s100_min2_max10": "label1d main RankBuffer",
    "label1d_lgb_xgb_rank__rolling_p10_h5": "label1d main Rolling",
    "label5d_xgb__topk10_drop2": "label5d XGB TopK10",
    "label5d_xgb__rolling_p10_h5": "label5d XGB Rolling",
}

CURVE_COLORS = {
    "benchmark_000300_sh_weight": "#4f4a45",
    "benchmark_equal_weight_universe": "#8c857d",
    "label1d_lgb_xgb_rank__topk10_drop2": "#58708f",
    "label1d_lgb_xgb_rank__topk20_drop3": "#7895a8",
    "label1d_lgb_xgb_rank__rankbuf_p20_b50_s100_min2_max10": "#8f735e",
    "label1d_lgb_xgb_rank__rolling_p10_h5": "#7c8d62",
    "label5d_xgb__topk10_drop2": "#9a6f79",
    "label5d_xgb__rolling_p10_h5": "#6d8f81",
}

CURVE_STYLES = {
    "benchmark_000300_sh_weight": "--",
    "benchmark_equal_weight_universe": ":",
}


def _setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "#fbfaf7",
            "axes.facecolor": "#fbfaf7",
            "axes.edgecolor": "#d8d2c9",
            "axes.labelcolor": "#2a2927",
            "xtick.color": "#4d4945",
            "ytick.color": "#4d4945",
            "text.color": "#24211f",
            "legend.frameon": False,
            "axes.titleweight": "bold",
        }
    )


def _load_equity(root: Path) -> pd.DataFrame:
    path = root / "equity_long.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")
    df["cum_return"] = df["equity"].astype(float) - 1.0
    df["drawdown"] = df["drawdown"].astype(float)
    return df


def _plot_lines(
    ax: plt.Axes,
    df: pd.DataFrame,
    names: list[str],
    *,
    value_col: str,
    value_scale: float = 1.0,
    suffix: str = "",
    label_prefix: str = "",
) -> None:
    for name in names:
        sub = df[df["name"].astype(str).eq(name)].sort_values("trade_date")
        if sub.empty:
            continue
        label = label_prefix + CURVE_LABELS.get(name, name)
        color = CURVE_COLORS.get(name, "#666666")
        linestyle = CURVE_STYLES.get(name, "-")
        linewidth = 2.4 if not name.startswith("benchmark") else 1.8
        ax.plot(
            sub["trade_date"],
            sub[value_col].astype(float) * value_scale,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.96,
        )
    ax.grid(True, color="#e7e0d7", linewidth=0.8, alpha=0.75)
    ax.yaxis.set_major_formatter(lambda x, _pos: f"{x:.0f}{suffix}")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


def _finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_valid_return_drawdown(equity: pd.DataFrame, out_dir: Path) -> str:
    names = [
        "label1d_lgb_xgb_rank__topk10_drop2",
        "label1d_lgb_xgb_rank__topk20_drop3",
        "label1d_lgb_xgb_rank__rankbuf_p20_b50_s100_min2_max10",
        "label1d_lgb_xgb_rank__rolling_p10_h5",
        "benchmark_000300_sh_weight",
        "benchmark_equal_weight_universe",
    ]
    df = equity[equity["split"].eq("valid")].copy()
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.15]})
    _plot_lines(axes[0], df, names, value_col="cum_return", value_scale=100.0, suffix="%")
    axes[0].set_title("Valid Time-Return Curve: 2024 Stress Sample")
    axes[0].set_ylabel("Cumulative Return")
    _plot_lines(axes[1], df, names[:4], value_col="drawdown", value_scale=100.0, suffix="%")
    axes[1].set_title("Valid Strategy Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Trade Date")
    axes[0].legend(ncol=3, loc="upper left", fontsize=9)
    path = out_dir / "valid_strategy_return_drawdown.svg"
    _finish(fig, path)
    return str(path)


def plot_test_strategy_vs_benchmark(equity: pd.DataFrame, out_dir: Path) -> str:
    names = [
        "label1d_lgb_xgb_rank__topk10_drop2",
        "label1d_lgb_xgb_rank__topk20_drop3",
        "benchmark_000300_sh_weight",
        "benchmark_equal_weight_universe",
    ]
    df = equity[equity["split"].eq("test")].copy()
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    _plot_lines(ax, df, names, value_col="cum_return", value_scale=100.0, suffix="%")
    ax.set_title("Test Time-Return Curve: Main Strategy vs Benchmarks")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Trade Date")
    ax.legend(ncol=2, loc="upper left", fontsize=9)
    path = out_dir / "test_main_strategy_vs_benchmark_return.svg"
    _finish(fig, path)
    return str(path)


def plot_label_adaptation(equity: pd.DataFrame, out_dir: Path) -> str:
    names = [
        "label1d_lgb_xgb_rank__topk10_drop2",
        "label1d_lgb_xgb_rank__topk20_drop3",
        "label5d_xgb__topk10_drop2",
        "label5d_xgb__rolling_p10_h5",
    ]
    df = equity[equity["split"].eq("test")].copy()
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    _plot_lines(ax, df, names, value_col="cum_return", value_scale=100.0, suffix="%")
    ax.set_title("Test Time-Return Curve: label1d vs label5d Strategy Fit")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Trade Date")
    ax.legend(ncol=2, loc="upper left", fontsize=9)
    path = out_dir / "test_label1d_label5d_strategy_return.svg"
    _finish(fig, path)
    return str(path)


def plot_realistic_cost(base: pd.DataFrame, realistic: pd.DataFrame, out_dir: Path) -> str:
    names = [
        "label1d_lgb_xgb_rank__topk10_drop2",
        "label1d_lgb_xgb_rank__topk20_drop3",
        "label5d_xgb__rolling_p10_h5",
    ]
    base = base[base["split"].eq("test")].copy()
    real = realistic[realistic["split"].eq("test")].copy()
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    for source, df, style_suffix, alpha in [("5bps ", base, "", 0.98), ("25bps ", real, "--", 0.86)]:
        for name in names:
            sub = df[df["name"].astype(str).eq(name)].sort_values("trade_date")
            if sub.empty:
                continue
            ax.plot(
                sub["trade_date"],
                sub["cum_return"] * 100.0,
                label=source + CURVE_LABELS.get(name, name),
                color=CURVE_COLORS.get(name, "#666666"),
                linestyle=style_suffix or "-",
                linewidth=2.2,
                alpha=alpha,
            )
    ax.grid(True, color="#e7e0d7", linewidth=0.8, alpha=0.75)
    ax.yaxis.set_major_formatter(lambda x, _pos: f"{x:.0f}%")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_title("Test Time-Return Curve: Base Cost vs Realistic Cost")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Trade Date")
    ax.legend(ncol=2, loc="upper left", fontsize=8.5)
    path = out_dir / "test_realistic_cost_return.svg"
    _finish(fig, path)
    return str(path)


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Plot final strategy time-return curves for the Chinese report.")
    parser.add_argument("--out-root", default="outputs/report_figures")
    parser.add_argument("--run-name", default="final_strategy_curves")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--grid-root", default=str(GRID_ROOT))
    parser.add_argument("--realistic-root", default=str(REALISTIC_ROOT))
    args = parser.parse_args()

    _setup_matplotlib()
    out_dir = make_run_dir(args.out_root, args.run_name, timestamped=not args.no_timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(out_dir, command="final-strategy-curves", args=args)

    grid = _load_equity(Path(args.grid_root))
    realistic = _load_equity(Path(args.realistic_root))
    paths = {
        "valid_strategy_return_drawdown": plot_valid_return_drawdown(grid, out_dir),
        "test_main_strategy_vs_benchmark_return": plot_test_strategy_vs_benchmark(grid, out_dir),
        "test_label1d_label5d_strategy_return": plot_label_adaptation(grid, out_dir),
        "test_realistic_cost_return": plot_realistic_cost(grid, realistic, out_dir),
    }
    write_json(
        out_dir / "summary.json",
        {
            "grid_root": str(args.grid_root),
            "realistic_root": str(args.realistic_root),
            "figures": paths,
        },
    )
    print(json.dumps({"saved": str(out_dir), "figures": paths}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run_cli()
