from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import make_run_dir, write_json, write_run_metadata


GRID_ROOT = Path("outputs/strategy/20260613_162348__report_strategy_grid_final")
REALISTIC_ROOT = Path("outputs/strategy/20260613_162611__report_strategy_realistic_final")
COST_ROOT = Path("outputs/strategy/20260613_162834__report_strategy_cost_sensitivity_final")
RISK_ROOT = Path("outputs/strategy/20260613_163507__report_strategy_risk_sweep_final")

MODEL_LABELS = {
    "label1d_lgb_xgb_rank": "label1d LGB+XGB rank融合",
    "label5d_lgb": "label5d LightGBM",
    "label5d_xgb": "label5d XGBoost",
}

STRATEGY_LABELS = {
    "topk10_drop2": "TopK10 Drop2",
    "topk20_drop3": "TopK20 Drop3",
    "rolling_p10_h5": "Rolling P10 H5",
    "rankbuf_p20_b50_s100_min2_max10": "Rank Buffer P20 B50 S100",
}

RISK_LABELS = {
    "none": "无风控",
    "market_medium": "市场压力降仓",
    "dd_medium": "组合回撤控制",
    "combined_medium": "市场+回撤联合控制",
    "combined_hard": "强联合控制",
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _variant(row: pd.Series) -> str:
    name = str(row.get("name", ""))
    model = str(row.get("model", ""))
    prefix = f"{model}__"
    if name.startswith(prefix):
        return name[len(prefix) :]
    return name


def _decorate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "variant" not in out.columns:
        out["variant"] = out.apply(_variant, axis=1)
    out["模型"] = out["model"].map(MODEL_LABELS).fillna(out["model"])
    out["策略"] = out["variant"].map(STRATEGY_LABELS).fillna(out["variant"])
    return out


def _metric_columns(prefix: str = "") -> list[str]:
    return [
        f"{prefix}总收益",
        f"{prefix}年化收益",
        f"{prefix}夏普",
        f"{prefix}最大回撤",
        f"{prefix}平均换手",
        f"{prefix}平均持仓数",
    ]


def _select_metric_cols(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    rename = {
        "total_return": f"{prefix}总收益",
        "annual_return": f"{prefix}年化收益",
        "sharpe": f"{prefix}夏普",
        "max_drawdown": f"{prefix}最大回撤",
        "avg_turnover": f"{prefix}平均换手",
        "avg_n_holdings": f"{prefix}平均持仓数",
    }
    cols = ["model", "variant", "模型", "策略"] + [col for col in rename if col in df.columns]
    out = df[cols].rename(columns=rename)
    return out


def build_main_strategy_table(grid_root: Path) -> pd.DataFrame:
    valid = _decorate(_read(grid_root / "valid" / "strategy_metrics.csv"))
    test = _decorate(_read(grid_root / "test" / "strategy_metrics.csv"))
    valid = valid[~valid["strategy"].astype(str).str.startswith("benchmark")]
    test = test[~test["strategy"].astype(str).str.startswith("benchmark")]
    left = _select_metric_cols(valid, "Valid ")
    right = _select_metric_cols(test, "Test ").drop(columns=["模型", "策略"])
    out = left.merge(right, on=["model", "variant"], how="outer")
    order = ["模型", "策略", *_metric_columns("Valid "), *_metric_columns("Test ")]
    return out[[col for col in order if col in out.columns]].sort_values(["模型", "策略"], kind="mergesort")


def build_label_strategy_fit_table(grid_root: Path) -> pd.DataFrame:
    test = _decorate(_read(grid_root / "test" / "strategy_metrics.csv"))
    test = test[~test["strategy"].astype(str).str.startswith("benchmark")].copy()
    keep = {
        ("label1d_lgb_xgb_rank", "topk10_drop2"),
        ("label1d_lgb_xgb_rank", "topk20_drop3"),
        ("label5d_lgb", "rolling_p10_h5"),
        ("label5d_lgb", "topk10_drop2"),
        ("label5d_xgb", "rolling_p10_h5"),
        ("label5d_xgb", "topk10_drop2"),
    }
    test = test[test.apply(lambda row: (row["model"], row["variant"]) in keep, axis=1)]
    out = _select_metric_cols(test, "Test ")
    out["策略适配结论"] = out.apply(_fit_note, axis=1)
    order = ["模型", "策略", "策略适配结论", *_metric_columns("Test ")]
    return out[[col for col in order if col in out.columns]].sort_values(["模型", "策略"], kind="mergesort")


def _fit_note(row: pd.Series) -> str:
    model = str(row["模型"])
    strategy = str(row["策略"])
    if model.startswith("label1d") and strategy == "TopK10 Drop2":
        return "1日信号适合每日高频滚动换仓，收益弹性最高"
    if model.startswith("label1d") and strategy == "TopK20 Drop3":
        return "更分散，回撤略低，可作为保守备选"
    if model.startswith("label5d") and strategy == "Rolling P10 H5":
        return "更贴近5日标签持有期，换手和回撤较低"
    if model.startswith("label5d") and strategy == "TopK10 Drop2":
        return "低成本下收益高，但换手明显更高"
    return ""


def build_realistic_table(grid_root: Path, realistic_root: Path) -> pd.DataFrame:
    base = _decorate(_read(grid_root / "test" / "strategy_metrics.csv"))
    real = _decorate(_read(realistic_root / "test" / "strategy_metrics.csv"))
    base = base[~base["strategy"].astype(str).str.startswith("benchmark")]
    real = real[~real["strategy"].astype(str).str.startswith("benchmark")]
    base_m = _select_metric_cols(base, "基础5bps ")
    real_m = _select_metric_cols(real, "真实25bps ")
    out = base_m.merge(real_m.drop(columns=["模型", "策略"]), on=["model", "variant"], how="inner")
    out["夏普下降"] = out["基础5bps 夏普"] - out["真实25bps 夏普"]
    out["收益下降"] = out["基础5bps 总收益"] - out["真实25bps 总收益"]
    order = [
        "模型",
        "策略",
        "基础5bps 总收益",
        "基础5bps 夏普",
        "基础5bps 最大回撤",
        "基础5bps 平均换手",
        "真实25bps 总收益",
        "真实25bps 夏普",
        "真实25bps 最大回撤",
        "真实25bps 平均换手",
        "收益下降",
        "夏普下降",
    ]
    return out[[col for col in order if col in out.columns]].sort_values(["模型", "策略"], kind="mergesort")


def build_cost_table(cost_root: Path) -> pd.DataFrame:
    df = _decorate(_read(cost_root / "sensitivity_metrics.csv"))
    df = df[df["split"].astype(str).eq("test")].copy()
    out = df[
        [
            "模型",
            "策略",
            "total_cost_bps",
            "total_return",
            "annual_return",
            "sharpe",
            "max_drawdown",
            "avg_turnover",
        ]
    ].rename(
        columns={
            "total_cost_bps": "总成本bps",
            "total_return": "Test 总收益",
            "annual_return": "Test 年化收益",
            "sharpe": "Test 夏普",
            "max_drawdown": "Test 最大回撤",
            "avg_turnover": "Test 平均换手",
        }
    )
    return out.sort_values(["模型", "策略", "总成本bps"], kind="mergesort")


def build_risk_table(risk_root: Path) -> pd.DataFrame:
    df = _decorate(_read(risk_root / "risk_sweep_metrics.csv"))
    df = df[df["split"].astype(str).eq("test")].copy()
    df["风控方案"] = df["risk_profile"].map(RISK_LABELS).fillna(df["risk_profile"])
    out = df[
        [
            "模型",
            "策略",
            "风控方案",
            "total_return",
            "annual_return",
            "sharpe",
            "max_drawdown",
            "avg_turnover",
            "avg_gross_exposure",
            "market_stress_days",
            "drawdown_control_days",
        ]
    ].rename(
        columns={
            "total_return": "Test 总收益",
            "annual_return": "Test 年化收益",
            "sharpe": "Test 夏普",
            "max_drawdown": "Test 最大回撤",
            "avg_turnover": "Test 平均换手",
            "avg_gross_exposure": "平均总仓位",
            "market_stress_days": "市场降仓天数",
            "drawdown_control_days": "回撤控制天数",
        }
    )
    return out.sort_values(["模型", "策略", "风控方案"], kind="mergesort")


def build_benchmark_table(grid_root: Path) -> pd.DataFrame:
    test = _decorate(_read(grid_root / "test" / "strategy_metrics.csv"))
    rows = test[
        test["name"].isin(
            [
                "benchmark_000300_sh_weight",
                "benchmark_equal_weight_universe",
                "label1d_lgb_xgb_rank__topk10_drop2",
                "label1d_lgb_xgb_rank__topk20_drop3",
            ]
        )
    ].copy()
    rows["对比对象"] = rows["name"].replace(
        {
            "benchmark_000300_sh_weight": "沪深300成分权重基准",
            "benchmark_equal_weight_universe": "可交易股票池等权基准",
            "label1d_lgb_xgb_rank__topk10_drop2": "主模型 TopK10 Drop2",
            "label1d_lgb_xgb_rank__topk20_drop3": "主模型 TopK20 Drop3",
        }
    )
    out = rows[
        [
            "对比对象",
            "total_return",
            "annual_return",
            "sharpe",
            "max_drawdown",
            "avg_turnover",
            "avg_n_holdings",
        ]
    ].rename(
        columns={
            "total_return": "Test 总收益",
            "annual_return": "Test 年化收益",
            "sharpe": "Test 夏普",
            "max_drawdown": "Test 最大回撤",
            "avg_turnover": "Test 平均换手",
            "avg_n_holdings": "平均持仓数",
        }
    )
    return out


def build_constraint_table(realistic_root: Path) -> pd.DataFrame:
    df = _read(realistic_root / "trade_constraint_summary.csv")
    df["模型"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    return df[["模型", "split", "rows", "matched_rate", "buyable_rate", "rules"]].rename(
        columns={
            "split": "样本",
            "rows": "样本行数",
            "matched_rate": "约束匹配率",
            "buyable_rate": "可买样本占比",
            "rules": "约束规则",
        }
    )


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Build final Chinese strategy report tables from completed strategy outputs.")
    parser.add_argument("--out-root", default="outputs/report_tables")
    parser.add_argument("--run-name", default="final_strategy_report_tables")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--grid-root", default=str(GRID_ROOT))
    parser.add_argument("--realistic-root", default=str(REALISTIC_ROOT))
    parser.add_argument("--cost-root", default=str(COST_ROOT))
    parser.add_argument("--risk-root", default=str(RISK_ROOT))
    args = parser.parse_args()

    out_dir = make_run_dir(args.out_root, args.run_name, timestamped=not args.no_timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(out_dir, command="final-strategy-report-tables", args=args)

    grid_root = Path(args.grid_root)
    realistic_root = Path(args.realistic_root)
    cost_root = Path(args.cost_root)
    risk_root = Path(args.risk_root)

    outputs: dict[str, str] = {}
    tables: list[tuple[str, pd.DataFrame]] = [
        ("策略主实验表.csv", build_main_strategy_table(grid_root)),
        ("label1d_label5d策略适配表.csv", build_label_strategy_fit_table(grid_root)),
        ("真实交易约束影响表.csv", build_realistic_table(grid_root, realistic_root)),
        ("交易成本敏感性表.csv", build_cost_table(cost_root)),
        ("风控实验表.csv", build_risk_table(risk_root)),
        ("基准对比表.csv", build_benchmark_table(grid_root)),
        ("可买约束覆盖表.csv", build_constraint_table(realistic_root)),
    ]
    for filename, table in tables:
        path = out_dir / filename
        table.to_csv(path, index=False)
        outputs[filename] = str(path)

    write_json(
        out_dir / "summary.json",
        {
            "grid_root": str(grid_root),
            "realistic_root": str(realistic_root),
            "cost_root": str(cost_root),
            "risk_root": str(risk_root),
            "tables": outputs,
        },
    )
    print(json.dumps({"saved": str(out_dir), "tables": outputs}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run_cli()
