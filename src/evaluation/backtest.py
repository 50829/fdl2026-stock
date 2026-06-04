from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from src.strategy import StrategyBacktestConfig, run_strategy


@dataclass(frozen=True)
class BacktestConfig:
    mode: str = "topk"
    n_hold: int = 20
    k_rotate: int = 5
    step_days: int = 5
    tranche_size: int = 4
    hold_days: int = 5
    daily_return_col: str = "label_1d"
    transaction_cost_bps: float = 5.0
    trading_days_per_year: int = 252


def backtest_config_from_cfg(cfg: dict) -> BacktestConfig:
    bt_cfg = cfg.get("backtest", {})
    return BacktestConfig(
        mode=str(bt_cfg.get("mode", "topk")),
        n_hold=int(bt_cfg.get("n_hold", 20)),
        k_rotate=int(bt_cfg.get("k_rotate", 5)),
        step_days=int(bt_cfg.get("step_days", 5)),
        tranche_size=int(bt_cfg.get("tranche_size", 4)),
        hold_days=int(bt_cfg.get("hold_days", 5)),
        daily_return_col=str(bt_cfg.get("daily_return_col", "label_1d")),
        transaction_cost_bps=float(bt_cfg.get("transaction_cost_bps", 5.0)),
        trading_days_per_year=int(bt_cfg.get("trading_days_per_year", 252)),
    )


def _empty(mode: str) -> dict[str, object]:
    out = {
        "bt_periods": 0,
        "bt_mode": mode,
        "bt_total_return": math.nan,
        "bt_annual_return": math.nan,
        "bt_sharpe": math.nan,
        "bt_max_drawdown": math.nan,
        "bt_avg_turnover": math.nan,
    }
    if mode == "rolling_tranche":
        out["bt_avg_active_positions"] = math.nan
    return out


def _prepare_frame(pred_df: pd.DataFrame, return_col: str) -> pd.DataFrame:
    df = pred_df[["trade_date", "ts_code", "pred", return_col]].dropna().copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df["ts_code"] = df["ts_code"].astype(str)
    return df.sort_values(["trade_date", "pred"], ascending=[True, False], kind="mergesort").reset_index(drop=True)


def _strategy_metrics(result: dict[str, object]) -> dict[str, object]:
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    return metrics


def backtest_topk(pred_df: pd.DataFrame, return_col: str, cfg: BacktestConfig) -> dict[str, object]:
    if pred_df.empty or return_col not in pred_df.columns:
        return _empty("topk")

    step_days = max(1, int(cfg.step_days))
    df = _prepare_frame(pred_df, return_col)
    keep_dates = sorted(df["trade_date"].unique().tolist())[::step_days]
    df = df[df["trade_date"].isin(keep_dates)].copy()
    if df.empty:
        return _empty("topk")

    strategy_cfg = StrategyBacktestConfig(
        strategy="topk_drop",
        score_col="pred",
        return_col=return_col,
        transaction_cost_bps=float(cfg.transaction_cost_bps),
        trading_days_per_year=float(cfg.trading_days_per_year) / step_days,
        topk=int(cfg.n_hold),
        drop=int(cfg.k_rotate),
    )
    metrics = _strategy_metrics(run_strategy(df, strategy_cfg, name="topk"))
    return {
        "bt_periods": int(metrics.get("periods", 0)),
        "bt_mode": "topk",
        "bt_step_days": step_days,
        "bt_n_hold": int(cfg.n_hold),
        "bt_k_rotate": int(cfg.k_rotate),
        "bt_transaction_cost_bps": float(cfg.transaction_cost_bps),
        "bt_total_return": metrics.get("total_return", math.nan),
        "bt_annual_return": metrics.get("annual_return", math.nan),
        "bt_sharpe": metrics.get("sharpe", math.nan),
        "bt_max_drawdown": metrics.get("max_drawdown", math.nan),
        "bt_avg_turnover": metrics.get("avg_turnover", math.nan),
    }


def backtest_rolling_tranche(pred_df: pd.DataFrame, cfg: BacktestConfig) -> dict[str, object]:
    return_col = str(cfg.daily_return_col)
    if pred_df.empty or return_col not in pred_df.columns:
        return _empty("rolling_tranche")

    df = _prepare_frame(pred_df, return_col)
    if df.empty:
        return _empty("rolling_tranche")

    tranche_size = max(1, int(cfg.tranche_size))
    hold_days = max(1, int(cfg.hold_days))
    strategy_cfg = StrategyBacktestConfig(
        strategy="rolling_tranche",
        score_col="pred",
        return_col=return_col,
        transaction_cost_bps=float(cfg.transaction_cost_bps),
        trading_days_per_year=int(cfg.trading_days_per_year),
        target_positions=tranche_size * hold_days,
        hold_days=hold_days,
        daily_buy=tranche_size,
    )
    metrics = _strategy_metrics(run_strategy(df, strategy_cfg, name="rolling_tranche"))
    return {
        "bt_periods": int(metrics.get("periods", 0)),
        "bt_mode": "rolling_tranche",
        "bt_step_days": 1,
        "bt_tranche_size": tranche_size,
        "bt_hold_days": hold_days,
        "bt_target_active": tranche_size * hold_days,
        "bt_daily_return_col": return_col,
        "bt_transaction_cost_bps": float(cfg.transaction_cost_bps),
        "bt_total_return": metrics.get("total_return", math.nan),
        "bt_annual_return": metrics.get("annual_return", math.nan),
        "bt_sharpe": metrics.get("sharpe", math.nan),
        "bt_max_drawdown": metrics.get("max_drawdown", math.nan),
        "bt_avg_turnover": metrics.get("avg_turnover", math.nan),
        "bt_avg_active_positions": metrics.get("avg_n_holdings", math.nan),
    }
