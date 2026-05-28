from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    metrics: dict[str, float | int]


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    dd = equity / (peak + 1e-12) - 1.0
    return float(dd.min())


def _annualized_return(equity_end: float, num_days: int, trading_days: int = 252) -> float:
    if num_days <= 0 or equity_end <= 0:
        return float("nan")
    return float(equity_end ** (trading_days / num_days) - 1.0)


def _sharpe(returns: np.ndarray, trading_days: float) -> float:
    if len(returns) < 2:
        return float("nan")
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    if sd <= 0:
        return float("nan")
    return float(mu / sd * np.sqrt(trading_days))


def backtest_topk_rotate(
    pred_df: pd.DataFrame,
    n_hold: int = 20,
    k_rotate: int = 5,
    horizon: int = 1,
    transaction_cost_bps: float = 0.0,
    trading_days_per_year: int = 252,
) -> BacktestResult:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if n_hold < 1:
        raise ValueError("n_hold must be >= 1")
    if k_rotate < 0:
        raise ValueError("k_rotate must be >= 0")

    df = pred_df[["trade_date", "ts_code", "pred", "label"]].copy()
    df = df.dropna(subset=["pred", "label"]).copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df["ts_code"] = df["ts_code"].astype(str)
    df = df.sort_values(["trade_date", "ts_code"], kind="mergesort")

    dates = sorted(df["trade_date"].unique().tolist())
    if not dates:
        raise ValueError("No usable rows in pred_df for backtest")

    step = int(horizon)
    if step > 1:
        dates = dates[::step]

    holdings: set[str] = set()
    equity = 1.0
    rows = []

    for d in dates:
        day = df[df["trade_date"] == d]
        if day.empty:
            continue

        day_sorted = day.sort_values("pred", ascending=False, kind="mergesort")

        if not holdings:
            picks = day_sorted.head(n_hold)["ts_code"].tolist()
            holdings = set(picks)
            sells = 0
            buys = len(holdings)
        else:
            held = day_sorted[day_sorted["ts_code"].isin(holdings)]
            sells_list: list[str] = []
            if k_rotate > 0 and not held.empty:
                held_sorted = held.sort_values("pred", ascending=True, kind="mergesort")
                sells_list = held_sorted.head(min(k_rotate, len(held_sorted)))["ts_code"].tolist()
            after_sell = set(holdings) - set(sells_list)

            need = n_hold - len(after_sell)
            candidates = day_sorted[~day_sorted["ts_code"].isin(after_sell)]
            buy_list = candidates.head(max(0, need))["ts_code"].tolist()

            holdings = after_sell | set(buy_list)
            sells = len(sells_list)
            buys = len(buy_list)

        held_day = day[day["ts_code"].isin(holdings)]
        if held_day.empty:
            port_ret = 0.0
        else:
            port_ret = float(held_day["label"].mean())

        turnover = (sells + buys) / max(1, n_hold)
        cost = turnover * (transaction_cost_bps / 10000.0)
        net_ret = port_ret - cost
        equity *= 1.0 + net_ret

        rows.append(
            {
                "trade_date": d,
                "gross_ret": port_ret,
                "net_ret": net_ret,
                "turnover": turnover,
                "holdings": len(holdings),
                "equity": equity,
            }
        )

    curve = pd.DataFrame(rows)
    if curve.empty:
        raise ValueError("Backtest produced empty curve")

    num_periods = int(len(curve))
    num_days = int(num_periods * step)
    ann_ret = _annualized_return(float(curve["equity"].iloc[-1]), num_days=num_days, trading_days=trading_days_per_year)
    sharpe = _sharpe(curve["net_ret"].to_numpy(dtype=np.float64), trading_days=float(trading_days_per_year / step))
    mdd = _max_drawdown(curve["equity"].to_numpy(dtype=np.float64))
    avg_turnover = float(curve["turnover"].mean())

    metrics = {
        "periods": num_periods,
        "step_days": step,
        "total_return": float(curve["equity"].iloc[-1] - 1.0),
        "annual_return": float(ann_ret),
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd),
        "avg_turnover": float(avg_turnover),
        "n_hold": int(n_hold),
        "k_rotate": int(k_rotate),
        "transaction_cost_bps": float(transaction_cost_bps),
    }
    return BacktestResult(equity_curve=curve, metrics=metrics)
