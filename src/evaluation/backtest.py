from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import max_drawdown, sharpe_ratio


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


def _empty_topk() -> dict[str, object]:
    return {
        "bt_periods": 0,
        "bt_total_return": math.nan,
        "bt_annual_return": math.nan,
        "bt_sharpe": math.nan,
        "bt_max_drawdown": math.nan,
        "bt_avg_turnover": math.nan,
    }


def backtest_topk(pred_df: pd.DataFrame, return_col: str, cfg: BacktestConfig) -> dict[str, object]:
    if pred_df.empty or return_col not in pred_df.columns:
        return _empty_topk()

    df = pred_df[["trade_date", "ts_code", "pred", return_col]].dropna().copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df["ts_code"] = df["ts_code"].astype(str)
    dates = sorted(df["trade_date"].unique().tolist())[:: max(1, int(cfg.step_days))]

    holdings: set[str] = set()
    equity = 1.0
    curve = []
    for d in dates:
        day = df[df["trade_date"] == d].sort_values("pred", ascending=False, kind="mergesort")
        if day.empty:
            continue

        if not holdings:
            picks = day.head(cfg.n_hold)["ts_code"].tolist()
            buys = len(picks)
            sells = 0
            holdings = set(picks)
        else:
            held = day[day["ts_code"].isin(holdings)].sort_values("pred", ascending=True, kind="mergesort")
            sell_list = held.head(min(cfg.k_rotate, len(held)))["ts_code"].tolist()
            after_sell = holdings - set(sell_list)
            need = max(0, cfg.n_hold - len(after_sell))
            buy_list = day[~day["ts_code"].isin(after_sell)].head(need)["ts_code"].tolist()
            holdings = after_sell | set(buy_list)
            sells = len(sell_list)
            buys = len(buy_list)

        held_day = day[day["ts_code"].isin(holdings)]
        gross_ret = float(held_day[return_col].mean()) if not held_day.empty else 0.0
        turnover = float((buys + sells) / max(1, cfg.n_hold))
        net_ret = gross_ret - turnover * cfg.transaction_cost_bps / 10000.0
        equity *= 1.0 + net_ret
        curve.append({"trade_date": d, "net_ret": net_ret, "turnover": turnover, "equity": equity})

    curve_df = pd.DataFrame(curve)
    if curve_df.empty:
        return _empty_topk()

    periods = int(len(curve_df))
    years = max(1e-12, periods * cfg.step_days / cfg.trading_days_per_year)
    total_return = float(curve_df["equity"].iloc[-1] - 1.0)
    annual_return = float(curve_df["equity"].iloc[-1] ** (1.0 / years) - 1.0)
    return {
        "bt_periods": periods,
        "bt_mode": "topk",
        "bt_step_days": int(cfg.step_days),
        "bt_n_hold": int(cfg.n_hold),
        "bt_k_rotate": int(cfg.k_rotate),
        "bt_transaction_cost_bps": float(cfg.transaction_cost_bps),
        "bt_total_return": total_return,
        "bt_annual_return": annual_return,
        "bt_sharpe": sharpe_ratio(curve_df["net_ret"].to_numpy(dtype=np.float64), cfg.trading_days_per_year / cfg.step_days),
        "bt_max_drawdown": max_drawdown(curve_df["equity"].to_numpy(dtype=np.float64)),
        "bt_avg_turnover": float(curve_df["turnover"].mean()),
    }


def _empty_rolling() -> dict[str, object]:
    return {
        "bt_periods": 0,
        "bt_mode": "rolling_tranche",
        "bt_total_return": math.nan,
        "bt_annual_return": math.nan,
        "bt_sharpe": math.nan,
        "bt_max_drawdown": math.nan,
        "bt_avg_turnover": math.nan,
    }


def backtest_rolling_tranche(pred_df: pd.DataFrame, cfg: BacktestConfig) -> dict[str, object]:
    return_col = str(cfg.daily_return_col)
    if pred_df.empty or return_col not in pred_df.columns:
        return _empty_rolling()

    df = pred_df[["trade_date", "ts_code", "pred", return_col]].dropna().copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df["ts_code"] = df["ts_code"].astype(str)
    dates = sorted(df["trade_date"].unique().tolist())
    day_map = {d: g.set_index("ts_code") for d, g in df.groupby("trade_date", sort=False)}

    active: list[dict[str, object]] = []
    equity = 1.0
    curve = []
    tranche_size = max(1, int(cfg.tranche_size))
    hold_days = max(1, int(cfg.hold_days))
    target_active = tranche_size * hold_days

    for d in dates:
        day = day_map.get(d)
        if day is None or day.empty:
            continue

        expired_codes: list[str] = []
        next_active: list[dict[str, object]] = []
        for tr in active:
            if int(tr["days_left"]) <= 0:
                expired_codes.extend(list(tr["codes"]))
            else:
                next_active.append(tr)
        active = next_active

        held_after_expiry = {code for tr in active for code in list(tr["codes"])}
        ranked = day.sort_values("pred", ascending=False, kind="mergesort")
        buy_list = ranked[~ranked.index.isin(held_after_expiry)].head(tranche_size).index.astype(str).tolist()
        if buy_list:
            active.append({"codes": buy_list, "days_left": hold_days})

        active_codes: list[str] = []
        for tr in active:
            active_codes.extend(list(tr["codes"]))
        held_ret = day.loc[day.index.intersection(active_codes), return_col]
        gross_ret = float(held_ret.mean()) if len(held_ret) else 0.0

        buys = len(buy_list)
        sells = len(expired_codes)
        turnover = float((buys + sells) / max(1, target_active))
        net_ret = gross_ret - turnover * cfg.transaction_cost_bps / 10000.0
        equity *= 1.0 + net_ret
        for tr in active:
            tr["days_left"] = int(tr["days_left"]) - 1
        curve.append(
            {
                "trade_date": d,
                "net_ret": net_ret,
                "gross_ret": gross_ret,
                "turnover": turnover,
                "active_positions": int(sum(len(list(tr["codes"])) for tr in active)),
                "equity": equity,
            }
        )

    curve_df = pd.DataFrame(curve)
    if curve_df.empty:
        return _empty_rolling()

    periods = int(len(curve_df))
    years = max(1e-12, periods / float(cfg.trading_days_per_year))
    total_return = float(curve_df["equity"].iloc[-1] - 1.0)
    annual_return = float(curve_df["equity"].iloc[-1] ** (1.0 / years) - 1.0)
    return {
        "bt_periods": periods,
        "bt_mode": "rolling_tranche",
        "bt_step_days": 1,
        "bt_tranche_size": tranche_size,
        "bt_hold_days": hold_days,
        "bt_target_active": target_active,
        "bt_daily_return_col": return_col,
        "bt_transaction_cost_bps": float(cfg.transaction_cost_bps),
        "bt_total_return": total_return,
        "bt_annual_return": annual_return,
        "bt_sharpe": sharpe_ratio(curve_df["net_ret"].to_numpy(dtype=np.float64), cfg.trading_days_per_year),
        "bt_max_drawdown": max_drawdown(curve_df["equity"].to_numpy(dtype=np.float64)),
        "bt_avg_turnover": float(curve_df["turnover"].mean()),
        "bt_avg_active_positions": float(curve_df["active_positions"].mean()),
    }
