from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation import BacktestConfig, ic_metrics, max_drawdown, sharpe_ratio
from src.utils import write_json


def load_pred(path: str | Path, label_col: str, raw_return_col: str, daily_return_col: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"trade_date", "ts_code", "pred", label_col, raw_return_col, daily_return_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df["ts_code"] = df["ts_code"].astype(str)
    return df.dropna(subset=["pred", label_col]).reset_index(drop=True)


def base_metrics(df: pd.DataFrame, label_col: str) -> dict:
    out = {"samples": int(len(df))}
    out.update(ic_metrics(df, label_col=label_col))
    return out


def yearly_metrics(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []
    for year, part in df.groupby(df["trade_date"].str[:4], sort=True):
        row = {"year": str(year), **base_metrics(part, label_col)}
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_day_map(df: pd.DataFrame, cols: list[str]) -> dict[str, pd.DataFrame]:
    use_cols = ["trade_date", "ts_code", "pred"] + cols
    out = {}
    for d, g in df[use_cols].dropna(subset=["pred"]).groupby("trade_date", sort=True):
        day = g.sort_values("pred", ascending=False, kind="mergesort").set_index("ts_code")
        out[str(d)] = day
    return out


def fast_topk(day_map: dict[str, pd.DataFrame], return_col: str, cfg: BacktestConfig) -> dict:
    dates = sorted(day_map)[:: max(1, int(cfg.step_days))]
    holdings: set[str] = set()
    equity = 1.0
    curve = []

    for d in dates:
        day = day_map[d]
        if day.empty:
            continue
        if not holdings:
            picks = day.head(cfg.n_hold).index.astype(str).tolist()
            buys = len(picks)
            sells = 0
            holdings = set(picks)
        else:
            held = day[day.index.isin(holdings)].sort_values("pred", ascending=True, kind="mergesort")
            sell_list = held.head(min(cfg.k_rotate, len(held))).index.astype(str).tolist()
            after_sell = holdings - set(sell_list)
            need = max(0, cfg.n_hold - len(after_sell))
            buy_list = day[~day.index.isin(after_sell)].head(need).index.astype(str).tolist()
            holdings = after_sell | set(buy_list)
            sells = len(sell_list)
            buys = len(buy_list)

        held_day = day[day.index.isin(holdings)]
        gross_ret = float(held_day[return_col].mean()) if not held_day.empty else 0.0
        turnover = float((buys + sells) / max(1, cfg.n_hold))
        net_ret = gross_ret - turnover * cfg.transaction_cost_bps / 10000.0
        equity *= 1.0 + net_ret
        curve.append({"trade_date": d, "net_ret": net_ret, "turnover": turnover, "equity": equity})

    curve_df = pd.DataFrame(curve)
    if curve_df.empty:
        return {
            "bt_periods": 0,
            "bt_mode": "topk",
            "bt_total_return": float("nan"),
            "bt_annual_return": float("nan"),
            "bt_sharpe": float("nan"),
            "bt_max_drawdown": float("nan"),
            "bt_avg_turnover": float("nan"),
        }
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
        "bt_sharpe": sharpe_ratio(curve_df["net_ret"].to_numpy(), cfg.trading_days_per_year / cfg.step_days),
        "bt_max_drawdown": max_drawdown(curve_df["equity"].to_numpy()),
        "bt_avg_turnover": float(curve_df["turnover"].mean()),
    }


def fast_rolling(day_map: dict[str, pd.DataFrame], daily_return_col: str, cfg: BacktestConfig) -> dict:
    dates = sorted(day_map)
    active: list[dict[str, object]] = []
    equity = 1.0
    curve = []
    tranche_size = max(1, int(cfg.tranche_size))
    hold_days = max(1, int(cfg.hold_days))
    target_active = tranche_size * hold_days

    for d in dates:
        day = day_map[d]
        expired_codes: list[str] = []
        next_active: list[dict[str, object]] = []
        for tr in active:
            if int(tr["days_left"]) <= 0:
                expired_codes.extend(list(tr["codes"]))
            else:
                next_active.append(tr)
        active = next_active

        held_after_expiry = {code for tr in active for code in list(tr["codes"])}
        buy_list = day[~day.index.isin(held_after_expiry)].head(tranche_size).index.astype(str).tolist()
        if buy_list:
            active.append({"codes": buy_list, "days_left": hold_days})

        active_codes: list[str] = []
        for tr in active:
            active_codes.extend(list(tr["codes"]))
        held_ret = day.loc[day.index.intersection(active_codes), daily_return_col]
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
                "turnover": turnover,
                "active_positions": int(sum(len(list(tr["codes"])) for tr in active)),
                "equity": equity,
            }
        )

    curve_df = pd.DataFrame(curve)
    if curve_df.empty:
        return {
            "bt_periods": 0,
            "bt_mode": "rolling_tranche",
            "bt_total_return": float("nan"),
            "bt_annual_return": float("nan"),
            "bt_sharpe": float("nan"),
            "bt_max_drawdown": float("nan"),
            "bt_avg_turnover": float("nan"),
        }
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
        "bt_daily_return_col": daily_return_col,
        "bt_transaction_cost_bps": float(cfg.transaction_cost_bps),
        "bt_total_return": total_return,
        "bt_annual_return": annual_return,
        "bt_sharpe": sharpe_ratio(curve_df["net_ret"].to_numpy(), cfg.trading_days_per_year),
        "bt_max_drawdown": max_drawdown(curve_df["equity"].to_numpy()),
        "bt_avg_turnover": float(curve_df["turnover"].mean()),
        "bt_avg_active_positions": float(curve_df["active_positions"].mean()),
    }


def topk_sensitivity(day_map: dict[str, pd.DataFrame], raw_return_col: str, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for n_hold in args.n_hold_grid:
        for k_rotate in args.k_rotate_grid:
            if k_rotate > n_hold:
                continue
            for step_days in args.step_days_grid:
                for cost in args.cost_grid:
                    cfg = BacktestConfig(
                        mode="topk",
                        n_hold=int(n_hold),
                        k_rotate=int(k_rotate),
                        step_days=int(step_days),
                        transaction_cost_bps=float(cost),
                    )
                    m = fast_topk(day_map, return_col=raw_return_col, cfg=cfg)
                    rows.append({"n_hold": n_hold, "k_rotate": k_rotate, "step_days": step_days, "cost_bps": cost, **m})
    return pd.DataFrame(rows).sort_values(["bt_sharpe", "bt_total_return"], ascending=False, kind="mergesort")


def rolling_sensitivity(day_map: dict[str, pd.DataFrame], daily_return_col: str, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for hold_days in args.hold_days_grid:
        for target_active in args.target_active_grid:
            tranche_size = max(1, int(round(target_active / hold_days)))
            for cost in args.cost_grid:
                cfg = BacktestConfig(
                    mode="rolling_tranche",
                    tranche_size=tranche_size,
                    hold_days=int(hold_days),
                    daily_return_col=daily_return_col,
                    transaction_cost_bps=float(cost),
                )
                m = fast_rolling(day_map, daily_return_col=daily_return_col, cfg=cfg)
                rows.append(
                    {
                        "target_active": target_active,
                        "hold_days": hold_days,
                        "tranche_size": tranche_size,
                        "cost_bps": cost,
                        **m,
                    }
                )
    return pd.DataFrame(rows).sort_values(["bt_sharpe", "bt_total_return"], ascending=False, kind="mergesort")


def run_one(name: str, split: str, path: str, out_root: Path, args: argparse.Namespace) -> dict:
    df = load_pred(path, args.label_col, args.raw_return_col, args.daily_return_col)
    out_dir = out_root / name / split
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = base_metrics(df, args.label_col)
    write_json(out_dir / "overall_ic.json", overall)

    years = yearly_metrics(df, args.label_col)
    years.to_csv(out_dir / "yearly_ic.csv", index=False)

    day_map = prepare_day_map(df, [args.raw_return_col, args.daily_return_col])
    topk = topk_sensitivity(day_map, args.raw_return_col, args)
    topk.to_csv(out_dir / "topk_sensitivity.csv", index=False)

    rolling = rolling_sensitivity(day_map, args.daily_return_col, args)
    rolling.to_csv(out_dir / "rolling_sensitivity.csv", index=False)

    summary = {
        "name": name,
        "split": split,
        "path": path,
        "overall": overall,
        "best_topk_by_sharpe": topk.iloc[0].to_dict() if not topk.empty else {},
        "best_rolling_by_sharpe": rolling.iloc[0].to_dict() if not rolling.empty else {},
        "worst_year_ic": years.sort_values("ic_mean", ascending=True, kind="mergesort").iloc[0].to_dict()
        if not years.empty
        else {},
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="outputs/sdd_gbdt_sensitivity")
    parser.add_argument("--label-col", default="label_5d__cs_rank")
    parser.add_argument("--raw-return-col", default="label_5d")
    parser.add_argument("--daily-return-col", default="label_1d")
    parser.add_argument("--n-hold-grid", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--k-rotate-grid", nargs="+", type=int, default=[2, 5, 10])
    parser.add_argument("--step-days-grid", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--hold-days-grid", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--target-active-grid", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--cost-grid", nargs="+", type=float, default=[0.0, 5.0, 10.0])
    parser.add_argument(
        "--pred",
        nargs=3,
        action="append",
        metavar=("NAME", "SPLIT", "PATH"),
        default=[],
        help="Prediction triplet. Can be repeated.",
    )
    args = parser.parse_args()

    if not args.pred:
        args.pred = [
            ("lightgbm", "valid", "outputs/sdd_gbdt_full/lightgbm/valid/valid_pred.parquet"),
            ("lightgbm", "test", "outputs/sdd_gbdt_full/lightgbm/test/test_pred.parquet"),
            ("xgboost", "valid", "outputs/sdd_gbdt_full/xgboost/valid/valid_pred.parquet"),
            ("xgboost", "test", "outputs/sdd_gbdt_full/xgboost/test/test_pred.parquet"),
        ]

    out_root = Path(args.out_root)
    summaries = [run_one(name, split, path, out_root, args) for name, split, path in args.pred]
    write_json(out_root / "summary.json", {"experiments": summaries})
