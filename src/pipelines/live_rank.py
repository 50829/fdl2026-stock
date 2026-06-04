from __future__ import annotations

import argparse
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.load import downcast_numeric
from src.data.preprocess import build_features, load_config
from src.models.fusion import add_meta_prediction_features, load_residual_rank_fusion


DATE_RE = re.compile(r"(20\d{6})")


DEFAULT_WATCHLIST = [
    ("新和成", "002001.SZ"),
    ("奥福科技", "688021.SH"),
    ("甘化科工", "000576.SZ"),
    ("山金国际", "000975.SZ"),
    ("天山铝业", "002532.SZ"),
    ("赛福天", "603028.SH"),
    ("太极股份", "002368.SZ"),
    ("河钢资源", "000923.SZ"),
    ("华达新材", "605158.SH"),
    ("中钢天源", "002057.SZ"),
]


def normalize_csv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed:")], errors="ignore")
    if "trade_date" in df.columns:
        df["trade_date"] = df["trade_date"].astype(str)
    if "ts_code" in df.columns:
        df["ts_code"] = df["ts_code"].astype(str)
    return df


def date_from_name(name: str) -> str | None:
    match = DATE_RE.search(Path(name).name)
    return match.group(1) if match else None


def read_zip_group(zip_path: Path, prefix: str, end_date: str) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".csv") or f"/{prefix}/" not in name:
                continue
            file_date = date_from_name(name)
            if file_date and file_date > end_date:
                continue
            with zf.open(name) as fh:
                df = normalize_csv(pd.read_csv(fh))
            if "trade_date" in df.columns:
                df = df[df["trade_date"].astype(str) <= end_date]
            if not df.empty:
                frames.append(df)
    return frames


def read_0602_zip(zip_path: Path, end_date: str) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    if end_date < "20260602":
        return [], [], [], []
    daily: list[pd.DataFrame] = []
    metric: list[pd.DataFrame] = []
    moneyflow: list[pd.DataFrame] = []
    st: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".csv"):
                continue
            with zf.open(name) as fh:
                df = normalize_csv(pd.read_csv(fh))
            if "trade_date" in df.columns:
                df = df[df["trade_date"].astype(str) <= end_date]
            if df.empty:
                continue
            base = Path(name).name.lower()
            if base.startswith("daily open"):
                continue
            if base.startswith("daily "):
                daily.append(df)
            elif base.startswith("metric"):
                metric.append(df)
            elif "moneyflow" in base:
                moneyflow.append(df)
            elif "stock_st" in base:
                st.append(df)
    return daily, metric, moneyflow, st


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def read_optional_csv(path: Path | None) -> list[pd.DataFrame]:
    if path is None:
        return []
    return [normalize_csv(pd.read_csv(path))]


def read_loose_raw(
    raw_dir: Path,
    decision_date: str,
    trade_date: str,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame | None]:
    daily_path = first_existing([raw_dir / f"daily {decision_date}.csv", raw_dir / f"daily_{decision_date}.csv"])
    metric_path = first_existing([raw_dir / f"metric {decision_date}.csv", raw_dir / f"metric_{decision_date}.csv"])
    moneyflow_path = first_existing([raw_dir / f"moneyflow {decision_date}.csv", raw_dir / f"moneyflow_{decision_date}.csv"])
    st_path = first_existing(
        [
            raw_dir / f"stock_st {decision_date}.csv",
            raw_dir / f"stock st {decision_date}.csv",
            raw_dir / "stock_st.csv",
            raw_dir / "stock st .csv",
        ]
    )
    open_path = first_existing(
        [
            raw_dir / f"daily open {trade_date}.csv",
            raw_dir / f"daily open{trade_date}.csv",
            raw_dir / f"daily_open_{trade_date}.csv",
        ]
    )
    open_df = normalize_csv(pd.read_csv(open_path)) if open_path else None
    return (
        read_optional_csv(daily_path),
        read_optional_csv(metric_path),
        read_optional_csv(moneyflow_path),
        read_optional_csv(st_path),
        open_df,
    )


def concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_panel_from_raw(
    raw_dir: Path,
    decision_date: str,
    trade_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, list[str]]:
    daily_frames = read_zip_group(raw_dir / "0601.zip", "daily", decision_date)
    metric_frames = read_zip_group(raw_dir / "0601.zip", "metric", decision_date)
    moneyflow_frames = read_zip_group(raw_dir / "0601.zip", "moneyflow", decision_date)
    st_frames = read_zip_group(raw_dir / "0601.zip", "stock_st", decision_date)

    d2, m2, mf2, st2 = read_0602_zip(raw_dir / "0602.zip", decision_date)
    daily_frames.extend(d2)
    metric_frames.extend(m2)
    moneyflow_frames.extend(mf2)
    st_frames.extend(st2)

    d3, m3, mf3, st3, open_df = read_loose_raw(raw_dir, decision_date, trade_date)
    daily_frames.extend(d3)
    metric_frames.extend(m3)
    moneyflow_frames.extend(mf3)
    st_frames.extend(st3)

    daily = concat(daily_frames).drop_duplicates(["trade_date", "ts_code"], keep="last")
    metric = concat(metric_frames).drop(columns=["close"], errors="ignore").drop_duplicates(["trade_date", "ts_code"], keep="last")
    moneyflow = concat(moneyflow_frames).drop_duplicates(["trade_date", "ts_code"], keep="last")

    panel = daily.merge(metric, on=["trade_date", "ts_code"], how="left")
    panel = panel.merge(moneyflow, on=["trade_date", "ts_code"], how="left")
    panel = panel[panel["trade_date"].astype(str) <= decision_date]
    panel = panel.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)

    st = concat(st_frames)
    if not st.empty:
        st = st[["trade_date", "ts_code"]].drop_duplicates()
        st["is_st"] = True
    else:
        st = pd.DataFrame(columns=["trade_date", "ts_code", "is_st"])
    return downcast_numeric(panel), st, open_df, sorted(panel["trade_date"].astype(str).unique().tolist())


def build_live_universe(panel: pd.DataFrame, st: pd.DataFrame, processed_dir: Path, config: dict) -> pd.DataFrame:
    latest = pd.read_parquet(processed_dir / "universe.parquet", columns=["trade_date", "ts_code", "market", "industry", "listed_days_in_data"])
    latest_date = str(latest["trade_date"].max())
    latest = latest[latest["trade_date"].astype(str) == latest_date].drop(columns=["trade_date"]).drop_duplicates("ts_code")

    universe = panel[["trade_date", "ts_code", "vol", "amount"]].copy()
    universe = universe.merge(latest, on="ts_code", how="left")
    trade_dt = pd.to_datetime(universe["trade_date"], format="%Y%m%d", errors="coerce")
    base_dt = pd.to_datetime(latest_date, format="%Y%m%d")
    universe["listed_days_in_data"] = (
        universe["listed_days_in_data"].fillna(0).astype("float64") + (trade_dt - base_dt).dt.days
    ).clip(lower=0).astype("int32")

    universe = universe.merge(st, on=["trade_date", "ts_code"], how="left")
    universe["is_st"] = universe["is_st"].fillna(False).astype(bool)

    liquidity_cfg = config.get("universe", {}).get("liquidity_filter", {}) or {}
    window = int(liquidity_cfg.get("window", 20))
    sorted_universe = universe.sort_values(["ts_code", "trade_date"])
    universe["amount_mean_20"] = sorted_universe.groupby("ts_code")["amount"].transform(
        lambda s: s.rolling(window, min_periods=min(5, window)).mean()
    ).reindex(universe.index)

    min_list_days = int(config.get("universe", {}).get("min_list_days", 60))
    base_mask = (
        universe["market"].ne("北交所")
        & ~universe["is_st"]
        & universe["vol"].gt(0)
        & universe["amount"].gt(0)
        & universe["listed_days_in_data"].ge(min_list_days)
    )

    if bool(liquidity_cfg.get("enabled", True)) and float(liquidity_cfg.get("bottom_pct", 0.2)) > 0:
        amount_mean = universe["amount_mean_20"].where(base_mask)
        liquidity_rank = amount_mean.groupby(universe["trade_date"]).rank(pct=True)
        liquidity_mask = liquidity_rank.gt(float(liquidity_cfg.get("bottom_pct", 0.2))).fillna(False)
    else:
        liquidity_mask = pd.Series(True, index=universe.index)

    universe["passes_liquidity"] = liquidity_mask.astype(bool)
    universe["in_universe"] = base_mask & universe["passes_liquidity"]
    return universe[
        [
            "trade_date",
            "ts_code",
            "in_universe",
            "is_st",
            "market",
            "industry",
            "listed_days_in_data",
            "amount_mean_20",
            "passes_liquidity",
        ]
    ]


def default_trade_date(decision_date: str) -> str:
    return (pd.Timestamp(decision_date) + pd.Timedelta(days=1)).strftime("%Y%m%d")


def load_watchlist(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(DEFAULT_WATCHLIST, columns=["stock_name", "ts_code"])
    watchlist = pd.read_csv(path)
    if "ts_code" not in watchlist.columns:
        raise ValueError(f"Watchlist {path} must contain a ts_code column")
    if "stock_name" not in watchlist.columns:
        name_col = "name" if "name" in watchlist.columns else None
        watchlist["stock_name"] = watchlist[name_col] if name_col else watchlist["ts_code"]
    out = watchlist[["stock_name", "ts_code"]].copy()
    out["stock_name"] = out["stock_name"].astype(str)
    out["ts_code"] = out["ts_code"].astype(str)
    return out.drop_duplicates("ts_code", keep="first").reset_index(drop=True)


def run(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trade_date = str(args.trade_date or default_trade_date(args.decision_date))
    panel, st, open_df, feature_dates = build_panel_from_raw(raw_dir, args.decision_date, trade_date)
    universe = build_live_universe(panel, st, processed_dir, config)
    features, _ = build_features(panel, universe, config)
    live_features = features[features["trade_date"].astype(str) == args.decision_date].copy()
    live_features.to_parquet(out_dir / f"live_features_{args.decision_date}.parquet", index=False)

    lgb_model = lgb.Booster(model_file=args.lgb_model)
    feature_cols = lgb_model.feature_name()
    missing = sorted(set(feature_cols) - set(live_features.columns))
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    xgb_model = xgb.Booster()
    xgb_model.load_model(args.xgb_model)
    xmat = xgb.DMatrix(live_features[feature_cols].to_numpy(dtype=np.float32, copy=False), feature_names=feature_cols)
    pred = live_features[["trade_date", "ts_code"]].copy()
    pred["pred_lgb"] = lgb_model.predict(live_features[feature_cols].to_numpy(dtype=np.float32, copy=False)).astype(np.float32)
    pred["pred_xgb"] = xgb_model.predict(xmat).astype(np.float32)
    pred = add_meta_prediction_features(pred)

    fusion = load_residual_rank_fusion(args.fusion_model, alpha=args.alpha)
    scored = fusion.predict_frame(pred, device=args.device)
    scored["model_rank"] = scored["final_pred"].rank(method="first", ascending=False).astype(int)

    decision_panel = panel[panel["trade_date"].astype(str) == args.decision_date][["trade_date", "ts_code", "close", "amount"]]
    decision_universe = universe[universe["trade_date"].astype(str) == args.decision_date]
    scored = scored.merge(decision_panel, on=["trade_date", "ts_code"], how="left")
    scored = scored.merge(
        decision_universe[["trade_date", "ts_code", "industry", "is_st", "passes_liquidity", "amount_mean_20"]],
        on=["trade_date", "ts_code"],
        how="left",
    )
    if open_df is not None and "trade_date" in open_df.columns:
        open_df = open_df[open_df["trade_date"].astype(str) == trade_date].copy()
    if open_df is not None and not open_df.empty:
        open_cols = open_df[["ts_code", "open", "pre_close"]].rename(
            columns={"open": f"open_{trade_date}", "pre_close": f"pre_close_{trade_date}"}
        )
        scored = scored.merge(open_cols, on="ts_code", how="left")

    scored = scored.sort_values("model_rank", kind="mergesort").reset_index(drop=True)
    scored.to_csv(out_dir / f"live_predictions_{args.decision_date}.csv", index=False)

    top10 = scored.head(10).copy()
    top10.insert(0, "trade_date_next", trade_date)
    top10.insert(1, "decision_date", args.decision_date)
    top10["target_weight_top10"] = 0.1
    top10.to_csv(out_dir / f"top10_{args.decision_date}_for_{trade_date}.csv", index=False)

    mentioned = load_watchlist(args.watchlist)
    mentioned = mentioned.merge(scored, on="ts_code", how="left")
    mentioned["note"] = np.where(mentioned["model_rank"].isna(), f"not in {args.decision_date} candidate universe", "")
    mentioned.to_csv(out_dir / f"mentioned_stock_ranks_{trade_date}.csv", index=False)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision_date": args.decision_date,
        "trade_date": trade_date,
        "model": args.model_name,
        "candidate_count": int(len(scored)),
        "raw_feature_dates": feature_dates,
        "top10_csv": str(out_dir / f"top10_{args.decision_date}_for_{trade_date}.csv"),
        "mentioned_csv": str(out_dir / f"mentioned_stock_ranks_{trade_date}.csv"),
        "predictions_csv": str(out_dir / f"live_predictions_{args.decision_date}.csv"),
        "features_parquet": str(out_dir / f"live_features_{args.decision_date}.parquet"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-date", required=True)
    parser.add_argument("--trade-date", default=None, help="Trade date for the generated plan. Defaults to decision_date + 1 calendar day.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--lgb-model", default="outputs/models/sdd_feature_selection/lightgbm_top40/lightgbm/model.txt")
    parser.add_argument("--xgb-model", default="outputs/models/sdd_feature_selection/xgboost_top40/xgboost/model.json")
    parser.add_argument("--fusion-model", default="outputs/models/sdd_fusion_rank_tune/alpha_ext_h128_d010_wd1e4/residual_rank_mlp/residual_rank_mlp.pt")
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--watchlist", default=None, help="Optional CSV with stock_name/name and ts_code columns.")
    parser.add_argument("--model-name", default="final = lightgbm_top40 + 1.5 * residual_rank_mlp")
    run(parser.parse_args())


if __name__ == "__main__":
    run_cli()
