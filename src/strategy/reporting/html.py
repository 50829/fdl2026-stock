from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


CORE_VARIANTS = [
    "topk20_drop3",
    "topk20_drop5",
    "rankbuf_p20_b30_s100_min2_max10",
    "rankbuf_p20_b50_s100_min2_max10",
    "rolling_p10_h5",
    "rolling_p20_h3",
]


def _fmt(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _pct(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.{digits}f}%"


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _metric_table(df: pd.DataFrame, *, top_n: int = 12) -> str:
    if df.empty:
        return "<p class='muted'>没有可展示的结果。</p>"
    cols = ["display_name", "split", "total_return", "annual_return", "sharpe", "max_drawdown", "avg_turnover", "avg_n_holdings"]
    show = df[[col for col in cols if col in df.columns]].head(top_n).copy()
    headers = {
        "display_name": "模型 / 策略",
        "split": "区间",
        "total_return": "总收益",
        "annual_return": "年化收益",
        "sharpe": "Sharpe",
        "max_drawdown": "最大回撤",
        "avg_turnover": "换手",
        "avg_n_holdings": "持仓数",
    }
    lines = ["<table>", "<thead><tr>" + "".join(f"<th>{headers.get(col, col)}</th>" for col in show.columns) + "</tr></thead>", "<tbody>"]
    for _, row in show.iterrows():
        cells: list[str] = []
        for col in show.columns:
            value = row[col]
            if col in {"total_return", "annual_return", "max_drawdown"}:
                text = _pct(value)
            elif col in {"sharpe", "avg_turnover", "avg_n_holdings"}:
                text = _fmt(value, 2)
            else:
                text = str(value)
            cells.append(f"<td>{escape(text)}</td>")
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def _heat_color(value: float, lo: float, hi: float) -> str:
    if hi <= lo:
        ratio = 0.5
    else:
        ratio = (value - lo) / (hi - lo)
    ratio = max(0.0, min(1.0, ratio))
    red = round(245 - 120 * ratio)
    green = round(245 - 70 * (1 - ratio))
    blue = round(245 - 145 * ratio)
    return f"rgb({red},{green},{blue})"


def _heatmap(df: pd.DataFrame, split: str, metric: str = "sharpe", top_n: int = 16) -> str:
    sub = df[(df["split"] == split) & ~df["is_benchmark"].fillna(False)].copy()
    if sub.empty or metric not in sub:
        return "<p class='muted'>没有热力图数据。</p>"
    variants = (
        sub.groupby("variant", as_index=False)[metric]
        .max()
        .sort_values(metric, ascending=False, kind="mergesort")
        .head(top_n)["variant"]
        .tolist()
    )
    pivot = sub[sub["variant"].isin(variants)].pivot_table(index="variant", columns="model", values=metric, aggfunc="max")
    pivot = pivot.loc[variants]
    values = pivot.to_numpy(dtype=float)
    finite = values[pd.notna(values)]
    lo = float(finite.min()) if len(finite) else 0.0
    hi = float(finite.max()) if len(finite) else 1.0
    lines = ["<table class='heatmap'>", "<thead><tr><th>策略参数</th>" + "".join(f"<th>{escape(str(col))}</th>" for col in pivot.columns) + "</tr></thead>", "<tbody>"]
    for variant, row in pivot.iterrows():
        cells = [f"<td class='rowhead'>{escape(str(variant))}</td>"]
        for value in row:
            if pd.isna(value):
                cells.append("<td></td>")
            else:
                cells.append(f"<td style='background:{_heat_color(float(value), lo, hi)}'>{_fmt(value, 2)}</td>")
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def _plot_card(root: Path, rel_path: str, title: str, note: str = "") -> str:
    path = root / rel_path
    if not path.exists():
        return ""
    parts = [
        "<article class='plot-card'>",
        f"<div class='plot-head'><h3>{escape(title)}</h3><a href='{escape(rel_path)}'>打开 SVG</a></div>",
    ]
    if note:
        parts.append(f"<p>{escape(note)}</p>")
    parts.append(f"<img src='{escape(rel_path)}' alt='{escape(title)}'></article>")
    return "".join(parts)


def _plot_links(root: Path, split: str, subdir: str) -> str:
    directory = root / split / subdir
    if not directory.exists():
        return "<p class='muted'>没有生成对应图表。</p>"
    links = []
    for path in sorted(directory.glob("*.svg")):
        links.append(f"<a href='{escape(_rel(root, path))}'>{escape(path.stem)}</a>")
    return "<div class='link-grid'>" + "".join(links) + "</div>"


def write_html_report(
    out_root: str | Path,
    *,
    metrics_path: str | Path,
    benchmark_note: str = "",
    title: str | None = None,
) -> Path:
    root = Path(out_root)
    metrics = pd.read_csv(metrics_path)
    if "is_benchmark" not in metrics:
        metrics["is_benchmark"] = False
    metrics["is_benchmark"] = metrics["is_benchmark"].fillna(False).astype(bool)
    ranked = metrics[~metrics["is_benchmark"]].sort_values(["split", "sharpe", "total_return"], ascending=[True, False, False], kind="mergesort")
    run_title = title or root.name

    sections: list[str] = [
        "<!doctype html>",
        "<html lang='zh-CN'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{escape(run_title)} 策略报告</title>",
        "<style>",
        "body{font-family:Arial,'Noto Sans CJK SC','Microsoft YaHei',sans-serif;margin:0;background:#f7f7f4;color:#202124;}",
        "main{max-width:1320px;margin:0 auto;padding:28px 28px 56px;}",
        "h1{font-size:28px;margin:0 0 8px;} h2{font-size:22px;margin:34px 0 12px;} h3{font-size:16px;margin:0;}",
        ".muted{color:#666;} .summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:18px 0;}",
        ".card,.plot-card{background:white;border:1px solid #ddd;border-radius:8px;padding:14px;box-shadow:0 1px 2px rgba(0,0,0,.04);}",
        ".plot-card{margin:16px 0;} .plot-card img{width:100%;height:auto;border:1px solid #eee;background:white;}",
        ".plot-head{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:8px;}",
        "a{color:#0b57d0;text-decoration:none;} a:hover{text-decoration:underline;}",
        "table{width:100%;border-collapse:collapse;background:white;border:1px solid #ddd;margin:12px 0 20px;}",
        "th,td{padding:8px 10px;border-bottom:1px solid #eee;text-align:right;font-size:13px;} th:first-child,td:first-child{text-align:left;}",
        "th{background:#f1f3f4;font-weight:700;} .rowhead{font-weight:600;text-align:left!important;}",
        ".heatmap td,.heatmap th{text-align:center;} .heatmap td:first-child,.heatmap th:first-child{text-align:left;}",
        ".link-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:8px;margin:12px 0 24px;}",
        ".link-grid a{display:block;background:white;border:1px solid #ddd;border-radius:6px;padding:8px 10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}",
        "</style>",
        "</head>",
        "<body><main>",
        f"<h1>{escape(run_title)} 策略回测报告</h1>",
        "<p class='muted'>本报告由标准长表自动生成。选股信号只使用模型预测分数，收益和风险指标用于事后评估。</p>",
    ]
    if benchmark_note:
        sections.append(f"<p class='muted'>基准：{escape(benchmark_note)}</p>")

    cards = []
    for split in sorted(metrics["split"].dropna().astype(str).unique()):
        sub = ranked[ranked["split"] == split]
        if sub.empty:
            continue
        best = sub.iloc[0]
        cards.append(
            "<section class='card'>"
            f"<h3>{escape(split)} 最优 Sharpe</h3>"
            f"<p><strong>{escape(str(best['display_name']))}</strong></p>"
            f"<p>Sharpe {_fmt(best['sharpe'], 2)}，总收益 {_pct(best['total_return'])}，回撤 {_pct(best['max_drawdown'])}，换手 {_fmt(best['avg_turnover'], 2)}</p>"
            "</section>"
        )
    sections.append("<div class='summary'>" + "".join(cards) + "</div>")

    sections.extend(
        [
            "<h2>关键图</h2>",
            _plot_card(root, "valid/equity_top_valid_sharpe.svg", "valid：按 valid Sharpe 选出的核心对比", "先看 valid，避免按 test 结果挑参数。"),
            _plot_card(root, "test/equity_top_valid_sharpe.svg", "test：同一套选择规则的外推表现", "用于检验 valid 上的选择是否稳定。"),
            _plot_card(root, "test/plots_by_variant/topk20_drop3.svg", "test：topk20_drop3 模型对比", "固定策略参数，只比较 label1d 和 label5d 模型。"),
            _plot_card(root, "test/plots_by_variant/rankbuf_p20_b50_s100_min2_max10.svg", "test：rank buffer 模型对比", "观察高换手缓冲策略下的收益和回撤差异。"),
            "<h2>Sharpe 热力图</h2>",
        ]
    )
    for split in sorted(metrics["split"].dropna().astype(str).unique()):
        sections.append(f"<h3>{escape(split)}</h3>")
        sections.append(_heatmap(metrics, split, metric="sharpe"))

    for split in sorted(metrics["split"].dropna().astype(str).unique()):
        sections.append(f"<h2>{escape(split)} 排名前列</h2>")
        sub = ranked[ranked["split"] == split]
        sections.append(_metric_table(sub, top_n=14))

    sections.extend(["<h2>具体策略参数图</h2>"])
    for split in sorted(metrics["split"].dropna().astype(str).unique()):
        sections.append(f"<h3>{escape(split)}</h3>")
        sections.append(_plot_links(root, split, "plots_by_variant"))

    sections.extend(["<h2>模型视角图</h2>"])
    for split in sorted(metrics["split"].dropna().astype(str).unique()):
        sections.append(f"<h3>{escape(split)}</h3>")
        sections.append(_plot_links(root, split, "plots_by_model"))

    core_existing = [variant for variant in CORE_VARIANTS if ((root / "test" / "plots_by_variant" / f"{variant}.svg").exists())]
    if core_existing:
        sections.append("<h2>核心策略快速入口</h2>")
        links = []
        for variant in core_existing:
            for split in ["valid", "test"]:
                path = root / split / "plots_by_variant" / f"{variant}.svg"
                if path.exists():
                    links.append(f"<a href='{escape(_rel(root, path))}'>{escape(split)} / {escape(variant)}</a>")
        sections.append("<div class='link-grid'>" + "".join(links) + "</div>")

    sections.extend(["</main></body></html>"])
    out = root / "report.html"
    out.write_text("\n".join(sections), encoding="utf-8")
    return out
