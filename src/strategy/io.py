from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_strategy_outputs(result: dict[str, Any], out_dir: str | Path) -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    result["curve"].to_csv(path / "equity_curve.csv", index=False)
    result["trades"].to_csv(path / "trades.csv", index=False)
    result["holdings"].to_csv(path / "holdings.csv", index=False)
    with (path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, ensure_ascii=False, indent=2)
