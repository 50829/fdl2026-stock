from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from src.utils import slugify_run_name


CANONICAL_ROOTS = {"figures", "live", "models", "strategy"}
RUN_PREFIX_RE = re.compile(r"^20\d{6}_\d{6}__")
RUN_SUFFIX_DATETIME_RE = re.compile(r"^(?P<name>.+)_(?P<date>20\d{6})_(?P<time>\d{6})$")
RUN_SUFFIX_DATE_RE = re.compile(r"^(?P<name>.+)_(?P<date>20\d{6})$")
LIVE_FROM_RE = re.compile(r"^(?P<name>.+)_(?P<trade>20\d{6})_from_(?P<decision>20\d{6})(?P<suffix>.*)$")
LIVE_NO_FROM_RE = re.compile(r"^(?P<name>.+)_(?P<trade>20\d{6})$")


@dataclass(frozen=True)
class MovePlan:
    src: str
    dst: str
    reason: str


def normalize_live_dir_name(name: str) -> str | None:
    if re.match(r"^20\d{6}__", name):
        return None
    cleaned = name.strip("_")
    match = LIVE_FROM_RE.match(cleaned)
    if match:
        suffix = match.group("suffix").strip("_")
        normalized = f"{match.group('trade')}__{match.group('name')}__from_{match.group('decision')}"
        if suffix:
            normalized += f"__{suffix}"
        return normalized
    match = LIVE_NO_FROM_RE.match(cleaned)
    if match:
        return f"{match.group('trade')}__{match.group('name')}"
    return None


def timestamp_from_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")


def normalize_run_dir_name(path: Path) -> str | None:
    name = path.name
    if RUN_PREFIX_RE.match(name):
        return None
    match = RUN_SUFFIX_DATETIME_RE.match(name)
    if match:
        return f"{match.group('date')}_{match.group('time')}__{slugify_run_name(match.group('name'))}"
    match = RUN_SUFFIX_DATE_RE.match(name)
    if match:
        return f"{match.group('date')}_000000__{slugify_run_name(match.group('name'))}"
    return f"{timestamp_from_mtime(path)}__{slugify_run_name(name)}"


def discover_moves(outputs_root: str | Path) -> list[MovePlan]:
    root = Path(outputs_root)
    moves: list[MovePlan] = []
    if not root.exists():
        return moves

    models_root = root / "models"
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name in CANONICAL_ROOTS:
            continue
        moves.append(
            MovePlan(
                src=str(child),
                dst=str(models_root / child.name),
                reason="legacy model output at outputs root",
            )
        )

    live_root = root / "live"
    if live_root.exists():
        for child in sorted(live_root.iterdir()):
            if not child.is_dir():
                continue
            normalized = normalize_live_dir_name(child.name)
            if normalized and normalized != child.name:
                moves.append(
                    MovePlan(
                        src=str(child),
                        dst=str(live_root / normalized),
                        reason="legacy live output name",
                    )
                )
    for run_root_name in ["models", "strategy"]:
        run_root = root / run_root_name
        if not run_root.exists():
            continue
        for child in sorted(run_root.iterdir()):
            if not child.is_dir():
                continue
            normalized = normalize_run_dir_name(child)
            if normalized and normalized != child.name:
                moves.append(
                    MovePlan(
                        src=str(child),
                        dst=str(run_root / normalized),
                        reason=f"legacy {run_root_name} run name",
                    )
                )
    return moves


def apply_moves(moves: list[MovePlan], *, dry_run: bool) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for move in moves:
        src = Path(move.src)
        dst = Path(move.dst)
        status = "planned"
        if not src.exists():
            status = "missing"
        elif dst.exists():
            status = "conflict"
        elif not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            status = "moved"
        results.append({**asdict(move), "status": status})
    return results


def run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--apply", action="store_true", help="Move directories instead of only printing the plan.")
    parser.add_argument("--dry-run", action="store_true", help="Print the migration plan without moving directories.")
    args = parser.parse_args()
    if args.apply and args.dry_run:
        parser.error("--apply and --dry-run are mutually exclusive")

    dry_run = not args.apply
    root = Path(args.outputs_root)
    for name in ["models", "strategy", "live"]:
        if not dry_run:
            (root / name).mkdir(parents=True, exist_ok=True)

    moves = discover_moves(root)
    results = apply_moves(moves, dry_run=dry_run)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "outputs_root": str(root),
        "dry_run": dry_run,
        "moves": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not dry_run:
        manifest = root / "normalize_outputs_manifest.json"
        manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run_cli()
