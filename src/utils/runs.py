from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .io import read_yaml


_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MULTI_SEP_RE = re.compile(r"[_-]{2,}")


def current_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify_run_name(name: str) -> str:
    slug = _UNSAFE_CHARS_RE.sub("_", str(name).strip())
    slug = _MULTI_SEP_RE.sub("_", slug).strip("._-")
    if not slug:
        raise ValueError("run name cannot be empty after normalization")
    return slug


def format_run_dir_name(run_name: str, timestamp: str | None = None) -> str:
    slug = slugify_run_name(run_name)
    return f"{timestamp or current_run_timestamp()}__{slug}"


def make_run_dir(out_root: str | Path, run_name: str, *, timestamped: bool = True) -> Path:
    root = Path(out_root)
    if timestamped:
        return root / format_run_dir_name(run_name)
    return root / slugify_run_name(run_name)


def _git_text(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def git_snapshot() -> dict[str, Any]:
    status = _git_text(["status", "--short"])
    return {
        "commit": _git_text(["rev-parse", "HEAD"]),
        "branch": _git_text(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def registry_snapshots(paths: list[str | Path] | None) -> dict[str, Any]:
    snapshots: dict[str, Any] = {}
    for path in paths or []:
        p = Path(path)
        if p.exists():
            snapshots[str(p)] = read_yaml(p)
    return snapshots


def write_run_metadata(
    out_dir: str | Path,
    *,
    command: str,
    args: Any | None = None,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    registry_paths: list[str | Path] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "command": command,
        "argv": sys.argv[:],
        "args": jsonable(vars(args) if hasattr(args, "__dict__") else (args or {})),
        "inputs": jsonable(inputs or {}),
        "outputs": jsonable(outputs or {}),
        "registries": jsonable(registry_snapshots(registry_paths)),
        "git": git_snapshot(),
    }
    if extra:
        payload["extra"] = jsonable(extra)
    path = out / "run_meta.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
