from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


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
