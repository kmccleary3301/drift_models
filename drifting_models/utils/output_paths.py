from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"
_STABLE_PATTERN = re.compile(r"^stable_\d{8}_\d{6}$")
_EXPERIMENTAL_PATTERN = re.compile(r"^exp_[a-z0-9_]+_\d{8}_\d{6}$")


def make_run_timestamp(*, now_utc: datetime | None = None) -> str:
    now = datetime.now(timezone.utc) if now_utc is None else now_utc.astimezone(timezone.utc)
    return now.strftime(_TIMESTAMP_FMT)


def make_stable_run_name(*, timestamp: str | None = None) -> str:
    stamp = make_run_timestamp() if timestamp is None else timestamp
    return f"stable_{stamp}"


def make_experimental_run_name(*, name: str, timestamp: str | None = None) -> str:
    slug = slugify_experiment_name(name)
    stamp = make_run_timestamp() if timestamp is None else timestamp
    return f"exp_{slug}_{stamp}"


def build_stable_run_root(*, base_dir: Path, timestamp: str | None = None) -> Path:
    return base_dir / make_stable_run_name(timestamp=timestamp)


def build_experimental_run_root(*, base_dir: Path, name: str, timestamp: str | None = None) -> Path:
    return base_dir / make_experimental_run_name(name=name, timestamp=timestamp)


def slugify_experiment_name(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    if not slug:
        raise ValueError("Experiment name must contain at least one alphanumeric character")
    return slug


def is_stable_run_name(name: str) -> bool:
    return _STABLE_PATTERN.fullmatch(name) is not None


def is_experimental_run_name(name: str) -> bool:
    return _EXPERIMENTAL_PATTERN.fullmatch(name) is not None
