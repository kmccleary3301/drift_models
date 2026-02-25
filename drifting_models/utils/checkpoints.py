from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StepCheckpoint:
    step: int
    path: Path


_STEP_PATTERN = re.compile(r"^checkpoint_step_(\d{8})\.pt$")


def parse_step_checkpoint(path: Path) -> StepCheckpoint | None:
    match = _STEP_PATTERN.match(path.name)
    if not match:
        return None
    return StepCheckpoint(step=int(match.group(1)), path=path)


def list_step_checkpoints(checkpoint_dir: Path) -> list[StepCheckpoint]:
    checkpoints: list[StepCheckpoint] = []
    for path in checkpoint_dir.iterdir():
        if not path.is_file():
            continue
        parsed = parse_step_checkpoint(path)
        if parsed is None:
            continue
        checkpoints.append(parsed)
    checkpoints.sort(key=lambda item: item.step)
    return checkpoints


def select_last_k_checkpoints(checkpoint_dir: Path, *, k: int) -> list[StepCheckpoint]:
    if k <= 0:
        return []
    items = list_step_checkpoints(checkpoint_dir)
    return items[-k:]
