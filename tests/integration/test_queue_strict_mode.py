import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("script_name", ["train_latent.py", "train_pixel.py"])
def test_queue_strict_mode_runs_and_records_flag(script_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            f"scripts/{script_name}",
            "--device",
            "cpu",
            "--steps",
            "1",
            "--log-every",
            "1",
            "--groups",
            "2",
            "--negatives-per-group",
            "2",
            "--positives-per-group",
            "2",
            "--image-size",
            "16",
            "--patch-size",
            "4",
            "--hidden-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
            "--num-classes",
            "10",
            "--use-queue",
            "--unconditional-per-group",
            "2",
            "--queue-prime-samples",
            "20",
            "--queue-push-batch",
            "8",
            "--queue-per-class-capacity",
            "4",
            "--queue-global-capacity",
            "16",
            "--queue-strict-without-replacement",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["train_config"]["queue_strict_without_replacement"] is True


def test_queue_strict_mode_rejects_invalid_capacity() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--device",
            "cpu",
            "--steps",
            "1",
            "--groups",
            "2",
            "--negatives-per-group",
            "2",
            "--positives-per-group",
            "4",
            "--image-size",
            "16",
            "--patch-size",
            "4",
            "--hidden-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
            "--num-classes",
            "10",
            "--use-queue",
            "--unconditional-per-group",
            "2",
            "--queue-per-class-capacity",
            "2",
            "--queue-global-capacity",
            "16",
            "--queue-strict-without-replacement",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--queue-per-class-capacity must be >=" in result.stderr
