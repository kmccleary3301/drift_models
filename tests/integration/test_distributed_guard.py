import subprocess
import sys
import os
from pathlib import Path


def test_train_latent_refuses_world_size_gt_one(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.update({"WORLD_SIZE": "2"})
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--device",
            "cpu",
            "--steps",
            "1",
            "--log-every",
            "1",
            "--groups",
            "1",
            "--negatives-per-group",
            "2",
            "--positives-per-group",
            "2",
        ],
        cwd=repo_root,
        env={**env, **{"PYTHONUNBUFFERED": "1"}},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "WORLD_SIZE>1" in result.stderr


def test_train_pixel_refuses_world_size_gt_one(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.update({"WORLD_SIZE": "2"})
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_pixel.py",
            "--device",
            "cpu",
            "--steps",
            "1",
            "--log-every",
            "1",
            "--groups",
            "1",
            "--negatives-per-group",
            "2",
            "--positives-per-group",
            "2",
        ],
        cwd=repo_root,
        env={**env, **{"PYTHONUNBUFFERED": "1"}},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "WORLD_SIZE>1" in result.stderr
