import json
import subprocess
import sys
from pathlib import Path


def test_train_pixel_convnext_feature_encoder_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[2]
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
            "1",
            "--positives-per-group",
            "1",
            "--image-size",
            "32",
            "--patch-size",
            "4",
            "--hidden-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
            "--channels",
            "3",
            "--use-feature-loss",
            "--feature-encoder",
            "convnext_tiny",
            "--convnext-weights",
            "none",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["train_config"]["feature_encoder"] == "convnext_tiny"
    assert payload["train_config"]["convnext_weights"] == "none"
    assert payload["logs"][-1]["loss"] >= 0.0


def test_train_pixel_convnext_feature_encoder_requires_rgb() -> None:
    repo_root = Path(__file__).resolve().parents[2]
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
            "1",
            "--positives-per-group",
            "1",
            "--image-size",
            "32",
            "--patch-size",
            "4",
            "--hidden-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
            "--channels",
            "4",
            "--use-feature-loss",
            "--feature-encoder",
            "convnext_tiny",
            "--convnext-weights",
            "none",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requires --channels 3" in result.stderr
