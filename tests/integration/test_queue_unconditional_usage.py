import json
import subprocess
import sys
from pathlib import Path


def test_queue_unconditional_usage_pixel(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "pixel"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_pixel.py",
            "--device",
            "cpu",
            "--steps",
            "2",
            "--log-every",
            "1",
            "--groups",
            "2",
            "--negatives-per-group",
            "2",
            "--positives-per-group",
            "2",
            "--unconditional-per-group",
            "2",
            "--alpha-fixed",
            "3.0",
            "--num-classes",
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
            "--use-queue",
            "--queue-prime-samples",
            "32",
            "--queue-push-batch",
            "16",
            "--queue-per-class-capacity",
            "32",
            "--queue-global-capacity",
            "64",
            "--real-batch-source",
            "synthetic_dataset",
            "--real-dataset-size",
            "256",
            "--real-loader-batch-size",
            "64",
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    logs = payload.get("logs", [])
    assert logs, "expected non-empty logs"
    last = logs[-1]
    assert last["mean_unconditional_weight"] > 0.0
    assert last["mean_unconditional_negative_fraction"] > 0.0


def test_queue_unconditional_usage_latent(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "latent"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--device",
            "cpu",
            "--steps",
            "2",
            "--log-every",
            "1",
            "--groups",
            "2",
            "--negatives-per-group",
            "2",
            "--positives-per-group",
            "2",
            "--unconditional-per-group",
            "2",
            "--alpha-fixed",
            "3.0",
            "--num-classes",
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
            "--use-queue",
            "--queue-prime-samples",
            "32",
            "--queue-push-batch",
            "16",
            "--queue-per-class-capacity",
            "32",
            "--queue-global-capacity",
            "64",
            "--real-batch-source",
            "synthetic_dataset",
            "--real-dataset-size",
            "256",
            "--real-loader-batch-size",
            "64",
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    logs = payload.get("logs", [])
    assert logs, "expected non-empty logs"
    last = logs[-1]
    assert last["mean_unconditional_weight"] > 0.0
    assert last["mean_unconditional_negative_fraction"] > 0.0

