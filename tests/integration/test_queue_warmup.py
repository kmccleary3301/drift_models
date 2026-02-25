import json
import subprocess
import sys
from pathlib import Path


def test_class_balanced_queue_warmup_reports_coverage() -> None:
    repo_root = Path(__file__).resolve().parents[2]
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
            "--queue-prime-samples",
            "20",
            "--queue-warmup-mode",
            "class_balanced",
            "--queue-warmup-min-per-class",
            "2",
            "--queue-push-batch",
            "8",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    report = payload["queue_warmup_report"]
    assert report["warmup_mode"] == "class_balanced"
    assert report["covered_classes"] >= 10.0
