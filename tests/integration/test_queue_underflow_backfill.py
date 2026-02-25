import json
import subprocess
import sys
from pathlib import Path


def test_queue_underflow_backfill_counters_increment(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "queue_underflow"

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
            "8",
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
            "50",
            "--use-queue",
            "--queue-prime-samples",
            "10",
            "--queue-warmup-mode",
            "random",
            "--queue-push-batch",
            "8",
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    totals = payload["queue_underflow_totals"]
    assert totals["missing_labels"] > 0.0
    assert totals["backfilled_samples"] > 0.0

