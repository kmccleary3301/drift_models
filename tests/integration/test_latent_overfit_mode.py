import json
import subprocess
import sys
from pathlib import Path


def test_train_latent_overfit_fixed_batch_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "overfit_run"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
            "--steps",
            "3",
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
            "--overfit-fixed-batch",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["train_config"]["overfit_fixed_batch"] is True
    assert len(payload["logs"]) == 3
    assert all(float(log["loss"]) == float(log["loss"]) for log in payload["logs"])


def test_train_latent_overfit_fixed_batch_rejects_queue(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "bad_run"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
            "--steps",
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
            "--overfit-fixed-batch",
            "--use-queue",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--overfit-fixed-batch is incompatible with --use-queue" in result.stderr
