import subprocess
import sys
from pathlib import Path


def test_resume_mismatch_guard_blocks_incompatible_config(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = tmp_path / "resume_guard.pt"
    first = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--device",
            "cpu",
            "--steps",
            "2",
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
            "--checkpoint-path",
            str(checkpoint_path),
            "--save-every",
            "1",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0

    second = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--device",
            "cpu",
            "--steps",
            "3",
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
            "128",
            "--depth",
            "2",
            "--num-heads",
            "4",
            "--checkpoint-path",
            str(checkpoint_path),
            "--resume-from",
            str(checkpoint_path),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert second.returncode != 0
    assert "Resume config mismatch detected" in second.stderr or "Resume config mismatch detected" in second.stdout
