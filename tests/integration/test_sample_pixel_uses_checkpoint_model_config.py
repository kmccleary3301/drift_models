import subprocess
import sys
from pathlib import Path


def test_sample_pixel_uses_checkpoint_model_config(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = tmp_path / "pixel_ckpt.pt"
    output_root = tmp_path / "samples"

    # Train a non-default model config so that sampling must read it from checkpoint.
    subprocess.run(
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
            "--num-classes",
            "7",
            "--image-size",
            "16",
            "--channels",
            "3",
            "--patch-size",
            "4",
            "--hidden-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
            "--register-tokens",
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
    assert checkpoint_path.exists()

    # Do not pass any model-shape args; sampler should reconstruct from checkpoint.
    subprocess.run(
        [
            sys.executable,
            "scripts/sample_pixel.py",
            "--device",
            "cpu",
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-root",
            str(output_root),
            "--n-samples",
            "4",
            "--batch-size",
            "2",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (output_root / "sample_summary.json").exists()
    assert (output_root / "images").exists()

