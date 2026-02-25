import subprocess
import sys
from pathlib import Path


def test_train_latent_overfit_gate_passes(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "overfit_gate_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--device",
            "cpu",
            "--steps",
            "60",
            "--log-every",
            "10",
            "--groups",
            "2",
            "--negatives-per-group",
            "2",
            "--positives-per-group",
            "2",
            "--num-classes",
            "1000",
            "--image-size",
            "16",
            "--channels",
            "4",
            "--patch-size",
            "4",
            "--hidden-dim",
            "128",
            "--depth",
            "3",
            "--num-heads",
            "8",
            "--mlp-ratio",
            "4",
            "--register-tokens",
            "8",
            "--style-vocab-size",
            "1",
            "--style-token-count",
            "0",
            "--alpha-fixed",
            "1.0",
            "--temperature",
            "0.1",
            "--learning-rate",
            "3e-4",
            "--precision",
            "fp32",
            "--scheduler",
            "none",
            "--overfit-fixed-batch",
            "--output-dir",
            str(output_dir),
            "--checkpoint-path",
            str(checkpoint_path),
            "--save-every",
            "20",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    summary_path = output_dir / "latent_summary.json"
    assert summary_path.exists()
    assert checkpoint_path.exists()

    overfit_check_path = output_dir / "overfit_check.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/check_latent_overfit.py",
            "--summary-path",
            str(summary_path),
            "--output-path",
            str(overfit_check_path),
            "--loss-threshold",
            "1e-3",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert overfit_check_path.exists()

