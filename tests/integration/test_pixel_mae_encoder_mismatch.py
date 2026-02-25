import subprocess
import sys
from pathlib import Path


def test_pixel_mae_export_config_mismatch_fails(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    export_path = tmp_path / "mae_encoder_mismatch.pt"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_mae.py",
            "--device",
            "cpu",
            "--steps",
            "2",
            "--log-every",
            "1",
            "--batch-size",
            "8",
            "--in-channels",
            "4",
            "--base-channels",
            "8",
            "--stages",
            "2",
            "--export-encoder-path",
            str(export_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_pixel.py",
            "--device",
            "cpu",
            "--steps",
            "1",
            "--use-feature-loss",
            "--feature-encoder",
            "mae",
            "--feature-base-channels",
            "8",
            "--feature-stages",
            "2",
            "--mae-encoder-path",
            str(export_path),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "MAE export config mismatch" in result.stderr
