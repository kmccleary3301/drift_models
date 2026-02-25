import subprocess
import sys
from pathlib import Path

import torch


def test_mae_resume_and_export_encoder(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = tmp_path / "mae_ckpt.pt"
    export_path = tmp_path / "mae_encoder.pt"

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
            "--base-channels",
            "8",
            "--stages",
            "2",
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

    subprocess.run(
        [
            sys.executable,
            "scripts/train_mae.py",
            "--device",
            "cpu",
            "--steps",
            "3",
            "--log-every",
            "1",
            "--batch-size",
            "8",
            "--base-channels",
            "8",
            "--stages",
            "2",
            "--checkpoint-path",
            str(checkpoint_path),
            "--resume-from",
            str(checkpoint_path),
            "--export-encoder-path",
            str(export_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = torch.load(export_path, map_location="cpu")
    assert "encoder_state_dict" in payload
    assert payload["encoder_state_dict"]
