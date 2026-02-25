import json
import subprocess
import sys
from pathlib import Path

import torch


def test_train_mae_cls_ft_export_and_load(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    export_path = tmp_path / "mae_encoder_clsft.pt"

    result = subprocess.run(
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
            "4",
            "--image-size",
            "16",
            "--in-channels",
            "4",
            "--base-channels",
            "8",
            "--stages",
            "2",
            "--num-classes",
            "10",
            "--cls-ft-steps",
            "2",
            "--cls-ft-log-every",
            "1",
            "--cls-ft-batch-size",
            "4",
            "--export-cls-ft-encoder-path",
            str(export_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["cls_ft"]["steps"] == 2
    assert payload["export_cls_ft_encoder_path"] == str(export_path)

    exported = torch.load(export_path, map_location="cpu")
    assert exported["export_kind"] == "latent_resnet_mae_encoder"
    assert exported["export_version"] == 2
    assert "encoder_state_dict" in exported
    assert exported["encoder_state_dict"]
    assert "cls_ft" in exported

    latent_result = subprocess.run(
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
            "--num-classes",
            "10",
            "--image-size",
            "16",
            "--channels",
            "4",
            "--patch-size",
            "4",
            "--hidden-dim",
            "128",
            "--depth",
            "2",
            "--num-heads",
            "4",
            "--use-feature-loss",
            "--feature-encoder",
            "mae",
            "--feature-base-channels",
            "8",
            "--feature-stages",
            "2",
            "--feature-selected-stages",
            "0",
            "1",
            "--mae-encoder-path",
            str(export_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    latent_payload = json.loads(latent_result.stdout)
    assert latent_payload["train_config"]["mae_encoder_path"] == str(export_path)
