import json
import subprocess
import sys
from pathlib import Path


def test_latent_can_load_exported_mae_encoder(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
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
            "--image-size",
            "16",
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
    assert export_path.exists()

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
    payload = json.loads(result.stdout)
    assert payload["train_config"]["feature_encoder"] == "mae"
    assert payload["train_config"]["mae_encoder_path"] == str(export_path)
    assert len(payload["logs"]) >= 1


def test_latent_can_load_legacy_exported_mae_encoder(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    export_path = tmp_path / "mae_encoder_legacy.pt"

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
            "--image-size",
            "16",
            "--in-channels",
            "4",
            "--base-channels",
            "8",
            "--stages",
            "2",
            "--encoder-arch",
            "legacy_conv",
            "--export-encoder-path",
            str(export_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert export_path.exists()

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
            "--mae-encoder-arch",
            "legacy_conv",
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
    payload = json.loads(result.stdout)
    assert payload["train_config"]["mae_encoder_arch"] == "legacy_conv"
    assert payload["train_config"]["mae_encoder_path"] == str(export_path)
    assert len(payload["logs"]) >= 1


def test_latent_can_load_paper_arch_exported_mae_encoder(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    export_path = tmp_path / "mae_encoder_paper.pt"

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
            "--image-size",
            "16",
            "--in-channels",
            "4",
            "--base-channels",
            "4",
            "--stages",
            "4",
            "--encoder-arch",
            "paper_resnet34_unet",
            "--export-encoder-path",
            str(export_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert export_path.exists()

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
            "--mae-encoder-arch",
            "paper_resnet34_unet",
            "--feature-base-channels",
            "4",
            "--feature-stages",
            "4",
            "--feature-selected-stages",
            "0",
            "1",
            "2",
            "3",
            "--mae-encoder-path",
            str(export_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["train_config"]["mae_encoder_arch"] == "paper_resnet34_unet"
    assert payload["train_config"]["mae_encoder_path"] == str(export_path)
    assert len(payload["logs"]) >= 1
