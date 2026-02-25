import subprocess
import sys
from pathlib import Path

import torch

from drifting_models.models import DiTLikeConfig, DiTLikeGenerator


def _build_checkpoint(path: Path, *, config_hash: str) -> None:
    config = DiTLikeConfig(
        image_size=16,
        in_channels=4,
        out_channels=4,
        patch_size=4,
        hidden_dim=64,
        depth=2,
        num_heads=4,
        register_tokens=4,
        style_vocab_size=1,
        style_token_count=0,
    )
    model = DiTLikeGenerator(config)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "step": 0,
        "extra": {"config_hash": config_hash, "model_config": config.__dict__},
    }
    torch.save(payload, path)


def test_sample_latent_config_hash_mismatch_fails(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = tmp_path / "ckpt.pt"
    config_path = tmp_path / "config.txt"
    config_path.write_text("image-size: 16\nchannels: 4\n", encoding="utf-8")
    _build_checkpoint(checkpoint_path, config_hash="not_the_real_hash")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/sample_latent.py",
            "--device",
            "cpu",
            "--checkpoint-path",
            str(checkpoint_path),
            "--config",
            str(config_path),
            "--output-root",
            str(tmp_path / "out"),
            "--n-samples",
            "2",
            "--batch-size",
            "2",
            "--image-size",
            "16",
            "--channels",
            "4",
            "--patch-size",
            "4",
            "--hidden-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Config hash mismatch" in result.stderr


def test_sample_latent_allow_config_mismatch_flag(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = tmp_path / "ckpt.pt"
    config_path = tmp_path / "config.txt"
    config_path.write_text("image-size: 16\nchannels: 4\n", encoding="utf-8")
    _build_checkpoint(checkpoint_path, config_hash="not_the_real_hash")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/sample_latent.py",
            "--device",
            "cpu",
            "--checkpoint-path",
            str(checkpoint_path),
            "--config",
            str(config_path),
            "--allow-config-mismatch",
            "--output-root",
            str(tmp_path / "out"),
            "--n-samples",
            "2",
            "--batch-size",
            "2",
            "--image-size",
            "16",
            "--channels",
            "4",
            "--patch-size",
            "4",
            "--hidden-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout
