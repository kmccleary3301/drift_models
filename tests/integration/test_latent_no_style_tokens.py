import json
import subprocess
import sys
from pathlib import Path

import torch


def test_train_and_sample_latent_with_no_style_tokens(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    train_out = tmp_path / "train_out"
    sample_out = tmp_path / "sample_out"
    checkpoint_path = train_out / "checkpoint.pt"

    train = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--device",
            "cpu",
            "--output-dir",
            str(train_out),
            "--checkpoint-path",
            str(checkpoint_path),
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
            "--style-vocab-size",
            "1",
            "--style-token-count",
            "0",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    train_payload = json.loads(train.stdout)
    assert train_payload["model_config"]["style_token_count"] == 0
    assert train_payload["model_config"]["style_vocab_size"] == 1
    assert checkpoint_path.exists()

    sample = subprocess.run(
        [
            sys.executable,
            "scripts/sample_latent.py",
            "--device",
            "cpu",
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-root",
            str(sample_out),
            "--n-samples",
            "4",
            "--batch-size",
            "2",
            "--alpha",
            "1.5",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    sample_payload = json.loads(sample.stdout)
    assert sample_payload["model_config"]["style_token_count"] == 0
    assert sample_payload["model_config"]["style_vocab_size"] == 1

    tensor_payload = torch.load(sample_out / "latents.pt", map_location="cpu")
    assert tensor_payload["latents"].shape[0] == 4
    assert tensor_payload["labels"].shape[0] == 4
