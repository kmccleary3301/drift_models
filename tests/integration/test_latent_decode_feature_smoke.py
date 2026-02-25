import json
import subprocess
import sys
from pathlib import Path


def test_latent_feature_decode_conv_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
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
            "--use-feature-loss",
            "--feature-base-channels",
            "8",
            "--feature-stages",
            "2",
            "--latent-feature-decode-mode",
            "conv",
            "--latent-decoder-out-channels",
            "3",
            "--latent-decoder-image-size",
            "16",
            "--latent-decoder-hidden-channels",
            "16",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["train_config"]["latent_feature_decode_mode"] == "conv"
    assert payload["logs"][0]["loss"] >= 0.0
