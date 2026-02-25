import json
import subprocess
import sys
from pathlib import Path


def test_latent_grad_accumulation_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[2]
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
            "--grad-accum-steps",
            "2",
            "--precision",
            "fp32",
            "--scheduler",
            "cosine",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["train_config"]["grad_accum_steps"] == 2
    assert payload["train_config"]["precision"] == "fp32"
    assert payload["train_config"]["scheduler"] == "cosine"
    assert "lr" in payload["logs"][-1]
    assert payload["logs"][-1]["loss"] >= 0.0
