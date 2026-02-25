import json
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image


def _save_random_rgb(path: Path) -> None:
    data = torch.rand(32, 32, 3).mul(255).to(torch.uint8).numpy()
    Image.fromarray(data, mode="RGB").save(path)


def test_end_to_end_pixel_eval_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    reference_root = tmp_path / "reference"
    (reference_root / "0").mkdir(parents=True, exist_ok=True)
    (reference_root / "1").mkdir(parents=True, exist_ok=True)
    for idx in range(4):
        _save_random_rgb(reference_root / "0" / f"{idx:06d}.png")
        _save_random_rgb(reference_root / "1" / f"{idx:06d}.png")

    config_path = tmp_path / "train_config.txt"
    config_path.write_text(
        "\n".join(
            [
                "seed: 1337",
                "device: cpu",
                "steps: 2",
                "log-every: 1",
                "groups: 2",
                "negatives-per-group: 2",
                "positives-per-group: 2",
                "num-classes: 2",
                "image-size: 16",
                "channels: 3",
                "patch-size: 4",
                "hidden-dim: 64",
                "depth: 2",
                "num-heads: 4",
                "mlp-ratio: 4.0",
                "register-tokens: 4",
                "temperature: 0.05",
                "learning-rate: 0.0001",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_end_to_end_pixel_eval.py",
            "--output-root",
            str(output_root),
            "--device",
            "cpu",
            "--train-config",
            str(config_path),
            "--train-steps",
            "2",
            "--sample-count",
            "8",
            "--sample-batch-size",
            "4",
            "--reference-imagefolder-root",
            str(reference_root),
            "--inception-weights",
            "none",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert (output_root / "run_summary.json").exists()
    assert (output_root / "RUN.md").exists()
    assert (output_root / "eval" / "eval_summary.json").exists()
    assert payload["paths"]["checkpoint_path"]
