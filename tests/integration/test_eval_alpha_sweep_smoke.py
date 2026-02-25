import json
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image


def _save_random_rgb(path: Path) -> None:
    data = torch.rand(32, 32, 3).mul(255).to(torch.uint8).numpy()
    Image.fromarray(data, mode="RGB").save(path)


def test_eval_alpha_sweep_pixel_smoke(tmp_path: Path) -> None:
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

    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = train_dir / "checkpoint.pt"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_pixel.py",
            "--device",
            "cpu",
            "--config",
            str(config_path),
            "--steps",
            "2",
            "--log-every",
            "1",
            "--checkpoint-path",
            str(checkpoint_path),
            "--save-every",
            "1",
            "--output-dir",
            str(train_dir),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert checkpoint_path.exists()

    output_root = tmp_path / "sweep"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_alpha_sweep.py",
            "--output-root",
            str(output_root),
            "--device",
            "cpu",
            "--mode",
            "pixel",
            "--checkpoint-path",
            str(checkpoint_path),
            "--config",
            str(config_path),
            "--alphas",
            "1.0",
            "2.0",
            "--n-samples",
            "8",
            "--batch-size",
            "4",
            "--reference-imagefolder-root",
            str(reference_root),
            "--inception-weights",
            "none",
            "--reference-cache",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert (output_root / "alpha_sweep_summary.json").exists()
    assert (output_root / "alpha_sweep_summary.md").exists()
    assert (output_root / "RUN.md").exists()
    assert (output_root / "reference_stats.pt").exists()
    assert isinstance(payload["results"], list)
    assert len(payload["results"]) == 2
    for entry in payload["results"]:
        assert entry["paths"]["eval_summary_path"]
