import json
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image


def _save_random_rgb(path: Path) -> None:
    data = torch.rand(64, 64, 3).mul(255).to(torch.uint8).numpy()
    Image.fromarray(data, mode="RGB").save(path)


def test_eval_max_reference_samples_imagefolder(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ref_root = tmp_path / "ref"
    gen_root = tmp_path / "gen"
    for root in (ref_root, gen_root):
        (root / "0").mkdir(parents=True, exist_ok=True)
        (root / "1").mkdir(parents=True, exist_ok=True)
        for idx in range(10):
            _save_random_rgb(root / "0" / f"{idx:06d}.png")
            _save_random_rgb(root / "1" / f"{idx:06d}.png")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_fid_is.py",
            "--device",
            "cpu",
            "--batch-size",
            "8",
            "--inception-weights",
            "none",
            "--reference-source",
            "imagefolder",
            "--reference-imagefolder-root",
            str(ref_root),
            "--generated-source",
            "imagefolder",
            "--generated-imagefolder-root",
            str(gen_root),
            "--max-reference-samples",
            "5",
            "--max-generated-samples",
            "7",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["reference_samples"] == 5
    assert payload["generated_samples"] == 7

