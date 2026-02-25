import json
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image


def _save_random_rgb(path: Path) -> None:
    data = torch.rand(32, 32, 3).mul(255).to(torch.uint8).numpy()
    Image.fromarray(data, mode="RGB").save(path)


def test_eval_image_exts_filters_samples(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ref_root = tmp_path / "ref"
    gen_root = tmp_path / "gen"
    for root in (ref_root, gen_root):
        (root / "0").mkdir(parents=True, exist_ok=True)
        _save_random_rgb(root / "0" / "000000.png")
        _save_random_rgb(root / "0" / "000001.png")
        _save_random_rgb(root / "0" / "000002.jpg")

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
            "--image-exts",
            "png",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["reference_samples"] == 2
    assert payload["generated_samples"] == 2
