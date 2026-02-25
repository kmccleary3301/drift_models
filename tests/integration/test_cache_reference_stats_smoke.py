import json
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image


def _save_random_rgb(path: Path) -> None:
    data = torch.rand(32, 32, 3).mul(255).to(torch.uint8).numpy()
    Image.fromarray(data, mode="RGB").save(path)


def test_cache_reference_stats_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ref_root = tmp_path / "ref"
    (ref_root / "0").mkdir(parents=True, exist_ok=True)
    (ref_root / "1").mkdir(parents=True, exist_ok=True)
    for idx in range(10):
        _save_random_rgb(ref_root / "0" / f"{idx:06d}.png")
        _save_random_rgb(ref_root / "1" / f"{idx:06d}.png")

    out_path = tmp_path / "ref_stats.pt"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/cache_reference_stats.py",
            "--device",
            "cpu",
            "--batch-size",
            "8",
            "--inception-weights",
            "none",
            "--imagefolder-root",
            str(ref_root),
            "--output-path",
            str(out_path),
            "--max-samples",
            "5",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    assert summary["count"] == 5
    assert out_path.exists()

    payload = torch.load(out_path, map_location="cpu")
    assert "mean" in payload and "cov" in payload and "count" in payload
    assert "contract" in payload and "contract_sha256" in payload and "provenance" in payload
    assert int(payload["count"]) == 5
