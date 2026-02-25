from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import torch


def test_verify_tensor_shards_manifest_smoke(tmp_path: Path) -> None:
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    def write_shard(name: str, n: int) -> None:
        images = torch.randn(n, 4, 32, 32, dtype=torch.float16)
        labels = torch.randint(0, 10, (n,), dtype=torch.long)
        torch.save({"images": images, "labels": labels}, shards_dir / name)

    def sha256_file(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    write_shard("shard_00000000.pt", 7)
    write_shard("shard_00000001.pt", 9)
    sha0 = sha256_file(shards_dir / "shard_00000000.pt")
    sha1 = sha256_file(shards_dir / "shard_00000001.pt")

    manifest = {
        "kind": "sd_vae_latent_shards",
        "num_classes": 10,
        "shards": [
            {"path": "shard_00000000.pt", "count": 7, "sha256": sha0},
            {"path": "shard_00000001.pt", "count": 9, "sha256": sha1},
        ],
    }
    manifest_path = shards_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    out_path = tmp_path / "verify.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_tensor_shards_manifest.py",
            "--manifest-path",
            str(manifest_path),
            "--mode",
            "quick",
            "--check-sha256",
            "--output-path",
            str(out_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["total_shards"] == 2
    assert payload["total_items_manifest"] == 16
