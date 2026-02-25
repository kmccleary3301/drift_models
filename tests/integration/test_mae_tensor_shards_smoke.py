import json
import subprocess
import sys
from pathlib import Path

import torch


def _write_shard(path: Path, *, count: int) -> dict[str, object]:
    images = torch.randn(count, 4, 32, 32)
    labels = torch.randint(0, 1000, (count,), dtype=torch.long)
    torch.save({"images": images, "labels": labels, "meta": {}}, path)
    return {"path": path.name, "count": count}


def test_train_mae_on_tensor_shards_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = [
        _write_shard(shard_dir / "shard_00000000.pt", count=8),
        _write_shard(shard_dir / "shard_00000001.pt", count=8),
    ]
    manifest_path = shard_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"kind": "test_shards", "shards": shards}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_mae.py",
            "--device",
            "cpu",
            "--steps",
            "2",
            "--log-every",
            "1",
            "--batch-size",
            "4",
            "--base-channels",
            "8",
            "--stages",
            "2",
            "--mask-ratio",
            "0.5",
            "--real-batch-source",
            "tensor_shards",
            "--real-tensor-shards-manifest-path",
            str(manifest_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["config"]["steps"] == 2
    assert payload["logs"][-1]["loss"] >= 0.0
    assert "real_batch_provider" in payload
    assert payload["real_batch_provider"]["source"] == "tensor_shards"

