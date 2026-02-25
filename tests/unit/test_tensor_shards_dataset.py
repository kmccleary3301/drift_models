import json
from pathlib import Path

import torch

from drifting_models.data.providers import ShardedTensorFileDataset


def _write_shard(path: Path, *, start: int, count: int) -> dict[str, object]:
    images = torch.arange(start, start + count, dtype=torch.float32).view(count, 1, 1, 1)
    labels = torch.arange(start, start + count, dtype=torch.long)
    torch.save({"images": images, "labels": labels, "meta": {}}, path)
    return {"path": path.name, "count": count}


def test_sharded_tensor_dataset_roundtrip(tmp_path: Path) -> None:
    shard0 = _write_shard(tmp_path / "shard_00000000.pt", start=0, count=3)
    shard1 = _write_shard(tmp_path / "shard_00000001.pt", start=3, count=2)
    manifest = {
        "kind": "sd_vae_latent_shards",
        "shards": [shard0, shard1],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dataset = ShardedTensorFileDataset(manifest_path=manifest_path)
    assert len(dataset) == 5
    image, label = dataset[0]
    assert float(image.item()) == 0.0
    assert int(label.item()) == 0

    image, label = dataset[4]
    assert float(image.item()) == 4.0
    assert int(label.item()) == 4

