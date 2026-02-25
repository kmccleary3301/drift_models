from __future__ import annotations

import json
from pathlib import Path

import torch

from drifting_models.data.providers import RealBatchProvider, RealBatchProviderConfig


def _write_shard(path: Path, *, start: int, count: int) -> None:
    images = torch.randn(count, 4, 32, 32)
    labels = torch.arange(start, start + count) % 10
    torch.save({"images": images, "labels": labels}, path)


def test_tensor_shards_shuffle_does_not_reload_shards_excessively(monkeypatch, tmp_path: Path) -> None:
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    _write_shard(shards_dir / "shard_00000000.pt", start=0, count=10)
    _write_shard(shards_dir / "shard_00000001.pt", start=10, count=10)
    _write_shard(shards_dir / "shard_00000002.pt", start=20, count=10)

    manifest = {
        "kind": "sd_vae_latent_shards",
        "imagefolder_root": "dummy",
        "image_size": 256,
        "latent_sampling": "mean",
        "sd_vae": {},
        "num_classes": 10,
        "shard_size": 10,
        "shards": [
            {"path": "shard_00000000.pt", "sha256": "na", "count": 10, "images_shape": [10, 4, 32, 32], "images_dtype": "torch.float16"},
            {"path": "shard_00000001.pt", "sha256": "na", "count": 10, "images_shape": [10, 4, 32, 32], "images_dtype": "torch.float16"},
            {"path": "shard_00000002.pt", "sha256": "na", "count": 10, "images_shape": [10, 4, 32, 32], "images_dtype": "torch.float16"},
        ],
    }
    manifest_path = shards_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    load_calls = {"count": 0}
    import drifting_models.data.providers as providers_mod

    real_torch_load = providers_mod.torch.load

    def counted_load(*args, **kwargs):
        load_calls["count"] += 1
        return real_torch_load(*args, **kwargs)

    monkeypatch.setattr(providers_mod.torch, "load", counted_load)

    provider = RealBatchProvider(
        RealBatchProviderConfig(
            source="tensor_shards",
            tensor_shards_manifest_path=str(manifest_path),
            batch_size=4,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            seed=123,
            channels=4,
            image_size=32,
            num_classes=10,
        )
    )

    seen = 0
    for _ in range(8):  # 8 batches * 4 = 32 samples (covers full dataset + wrap)
        x, y = provider.next_batch(device=torch.device("cpu"))
        assert x.shape[0] == y.shape[0]
        seen += int(x.shape[0])

    # With a shard-aware sampler and single-shard caching, we expect at most ~1 load per shard per epoch.
    # Allow a small constant slack for wrap-around behavior.
    assert seen >= 30
    assert load_calls["count"] <= 8

