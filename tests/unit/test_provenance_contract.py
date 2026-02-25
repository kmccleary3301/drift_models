from __future__ import annotations

import json
from pathlib import Path

import pytest


_SOURCE_PROVENANCE = Path("outputs/datasets/imagenet1k_provenance.json")
_TRAIN_MANIFEST = Path("outputs/datasets/imagenet1k_train_sdvae_latents_shards/manifest.json")
_VAL_MANIFEST = Path("outputs/datasets/imagenet1k_val_sdvae_latents_shards/manifest.json")
_SDVAE_PROVENANCE = Path("outputs/datasets/sdvae_provenance.json")
_INCEPTION_SUMMARY = Path("outputs/datasets/imagenet1k_val_reference_stats_pretrained_summary.json")


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_artifacts_or_skip() -> None:
    required = (
        _SOURCE_PROVENANCE,
        _TRAIN_MANIFEST,
        _VAL_MANIFEST,
        _SDVAE_PROVENANCE,
        _INCEPTION_SUMMARY,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.skip(f"provenance artifacts not present in this environment: {missing}")


def test_source_archive_provenance_contract() -> None:
    _require_artifacts_or_skip()
    payload = _load_json(_SOURCE_PROVENANCE)
    archives = payload.get("archives")
    assert isinstance(archives, list) and archives
    for entry in archives:
        assert isinstance(entry, dict)
        assert entry.get("role")
        assert entry.get("path")
        assert int(entry.get("size_bytes", 0)) > 0
        hashes = entry.get("hashes")
        assert isinstance(hashes, dict)
        assert hashes.get("md5")
        assert hashes.get("sha256")


def test_split_manifest_contract() -> None:
    _require_artifacts_or_skip()
    train_manifest = _load_json(_TRAIN_MANIFEST)
    val_manifest = _load_json(_VAL_MANIFEST)

    assert int(train_manifest.get("exported_samples", 0)) > 0
    train_shards = train_manifest.get("shards")
    assert isinstance(train_shards, list) and train_shards
    assert int(train_shards[0].get("count", 0)) > 0
    assert train_shards[0].get("sha256")

    val_shards = val_manifest.get("shards")
    assert isinstance(val_shards, list) and val_shards
    assert int(val_shards[0].get("count", 0)) > 0
    assert val_shards[0].get("sha256")


def test_sdvae_and_evaluator_provenance_contract() -> None:
    _require_artifacts_or_skip()
    sdvae = _load_json(_SDVAE_PROVENANCE)
    inception = _load_json(_INCEPTION_SUMMARY)

    assert sdvae.get("model_id")
    assert sdvae.get("revision")
    files = sdvae.get("files")
    assert isinstance(files, list) and files
    assert files[0].get("path")
    assert files[0].get("sha256")

    assert inception.get("inception_weights") == "pretrained"
    assert int(inception.get("count", 0)) > 0
    assert int(inception.get("feature_dim", 0)) > 0
