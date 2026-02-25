from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class VerificationResult:
    manifest_path: str
    shards_dir: str
    mode: str
    checked_shards: int
    total_shards: int
    total_items_manifest: int
    total_items_loaded: int
    image_shape_item: list[int] | None
    images_dtype: str | None
    label_min: int | None
    label_max: int | None
    num_classes_manifest: int | None
    disk_free_gb: float
    ok: bool
    errors: list[str]


def main() -> None:
    args = _parse_args()
    manifest_path = Path(args.manifest_path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(str(manifest_path))
    shards_dir = manifest_path.parent

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = list(payload.get("shards", []))
    if not shards:
        raise ValueError("manifest contains no shards")

    mode = str(args.mode)
    shard_indices = _select_shards(
        total=len(shards),
        mode=mode,
        max_shards=int(args.max_shards),
        seed=int(args.seed),
    )

    errors: list[str] = []
    total_items_loaded = 0
    image_shape_item: list[int] | None = None
    images_dtype: str | None = None
    label_min: int | None = None
    label_max: int | None = None

    expected_num_classes = payload.get("num_classes")
    num_classes_manifest = int(expected_num_classes) if isinstance(expected_num_classes, int) else None

    for shard_index in shard_indices:
        record = shards[shard_index]
        rel_path = record.get("path")
        if not isinstance(rel_path, str) or not rel_path.strip():
            errors.append(f"shard[{shard_index}]: missing/invalid path")
            continue
        shard_path = (shards_dir / rel_path).resolve()
        if not shard_path.exists():
            errors.append(f"shard[{shard_index}]: missing file: {shard_path}")
            continue
        if args.check_sha256:
            expected_sha = record.get("sha256")
            if isinstance(expected_sha, str) and len(expected_sha) == 64:
                actual_sha = _sha256_file(shard_path)
                if actual_sha != expected_sha:
                    errors.append(f"shard[{shard_index}]: sha256 mismatch expected={expected_sha} got={actual_sha}")

        expected_count = record.get("count")
        expected_count_i = int(expected_count) if isinstance(expected_count, int) else None

        try:
            shard_payload = torch.load(shard_path, map_location="cpu")
        except Exception as exc:
            errors.append(f"shard[{shard_index}]: torch.load failed: {exc}")
            continue
        if not isinstance(shard_payload, dict) or "images" not in shard_payload or "labels" not in shard_payload:
            errors.append(f"shard[{shard_index}]: payload must contain 'images' and 'labels'")
            continue

        images = shard_payload["images"]
        labels = shard_payload["labels"]
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            errors.append(f"shard[{shard_index}]: images must be [N,C,H,W] tensor")
            continue
        if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
            errors.append(f"shard[{shard_index}]: labels must be [N] tensor")
            continue
        if images.shape[0] != labels.shape[0]:
            errors.append(f"shard[{shard_index}]: N mismatch images={images.shape[0]} labels={labels.shape[0]}")
            continue
        if expected_count_i is not None and int(images.shape[0]) != expected_count_i:
            errors.append(f"shard[{shard_index}]: count mismatch expected={expected_count_i} got={int(images.shape[0])}")
            continue

        total_items_loaded += int(images.shape[0])
        item_shape = list(images.shape[1:])
        if image_shape_item is None:
            image_shape_item = item_shape
        elif image_shape_item != item_shape:
            errors.append(f"shard[{shard_index}]: item shape mismatch expected={image_shape_item} got={item_shape}")

        dtype_str = str(images.dtype)
        if images_dtype is None:
            images_dtype = dtype_str
        elif images_dtype != dtype_str:
            errors.append(f"shard[{shard_index}]: dtype mismatch expected={images_dtype} got={dtype_str}")

        if labels.numel() > 0:
            shard_min = int(labels.min().item())
            shard_max = int(labels.max().item())
            label_min = shard_min if label_min is None else min(label_min, shard_min)
            label_max = shard_max if label_max is None else max(label_max, shard_max)
            if num_classes_manifest is not None:
                if shard_min < 0 or shard_max >= num_classes_manifest:
                    errors.append(
                        f"shard[{shard_index}]: labels out of range [0,{num_classes_manifest-1}] "
                        f"min={shard_min} max={shard_max}"
                    )

    total_items_manifest = int(sum(int(s.get("count", 0)) for s in shards if isinstance(s.get("count"), int)))
    free_gb = float(shutil.disk_usage(args.drive_mount).free) / (1024.0**3)

    result = VerificationResult(
        manifest_path=str(manifest_path),
        shards_dir=str(shards_dir),
        mode=mode,
        checked_shards=len(shard_indices),
        total_shards=len(shards),
        total_items_manifest=total_items_manifest,
        total_items_loaded=int(total_items_loaded),
        image_shape_item=image_shape_item,
        images_dtype=images_dtype,
        label_min=label_min,
        label_max=label_max,
        num_classes_manifest=num_classes_manifest,
        disk_free_gb=float(free_gb),
        ok=len(errors) == 0,
        errors=errors,
    )

    output = json.dumps(result.__dict__, indent=2)
    if args.output_path is not None:
        out_path = Path(args.output_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")

    print(output)
    if not result.ok:
        raise SystemExit(2)


def _select_shards(*, total: int, mode: str, max_shards: int, seed: int) -> list[int]:
    if total <= 0:
        return []
    if mode not in {"quick", "full"}:
        raise ValueError("--mode must be one of {quick, full}")
    if mode == "full":
        return list(range(total))
    # quick: first, last, and a few random in-between.
    indices = {0, total - 1}
    if total > 2:
        rng = random.Random(int(seed))
        population = list(range(1, total - 1))
        rng.shuffle(population)
        for idx in population[: max(0, max_shards - len(indices))]:
            indices.add(idx)
    selected = sorted(indices)
    if max_shards > 0:
        selected = selected[:max_shards]
    return selected


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify a sharded tensor-file manifest (structural + optional full scan).")
    p.add_argument("--manifest-path", type=str, required=True)
    p.add_argument("--mode", choices=("quick", "full"), default="quick")
    p.add_argument("--max-shards", type=int, default=8, help="Quick mode: cap shards loaded (<=0 means no cap).")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--drive-mount", type=str, default="/mnt/drive_4")
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--check-sha256", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    # Make python crash dumps easier to read in logs.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
