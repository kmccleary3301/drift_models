from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler


class SyntheticClassImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        *,
        size: int,
        channels: int,
        image_size: int,
        num_classes: int,
        seed: int,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be > 0")
        if channels <= 0:
            raise ValueError("channels must be > 0")
        if image_size <= 0:
            raise ValueError("image_size must be > 0")
        if num_classes <= 0:
            raise ValueError("num_classes must be > 0")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.images = torch.randn(size, channels, image_size, image_size, generator=generator)
        self.labels = torch.randint(0, num_classes, (size,), generator=generator, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


@dataclass(frozen=True)
class RealBatchProviderConfig:
    source: str = "synthetic_dataset"
    dataset_size: int = 4096
    batch_size: int = 128
    shuffle: bool = True
    drop_last: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    seed: int = 1337
    channels: int = 4
    image_size: int = 32
    num_classes: int = 1000
    imagefolder_root: str | None = None
    webdataset_urls: str | None = None
    webdataset_image_key: str = "jpg;png;jpeg;webp"
    webdataset_label_key: str = "cls"
    tensor_file_path: str | None = None
    tensor_shards_manifest_path: str | None = None
    transform_resize: int | None = None
    transform_center_crop: bool = True
    transform_horizontal_flip: bool = False
    transform_normalize: bool = False
    distributed_world_size: int = 1
    distributed_rank: int = 0


class RealBatchProvider:
    def __init__(self, config: RealBatchProviderConfig) -> None:
        _validate_provider_config(config)
        dataset = _build_dataset(config)
        loader_generator = torch.Generator(device="cpu")
        loader_generator.manual_seed(config.seed + 17)
        self._distributed_sampler = _build_distributed_sampler(config=config, dataset=dataset)
        self._tensor_shards_sampler = _build_tensor_shards_sampler(config=config, dataset=dataset)
        self._epoch = 0
        sampler = self._distributed_sampler or self._tensor_shards_sampler
        self._loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle if sampler is None else False,
            sampler=sampler,
            drop_last=config.drop_last,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
            generator=loader_generator,
        )
        self._iter = iter(self._loader)
        self.config = config
        self.manifest_fingerprint = dataset_manifest_fingerprint(config)

    def next_batch(self, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            images, labels = next(self._iter)
        except StopIteration:
            self._epoch += 1
            if self._distributed_sampler is not None:
                self._distributed_sampler.set_epoch(self._epoch)
            if self._tensor_shards_sampler is not None:
                self._tensor_shards_sampler.set_epoch(self._epoch)
            self._iter = iter(self._loader)
            images, labels = next(self._iter)
        return images.to(device), labels.to(device).long()


def dataset_manifest_fingerprint(config: RealBatchProviderConfig) -> str:
    payload: dict[str, Any] = {
        "source": config.source,
        "seed": config.seed,
        "channels": config.channels,
        "image_size": config.image_size,
        "num_classes": config.num_classes,
    }
    if config.source == "synthetic_dataset":
        payload["dataset_size"] = config.dataset_size
    elif config.source == "imagefolder":
        root = Path(_required_path(config.imagefolder_root, "imagefolder_root"))
        payload["root"] = str(root.resolve())
        # Full per-file SHA256 manifests don't scale to ImageNet. Use an auto mode:
        # - small trees: include per-file sha256 (strong mismatch guard)
        # - large trees: include a streaming digest over path/size/mtime (fast, scalable)
        payload["imagefolder_manifest"] = _collect_tree_manifest_auto(root)
    elif config.source == "tensor_file":
        tensor_path = Path(_required_path(config.tensor_file_path, "tensor_file_path"))
        payload["tensor_file"] = {
            "path": str(tensor_path.resolve()),
            "size": tensor_path.stat().st_size,
            "mtime_ns": tensor_path.stat().st_mtime_ns,
        }
    elif config.source == "tensor_shards":
        manifest_path = Path(_required_path(config.tensor_shards_manifest_path, "tensor_shards_manifest_path"))
        manifest = manifest_path.read_bytes()
        payload["tensor_shards_manifest"] = {
            "path": str(manifest_path.resolve()),
            "size": manifest_path.stat().st_size,
            "mtime_ns": manifest_path.stat().st_mtime_ns,
            "sha256": sha256(manifest).hexdigest(),
        }
    elif config.source == "webdataset":
        payload["urls"] = _required_path(config.webdataset_urls, "webdataset_urls")
    digest = sha256(repr(sorted(payload.items())).encode("utf-8")).hexdigest()
    return digest


def _validate_provider_config(config: RealBatchProviderConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if config.dataset_size <= 0 and config.source in {"synthetic_dataset", "webdataset"}:
        raise ValueError("dataset_size must be > 0 for synthetic_dataset and webdataset")
    if config.channels <= 0:
        raise ValueError("channels must be > 0")
    if config.image_size <= 0:
        raise ValueError("image_size must be > 0")
    if config.num_classes <= 0:
        raise ValueError("num_classes must be > 0")
    if config.distributed_world_size <= 0:
        raise ValueError("distributed_world_size must be > 0")
    if config.distributed_rank < 0 or config.distributed_rank >= config.distributed_world_size:
        raise ValueError("distributed_rank must be in [0, distributed_world_size)")
    if config.source == "imagefolder":
        _required_path(config.imagefolder_root, "imagefolder_root")
    elif config.source == "tensor_file":
        _required_path(config.tensor_file_path, "tensor_file_path")
        if config.num_workers != 0:
            raise ValueError("tensor_file source requires num_workers=0 (workers would duplicate the loaded tensor)")
    elif config.source == "tensor_shards":
        _required_path(config.tensor_shards_manifest_path, "tensor_shards_manifest_path")
        if config.num_workers != 0:
            raise ValueError(
                "tensor_shards source requires num_workers=0 (each worker would have its own shard cache, causing thrash)"
            )
    elif config.source == "webdataset":
        _required_path(config.webdataset_urls, "webdataset_urls")
        if config.channels not in {1, 3}:
            raise ValueError("webdataset source currently supports channels in {1, 3}")
    elif config.source != "synthetic_dataset":
        raise ValueError(f"Unsupported source: {config.source}")
    if config.prefetch_factor is not None and config.prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be > 0 when set")
    if config.num_workers == 0:
        if config.persistent_workers:
            raise ValueError("persistent_workers requires num_workers > 0")
        if config.prefetch_factor is not None:
            raise ValueError("prefetch_factor requires num_workers > 0")


def _build_dataset(config: RealBatchProviderConfig) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
    if config.source == "synthetic_dataset":
        return SyntheticClassImageDataset(
            size=config.dataset_size,
            channels=config.channels,
            image_size=config.image_size,
            num_classes=config.num_classes,
            seed=config.seed,
        )
    if config.source == "imagefolder":
        return _build_imagefolder_dataset(config)
    if config.source == "tensor_file":
        return _build_tensor_file_dataset(config)
    if config.source == "tensor_shards":
        return _build_tensor_shards_dataset(config)
    if config.source == "webdataset":
        return _build_webdataset(config)
    raise ValueError(f"Unsupported source: {config.source}")


def _build_imagefolder_dataset(config: RealBatchProviderConfig) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
    try:
        from torchvision import datasets
    except ModuleNotFoundError as error:
        raise RuntimeError("torchvision is required for imagefolder source") from error
    transform = _build_image_transform(config)
    root = _required_path(config.imagefolder_root, "imagefolder_root")
    return datasets.ImageFolder(root=root, transform=transform)


def _build_tensor_file_dataset(config: RealBatchProviderConfig) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
    tensor_path = Path(_required_path(config.tensor_file_path, "tensor_file_path"))
    payload = torch.load(tensor_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("tensor_file payload must be a dict with 'images' and 'labels'")
    if "images" not in payload or "labels" not in payload:
        raise ValueError("tensor_file payload must contain 'images' and 'labels'")
    images = payload["images"]
    labels = payload["labels"]
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("images must be a [N, C, H, W] tensor")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError("labels must be a [N] tensor")
    if images.shape[0] != labels.shape[0]:
        raise ValueError("images and labels must have the same first dimension")
    return TensorDataset(images.float(), labels.long())


class ShardedTensorFileDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, *, manifest_path: Path) -> None:
        if not manifest_path.exists():
            raise FileNotFoundError(str(manifest_path))
        payload = _load_json(manifest_path)
        shards = list(payload.get("shards", []))
        if not shards:
            raise ValueError("manifest contains no shards")
        self._root = manifest_path.parent
        self._shards: list[dict[str, Any]] = shards

        offsets: list[int] = []
        total = 0
        for shard in shards:
            count = int(shard["count"])
            if count <= 0:
                raise ValueError("shard count must be > 0")
            offsets.append(total)
            total += count
        self._offsets = offsets
        self._total = total

        self._cached_index: int | None = None
        self._cached_images: torch.Tensor | None = None
        self._cached_labels: torch.Tensor | None = None

    def shard_offsets(self) -> list[int]:
        return list(self._offsets)

    def shard_counts(self) -> list[int]:
        return [int(s["count"]) for s in self._shards]

    def __len__(self) -> int:
        return int(self._total)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self._total:
            raise IndexError(index)
        shard_index, local_index = _locate_shard(self._offsets, self._shards, index)
        images, labels = self._load_shard(shard_index)
        return images[local_index], labels[local_index]

    def _load_shard(self, shard_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cached_index == shard_index and self._cached_images is not None and self._cached_labels is not None:
            return self._cached_images, self._cached_labels
        shard = self._shards[shard_index]
        shard_path = self._root / str(shard["path"])
        payload = torch.load(shard_path, map_location="cpu")
        if not isinstance(payload, dict) or "images" not in payload or "labels" not in payload:
            raise ValueError(f"Invalid shard payload: {shard_path}")
        images = payload["images"]
        labels = payload["labels"]
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("shard images must be [N, C, H, W]")
        if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
            raise ValueError("shard labels must be [N]")
        if images.shape[0] != labels.shape[0]:
            raise ValueError("shard images/labels count mismatch")

        self._cached_index = shard_index
        self._cached_images = images.float()
        self._cached_labels = labels.long()
        return self._cached_images, self._cached_labels


def _build_tensor_shards_dataset(config: RealBatchProviderConfig) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
    manifest_path = Path(_required_path(config.tensor_shards_manifest_path, "tensor_shards_manifest_path"))
    return ShardedTensorFileDataset(manifest_path=manifest_path)


class _ShardShuffleSampler(Sampler[int]):
    def __init__(self, *, dataset: ShardedTensorFileDataset, seed: int) -> None:
        self._dataset = dataset
        self._seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._seed + 1009 * self._epoch)
        offsets = self._dataset.shard_offsets()
        counts = self._dataset.shard_counts()
        shard_count = len(offsets)
        shard_order = torch.randperm(shard_count, generator=generator)
        for shard_index_t in shard_order:
            shard_index = int(shard_index_t.item())
            base = offsets[shard_index]
            count = int(counts[shard_index])
            perm = torch.randperm(count, generator=generator)
            for local_index_t in perm:
                yield base + int(local_index_t.item())

    def __len__(self) -> int:
        return len(self._dataset)


def _build_tensor_shards_sampler(
    *, config: RealBatchProviderConfig, dataset: Dataset[tuple[torch.Tensor, torch.Tensor]]
) -> _ShardShuffleSampler | None:
    # Default DataLoader shuffling uses random indices. For a sharded dataset that caches only one shard at a time,
    # random access causes pathological shard reload thrash. Use a shard-aware sampler that shuffles within each shard
    # while preserving locality.
    if config.source != "tensor_shards" or not config.shuffle:
        return None
    if not isinstance(dataset, ShardedTensorFileDataset):
        return None
    if config.distributed_world_size != 1:
        # Distributed shuffling is handled via DistributedSampler.
        return None
    return _ShardShuffleSampler(dataset=dataset, seed=int(config.seed + 31))


def _locate_shard(offsets: list[int], shards: list[dict[str, Any]], index: int) -> tuple[int, int]:
    # Linear scan is fine for small shard counts (O(100s)), and avoids bringing in bisect edge cases.
    for shard_index in range(len(shards) - 1, -1, -1):
        start = offsets[shard_index]
        if index >= start:
            local = index - start
            if local >= int(shards[shard_index]["count"]):
                raise IndexError(index)
            return shard_index, int(local)
    raise IndexError(index)


def _load_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _build_webdataset(config: RealBatchProviderConfig) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
    try:
        import webdataset as wds
    except ModuleNotFoundError as error:
        raise RuntimeError("webdataset package is required for webdataset source") from error

    transform = _build_image_transform(config)
    urls = _required_path(config.webdataset_urls, "webdataset_urls")
    dataset = (
        wds.WebDataset(urls, shardshuffle=config.shuffle)
        .decode("pil")
        .to_tuple(config.webdataset_image_key, config.webdataset_label_key)
        .map_tuple(transform, _coerce_webdataset_label)
        .with_length(config.dataset_size)
    )
    return dataset  # type: ignore[return-value]


def _build_image_transform(config: RealBatchProviderConfig):
    try:
        from torchvision import transforms
    except ModuleNotFoundError as error:
        raise RuntimeError("torchvision is required for non-synthetic sources") from error

    operations: list[Any] = []
    if config.transform_resize is not None:
        operations.append(transforms.Resize((config.transform_resize, config.transform_resize), antialias=True))
    if config.transform_center_crop:
        operations.append(transforms.CenterCrop(config.image_size))
    if config.transform_horizontal_flip:
        operations.append(transforms.RandomHorizontalFlip(p=0.5))
    if config.channels == 3:
        operations.append(transforms.Lambda(lambda image: image.convert("RGB")))
    elif config.channels == 1:
        operations.append(transforms.Lambda(lambda image: image.convert("L")))
    operations.append(transforms.ToTensor())
    if config.transform_normalize:
        operations.append(transforms.Normalize(mean=[0.5] * config.channels, std=[0.5] * config.channels))
    return transforms.Compose(operations)


def _coerce_webdataset_label(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, bytes):
        return int(value.decode("utf-8"))
    if isinstance(value, str):
        return int(value)
    if isinstance(value, torch.Tensor):
        return int(value.item())
    raise ValueError(f"Unsupported webdataset label type: {type(value).__name__}")


def _required_path(value: str | None, field_name: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{field_name} must be provided")
    return value


def _collect_tree_manifest(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        files.append(
            {
                "relative_path": str(path.relative_to(root)),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": _file_sha256(path),
            }
        )
    return files


def _collect_tree_manifest_auto(root: Path) -> dict[str, Any]:
    # Threshold chosen to keep CIFAR-scale strong fingerprints while making ImageNet practical.
    max_full_files = 50_000
    import os

    # Pass 1: always compute a fast digest + counts without reading file contents.
    digest = sha256()
    total_bytes = 0
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            path = Path(dirpath) / filename
            stat = path.stat()
            rel = str(path.relative_to(root)).encode("utf-8", errors="strict")
            digest.update(rel)
            digest.update(b"\0")
            digest.update(str(int(stat.st_size)).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))
            digest.update(b"\n")
            total_bytes += int(stat.st_size)
            count += 1

    if count > max_full_files:
        return {
            "mode": "fast",
            "file_count": int(count),
            "total_bytes": int(total_bytes),
            "digest_sha256": digest.hexdigest(),
        }

    # Pass 2 (small trees only): compute per-file sha256 for a stronger mismatch guard.
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            path = Path(dirpath) / filename
            stat = path.stat()
            files.append(
                {
                    "relative_path": str(path.relative_to(root)),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "sha256": _file_sha256(path),
                }
            )
    return {
        "mode": "full",
        "file_count": int(count),
        "total_bytes": int(total_bytes),
        "files_digest_sha256": digest.hexdigest(),
        "files": files,
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _build_distributed_sampler(
    *,
    config: RealBatchProviderConfig,
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
) -> Sampler[int] | None:
    if config.distributed_world_size <= 1:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=config.distributed_world_size,
        rank=config.distributed_rank,
        shuffle=config.shuffle,
        drop_last=config.drop_last,
        seed=config.seed,
    )
