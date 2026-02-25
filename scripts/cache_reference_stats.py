from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.transforms import v2

from drifting_models.eval import gaussian_statistics
from drifting_models.utils import (
    add_device_argument,
    resolve_device,
    codebase_fingerprint,
    environment_fingerprint,
    environment_snapshot,
    file_sha256,
    payload_sha256,
    write_json,
)


def main() -> None:
    args = _parse_args()
    device = resolve_device(args.device)

    model = _build_inception(device=device, weights_mode=args.inception_weights)
    weights_cache = _resolve_inception_weights_cache(
        weights=Inception_V3_Weights.IMAGENET1K_V1 if args.inception_weights == "pretrained" else None
    )
    dataset = _build_imagefolder_dataset(Path(args.imagefolder_root), image_exts=args.image_exts)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    max_samples = None if args.max_samples is None or int(args.max_samples) <= 0 else int(args.max_samples)
    contract = _evaluation_contract(inception_weights=args.inception_weights)
    contract_sha256 = payload_sha256(contract)
    features = _collect_inception_pool_features(
        model=model,
        loader=loader,
        device=device,
        max_samples=max_samples,
    )
    mean, cov = gaussian_statistics(features)
    stats_payload = {
        "mean": mean.detach().cpu(),
        "cov": cov.detach().cpu(),
        "count": int(features.shape[0]),
        "contract": contract,
        "contract_sha256": contract_sha256,
        "provenance": _reference_source_provenance(
            imagefolder_root=Path(args.imagefolder_root),
            max_samples=max_samples,
            image_exts=args.image_exts,
        ),
    }

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    write_json(out_path.parent / "env_snapshot.json", environment_snapshot(paths=[out_path.parent]))
    write_json(out_path.parent / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))
    write_json(out_path.parent / "env_fingerprint.json", environment_fingerprint())
    torch.save(stats_payload, out_path)

    summary = {
        "output_path": str(out_path),
        "count": int(stats_payload["count"]),
        "feature_dim": int(stats_payload["mean"].shape[0]),
        "imagefolder_root": str(Path(args.imagefolder_root).resolve()),
        "inception_weights": args.inception_weights,
        "inception_weights_cache": weights_cache,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "device": str(device),
        "env_fingerprint": environment_fingerprint(),
        "paths": {
            "env_snapshot_json": str(out_path.parent / "env_snapshot.json"),
            "codebase_fingerprint_json": str(out_path.parent / "codebase_fingerprint.json"),
            "env_fingerprint_json": str(out_path.parent / "env_fingerprint.json"),
        },
    }
    if args.summary_json_path is not None:
        write_json(Path(args.summary_json_path), summary)
    print(json.dumps(summary, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache Inception reference stats (mean/cov) for an ImageFolder dataset.")
    parser.add_argument("--imagefolder-root", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--summary-json-path", type=str, default=None)
    add_device_argument(parser, default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--inception-weights", choices=("pretrained", "none"), default="pretrained")
    parser.add_argument("--image-exts", nargs="*", type=str, default=[])
    return parser.parse_args()


def _evaluation_contract(*, inception_weights: str) -> dict[str, object]:
    return {
        "inception_model": "torchvision.inception_v3",
        "inception_weights": str(inception_weights),
        "preprocess": {
            "resize_hw": [299, 299],
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "input_normalization_policy": "auto_range_to_0_1_then_imagenet_normalize",
    }


def _reference_source_provenance(
    *,
    imagefolder_root: Path,
    max_samples: int | None,
    image_exts: list[str],
) -> dict[str, object]:
    return {
        "source": "imagefolder",
        "imagefolder_root": str(imagefolder_root.resolve()),
        "max_samples": None if max_samples is None else int(max_samples),
        "image_exts": [str(value) for value in image_exts],
    }


def _build_inception(*, device: torch.device, weights_mode: str) -> torch.nn.Module:
    if weights_mode == "none":
        torch.manual_seed(0)
    weights = Inception_V3_Weights.IMAGENET1K_V1 if weights_mode == "pretrained" else None
    aux_logits = True if weights is not None else False
    model = inception_v3(weights=weights, aux_logits=aux_logits, init_weights=False).to(device)
    model.eval()
    return model


def _resolve_inception_weights_cache(*, weights: Inception_V3_Weights | None) -> dict[str, object]:
    if weights is None:
        return {}
    url = getattr(weights, "url", None)
    filename = None
    if isinstance(url, str) and url:
        try:
            filename = url.rsplit("/", 1)[-1]
        except Exception:
            filename = None
    cache_path = None
    cache_sha256 = None
    if filename:
        try:
            checkpoints_dir = Path(torch.hub.get_dir()) / "checkpoints"
            candidate = checkpoints_dir / filename
            if candidate.exists():
                cache_path = str(candidate)
                cache_sha256 = file_sha256(candidate)
        except Exception:
            cache_path = None
            cache_sha256 = None
    return {
        "url": url,
        "filename": filename,
        "cache_path": cache_path,
        "cache_sha256": cache_sha256,
    }


def _build_imagefolder_dataset(root: Path, *, image_exts: list[str]) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
    if not root.exists() or not root.is_dir():
        raise ValueError(f"imagefolder root does not exist or is not a directory: {root}")
    dataset = ImageFolder(
        root=str(root),
        # Resize before collation to support variable-size datasets like ImageNet.
        transform=v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((299, 299), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
    )
    if image_exts:
        _filter_imagefolder_extensions(dataset, image_exts=image_exts)
    if len(dataset) == 0:
        raise ValueError(f"No samples found under: {root}")
    return dataset


def _filter_imagefolder_extensions(dataset: ImageFolder, *, image_exts: list[str]) -> None:
    normalized = []
    for ext in image_exts:
        cleaned = ext.strip().lower()
        if not cleaned:
            continue
        if not cleaned.startswith("."):
            cleaned = "." + cleaned
        normalized.append(cleaned)
    if not normalized:
        return

    keep_samples: list[tuple[str, int]] = []
    keep_targets: list[int] = []
    for path, target in dataset.samples:
        if Path(path).suffix.lower() in normalized:
            keep_samples.append((path, target))
            keep_targets.append(target)
    dataset.samples = keep_samples
    dataset.imgs = keep_samples
    dataset.targets = keep_targets


def _normalize_image_range(images: torch.Tensor) -> torch.Tensor:
    values = images.float()
    low = float(values.min().item())
    high = float(values.max().item())
    if low >= 0.0 and high <= 1.0:
        return values
    if low >= -1.0 and high <= 1.0:
        return (values + 1.0) / 2.0
    if low >= 0.0 and high > 1.0:
        return torch.clamp(values / 255.0, 0.0, 1.0)
    return torch.clamp(values, 0.0, 1.0)


def _collect_inception_pool_features(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int | None,
) -> torch.Tensor:
    features: list[torch.Tensor] = []
    feature_holder: dict[str, torch.Tensor] = {}
    transform = v2.Compose(
        [
            v2.Resize((299, 299), antialias=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def _capture_features(_module: torch.nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        feature_holder["pool"] = torch.flatten(output, start_dim=1).detach()

    hook = model.avgpool.register_forward_hook(_capture_features)
    seen = 0
    with torch.no_grad():
        for images, _labels in loader:
            if max_samples is not None and seen >= max_samples:
                break
            if max_samples is not None:
                remaining = max_samples - seen
                if int(images.shape[0]) > remaining:
                    images = images[:remaining]
            batch = transform(_normalize_image_range(images)).to(device)
            _ = model(batch)
            if "pool" not in feature_holder:
                raise RuntimeError("Failed to capture Inception pool features")
            features.append(feature_holder["pool"].cpu())
            seen += int(images.shape[0])
    hook.remove()
    if not features:
        raise ValueError("No images loaded for reference stats caching")
    return torch.cat(features, dim=0)


if __name__ == "__main__":
    main()
