from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.transforms import v2

from drifting_models.eval import frechet_distance, gaussian_statistics, inception_score_from_logits
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
    model, weights_enum = _build_inception(
        device=device,
        weights_mode=args.inception_weights,
    )
    weights_cache = _resolve_inception_weights_cache(
        weights=Inception_V3_Weights.IMAGENET1K_V1 if args.inception_weights == "pretrained" else None
    )
    transform = _build_transform()
    eval_contract = _evaluation_contract(inception_weights=args.inception_weights)
    eval_contract_sha256 = payload_sha256(eval_contract)
    if args.cache_reference_stats is not None and args.load_reference_stats is not None:
        raise ValueError("--cache-reference-stats and --load-reference-stats are mutually exclusive")
    gen_loader = _build_loader(
        kind="generated",
        source=args.generated_source,
        imagefolder_root=args.generated_imagefolder_root,
        tensor_file_path=args.generated_tensor_file_path,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_generated_samples,
        image_exts=args.image_exts,
    )
    reference_stats = None
    loaded_reference_stats_sha256 = None
    if args.load_reference_stats is not None:
        loaded_reference_stats_sha256 = file_sha256(Path(args.load_reference_stats))
        reference_stats = _load_reference_stats(Path(args.load_reference_stats))
        loaded_contract_sha256 = reference_stats.get("contract_sha256")
        if loaded_contract_sha256 is None:
            if not args.allow_reference_contract_mismatch:
                raise ValueError(
                    "Loaded reference stats do not contain a contract hash; "
                    "re-cache stats with current tooling or pass --allow-reference-contract-mismatch."
                )
        elif str(loaded_contract_sha256) != eval_contract_sha256 and not args.allow_reference_contract_mismatch:
            raise ValueError(
                "Reference stats contract mismatch: "
                f"expected {eval_contract_sha256}, loaded {loaded_contract_sha256}. "
                "Use --allow-reference-contract-mismatch to override."
            )
    if reference_stats is None:
        ref_loader = _build_loader(
            kind="reference",
            source=args.reference_source,
            imagefolder_root=args.reference_imagefolder_root,
            tensor_file_path=args.reference_tensor_file_path,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_reference_samples,
            image_exts=args.image_exts,
        )
        ref_features, _ = _collect_inception_outputs(
            model=model,
            loader=ref_loader,
            device=device,
            transform=transform,
        )
        mu_ref, sigma_ref = gaussian_statistics(ref_features)
        reference_stats = {"mean": mu_ref, "cov": sigma_ref, "count": int(ref_features.shape[0])}
        if args.cache_reference_stats is not None:
            _save_reference_stats(
                Path(args.cache_reference_stats),
                reference_stats,
                contract=eval_contract,
                contract_sha256=eval_contract_sha256,
                provenance=_reference_source_provenance(
                    source=args.reference_source,
                    imagefolder_root=args.reference_imagefolder_root,
                    tensor_file_path=args.reference_tensor_file_path,
                    max_samples=args.max_reference_samples,
                    image_exts=args.image_exts,
                ),
            )
    else:
        mu_ref = reference_stats["mean"]
        sigma_ref = reference_stats["cov"]
    gen_features, gen_logits = _collect_inception_outputs(
        model=model,
        loader=gen_loader,
        device=device,
        transform=transform,
    )

    mu_gen, sigma_gen = gaussian_statistics(gen_features)
    fid = frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)
    is_mean, is_std = inception_score_from_logits(gen_logits, splits=args.inception_splits)

    summary = {
        "device": str(device),
        "inception_weights": args.inception_weights,
        "reference_source": args.reference_source,
        "generated_source": args.generated_source,
        "reference_samples": int(reference_stats["count"]),
        "generated_samples": int(gen_features.shape[0]),
        "feature_dim": int(mu_ref.shape[0]),
        "fid": float(fid),
        "inception_score_mean": float(is_mean),
        "inception_score_std": float(is_std),
        "inception_splits": args.inception_splits,
        "metrics_validity": "approximate" if args.inception_weights == "none" else "standard",
        "cache_reference_stats": args.cache_reference_stats,
        "load_reference_stats": args.load_reference_stats,
        "protocol": {
            **eval_contract,
            "contract_sha256": eval_contract_sha256,
        },
        "inception_provenance": {
            "torchvision_version": str(getattr(torchvision, "__version__", "unknown")),
            "weights_mode": args.inception_weights,
            "weights_enum": weights_enum,
            "weights_cache": weights_cache,
        },
        "reference_stats_provenance": {
            "loaded_path_sha256": loaded_reference_stats_sha256,
            "loaded_contract_sha256": reference_stats.get("contract_sha256"),
            "loaded_contract": reference_stats.get("contract"),
            "loaded_source": reference_stats.get("provenance"),
        },
        "env_fingerprint": environment_fingerprint(),
    }
    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        repo_root = Path(__file__).resolve().parents[1]
        write_json(output_path.parent / "env_snapshot.json", environment_snapshot(paths=[output_path.parent]))
        write_json(output_path.parent / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))
        write_json(output_path.parent / "env_fingerprint.json", environment_fingerprint())
        summary["paths"] = {
            "env_snapshot_json": str(output_path.parent / "env_snapshot.json"),
            "codebase_fingerprint_json": str(output_path.parent / "codebase_fingerprint.json"),
            "env_fingerprint_json": str(output_path.parent / "env_fingerprint.json"),
        }
        write_json(output_path, summary)
    print(json.dumps(summary, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID + Inception Score for generated image sets")
    add_device_argument(parser, default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--inception-weights", choices=("pretrained", "none"), default="pretrained")
    parser.add_argument("--inception-splits", type=int, default=10)
    parser.add_argument("--max-reference-samples", type=int, default=None)
    parser.add_argument("--max-generated-samples", type=int, default=None)
    parser.add_argument("--cache-reference-stats", type=str, default=None)
    parser.add_argument("--load-reference-stats", type=str, default=None)
    parser.add_argument("--allow-reference-contract-mismatch", action="store_true")
    parser.add_argument("--image-exts", nargs="*", type=str, default=[])

    parser.add_argument("--reference-source", choices=("imagefolder", "tensor_file"), required=True)
    parser.add_argument("--reference-imagefolder-root", type=str, default=None)
    parser.add_argument("--reference-tensor-file-path", type=str, default=None)

    parser.add_argument("--generated-source", choices=("imagefolder", "tensor_file"), required=True)
    parser.add_argument("--generated-imagefolder-root", type=str, default=None)
    parser.add_argument("--generated-tensor-file-path", type=str, default=None)
    return parser.parse_args()


def _build_inception(*, device: torch.device, weights_mode: str) -> tuple[torch.nn.Module, str | None]:
    if weights_mode == "none":
        # Ensure deterministic random initialization when running in smoke/CI mode.
        torch.manual_seed(0)
    weights = Inception_V3_Weights.IMAGENET1K_V1 if weights_mode == "pretrained" else None
    # Torchvision requires aux logits enabled for pretrained inception_v3 weights.
    aux_logits = True if weights is not None else False
    # Keep init_weights=False for deterministic and fast init when not loading weights.
    # Torchvision also enforces init_weights=False when loading pretrained weights.
    model = inception_v3(weights=weights, aux_logits=aux_logits, init_weights=False).to(device)
    model.eval()
    enum_name = None if weights is None else f"{weights.__class__.__name__}.{weights.name}"
    return model, enum_name


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


def _build_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize((299, 299), antialias=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _build_loader(
    *,
    kind: str,
    source: str,
    imagefolder_root: str | None,
    tensor_file_path: str | None,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
    image_exts: list[str],
) -> DataLoader:
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]]
    if source == "imagefolder":
        if imagefolder_root is None:
            raise ValueError(f"--{kind}-imagefolder-root is required when {kind}_source=imagefolder")
        root = Path(imagefolder_root)
        if not root.exists() or not root.is_dir():
            raise ValueError(f"{kind} imagefolder root does not exist or is not a directory: {root}")
        dataset = ImageFolder(
            root=str(root),
            # ImageNet images have variable H/W, so we must resize before collation.
            # The Inception protocol also resizes to 299 anyway; keeping it here avoids
            # DataLoader stack failures on variable-size inputs.
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
    elif source == "tensor_file":
        if tensor_file_path is None:
            raise ValueError(f"--{kind}-tensor-file-path is required when {kind}_source=tensor_file")
        tensor_dataset = _load_tensor_dataset(Path(tensor_file_path))
        labels = torch.zeros(tensor_dataset.shape[0], dtype=torch.long)
        dataset = TensorDataset(tensor_dataset, labels)
    else:
        raise ValueError(f"Unsupported source: {source}")

    if max_samples is not None and max_samples > 0 and max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(max_samples)))
    if len(dataset) == 0:
        raise ValueError(f"No samples found for {kind} source '{source}'")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


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


def _load_tensor_dataset(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "images" in payload:
        images = payload["images"]
    elif isinstance(payload, torch.Tensor):
        images = payload
    else:
        raise ValueError("Tensor file payload must be Tensor or dict with 'images' key")
    if not isinstance(images, torch.Tensor):
        raise ValueError("Loaded images payload is not a torch.Tensor")
    if images.ndim != 4:
        raise ValueError(f"Expected images shape [N, C, H, W], got {tuple(images.shape)}")
    if images.shape[1] != 3:
        raise ValueError(f"Expected 3 channels for inception input, got {images.shape[1]}")
    return images.float()


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


def _collect_inception_outputs(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    transform: v2.Compose,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    feature_holder: dict[str, torch.Tensor] = {}

    def _capture_features(_module: torch.nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        feature_holder["pool"] = torch.flatten(output, start_dim=1).detach()

    hook = model.avgpool.register_forward_hook(_capture_features)
    with torch.no_grad():
        for images, _labels in loader:
            batch = transform(_normalize_image_range(images)).to(device)
            batch_output = model(batch)
            batch_logits = _extract_logits(batch_output)
            if "pool" not in feature_holder:
                raise RuntimeError("Failed to capture Inception pool features")
            features.append(feature_holder["pool"].cpu())
            logits.append(batch_logits.detach().cpu())
    hook.remove()

    if not features:
        raise ValueError("No images loaded for metric computation")
    return torch.cat(features, dim=0), torch.cat(logits, dim=0)


def _save_reference_stats(
    path: Path,
    stats: dict[str, object],
    *,
    contract: dict[str, object],
    contract_sha256: str,
    provenance: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": stats["mean"].detach().cpu(),
            "cov": stats["cov"].detach().cpu(),
            "count": int(stats["count"]),
            "contract": contract,
            "contract_sha256": contract_sha256,
            "provenance": provenance,
        },
        path,
    )


def _load_reference_stats(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Reference stats payload must be a dict")
    if "mean" not in payload or "cov" not in payload or "count" not in payload:
        raise ValueError("Reference stats must contain mean/cov/count")
    mean = payload["mean"]
    cov = payload["cov"]
    count = payload["count"]
    if not isinstance(mean, torch.Tensor) or not isinstance(cov, torch.Tensor):
        raise ValueError("Reference stats mean/cov must be tensors")
    return {
        "mean": mean,
        "cov": cov,
        "count": int(count),
        "contract": payload.get("contract"),
        "contract_sha256": payload.get("contract_sha256"),
        "provenance": payload.get("provenance"),
    }


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
    source: str,
    imagefolder_root: str | None,
    tensor_file_path: str | None,
    max_samples: int | None,
    image_exts: list[str],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "source": str(source),
        "max_samples": None if max_samples is None else int(max_samples),
        "image_exts": [str(value) for value in image_exts],
    }
    if source == "imagefolder":
        if imagefolder_root is not None:
            payload["imagefolder_root"] = str(Path(imagefolder_root).resolve())
    elif source == "tensor_file":
        if tensor_file_path is not None:
            tensor_path = Path(tensor_file_path).resolve()
            payload["tensor_file_path"] = str(tensor_path)
            payload["tensor_file_size"] = int(tensor_path.stat().st_size)
            payload["tensor_file_mtime_ns"] = int(tensor_path.stat().st_mtime_ns)
    return payload


def _extract_logits(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    # Torchvision may return InceptionOutputs when aux_logits=True.
    if hasattr(output, "logits"):
        logits = getattr(output, "logits")
        if isinstance(logits, torch.Tensor):
            return logits
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Unsupported inception output type: {type(output).__name__}")


if __name__ == "__main__":
    main()
