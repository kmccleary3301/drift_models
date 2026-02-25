from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from drifting_models.features.sd_vae import SDVAEConfig, encode_images_to_latents, load_sd_vae, resolve_dtype
from drifting_models.utils import (
    add_device_argument,
    codebase_fingerprint,
    environment_fingerprint,
    environment_snapshot,
    file_sha256,
    resolve_device,
    seed_everything,
    write_json,
)


def main() -> None:
    args = _parse_args()
    device = resolve_device(args.device)
    repo_root = Path(__file__).resolve().parents[1]
    run_root = _resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "env_snapshot.json", environment_snapshot(paths=[run_root]))
    write_json(run_root / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))
    write_json(run_root / "env_fingerprint.json", environment_fingerprint())
    if args.seed is not None:
        seed_everything(int(args.seed))

    try:
        from torchvision import datasets, transforms
    except ModuleNotFoundError as error:
        raise RuntimeError("torchvision is required for ImageFolder exports") from error

    root = Path(args.imagefolder_root)
    if not root.exists():
        raise FileNotFoundError(str(root))

    transform_ops: list[object] = [
        transforms.Resize((int(args.image_size), int(args.image_size)), antialias=True),
    ]
    transform_descriptor: list[str] = [f"Resize({int(args.image_size)}x{int(args.image_size)})"]
    if not args.disable_center_crop:
        transform_ops.append(transforms.CenterCrop(int(args.image_size)))
        transform_descriptor.append(f"CenterCrop({int(args.image_size)})")
    transform_ops.append(transforms.Lambda(lambda image: image.convert("RGB")))
    transform_descriptor.append("ConvertRGB")
    transform_ops.append(transforms.ToTensor())  # [0,1]
    transform_descriptor.append("ToTensor[0,1]")
    transform = transforms.Compose(transform_ops)
    dataset = datasets.ImageFolder(root=str(root), transform=transform)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        drop_last=False,
    )

    sd_cfg = SDVAEConfig(
        scaling_factor=float(args.scaling_factor),
        model_id=str(args.model_id),
        subfolder=args.subfolder,
        revision=args.revision,
        dtype=str(args.dtype),
    )
    vae = load_sd_vae(config=sd_cfg).to(device=device, dtype=resolve_dtype(sd_cfg.dtype))

    max_samples = None if args.max_samples <= 0 else int(args.max_samples)
    latent_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    buffered = 0
    produced = 0
    shard_records: list[dict[str, object]] = []
    shard_index = 0

    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        def tqdm(iterable, **_kwargs):  # type: ignore[no-redef]
            return iterable

    for images_0_1, labels in tqdm(loader, desc="export_sd_vae_latents", unit="batch"):
        if max_samples is not None and produced >= max_samples:
            break
        remaining = None if max_samples is None else max_samples - produced
        if remaining is not None and int(images_0_1.shape[0]) > remaining:
            images_0_1 = images_0_1[:remaining]
            labels = labels[:remaining]

        images_0_1 = images_0_1.to(device=device)
        latents = encode_images_to_latents(
            vae=vae,
            images_0_1=images_0_1,
            scaling_factor=sd_cfg.scaling_factor,
            sample=(args.latent_sampling == "sample"),
        )
        latent_chunks.append(latents.detach().cpu())
        label_chunks.append(labels.detach().cpu().long())
        buffered += int(labels.shape[0])
        produced += int(labels.shape[0])

        if args.shard_size > 0:
            shard_size = int(args.shard_size)
            while buffered >= shard_size:
                latents_buf = torch.cat(latent_chunks, dim=0)
                labels_buf = torch.cat(label_chunks, dim=0)
                shard_latents = latents_buf[:shard_size]
                shard_labels = labels_buf[:shard_size]
                record = _write_shard(
                    latents=shard_latents,
                labels=shard_labels,
                args=args,
                root=root,
                dataset_classes=len(dataset.classes),
                sd_cfg=sd_cfg,
                pre_encode_transforms=transform_descriptor,
                shard_index=shard_index,
            )
                shard_records.append(record)
                shard_index += 1

                remainder_latents = latents_buf[shard_size:]
                remainder_labels = labels_buf[shard_size:]
                latent_chunks = [] if remainder_latents.numel() == 0 else [remainder_latents]
                label_chunks = [] if remainder_labels.numel() == 0 else [remainder_labels]
                buffered = int(remainder_labels.shape[0]) if remainder_labels.numel() != 0 else 0

    if args.shard_size > 0:
        if buffered > 0:
            latents_buf = torch.cat(latent_chunks, dim=0)
            labels_buf = torch.cat(label_chunks, dim=0)
            record = _write_shard(
                latents=latents_buf,
                labels=labels_buf,
                args=args,
                root=root,
                dataset_classes=len(dataset.classes),
                sd_cfg=sd_cfg,
                pre_encode_transforms=transform_descriptor,
                shard_index=shard_index,
            )
            shard_records.append(record)
        summary = _finalize_sharded_export(
            args=args,
            imagefolder_root=root,
            dataset_classes=len(dataset.classes),
            device=device,
            sd_cfg=sd_cfg,
            shard_records=shard_records,
            pre_encode_transforms=transform_descriptor,
        )
    else:
        latents_all = torch.cat(latent_chunks, dim=0)
        labels_all = torch.cat(label_chunks, dim=0)
        if latents_all.shape[0] != labels_all.shape[0]:
            raise RuntimeError("latent/label count mismatch")

        out_path = Path(_required_path(args.output_tensor_path, "--output-tensor-path"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "images": latents_all.to(dtype=torch.float16 if args.save_dtype == "fp16" else torch.float32),
            "labels": labels_all,
            "meta": {
                "source": "sd_vae_latents_tensor_file",
                "imagefolder_root": str(root.resolve()),
                "image_size": int(args.image_size),
                "latent_sampling": str(args.latent_sampling),
                "sd_vae": sd_cfg.__dict__,
                "pre_encode_transforms": transform_descriptor,
                "transform_ordering": "preprocess_before_sd_vae_encode",
            },
        }
        torch.save(payload, out_path)

        summary = {
            "output_tensor_path": str(out_path),
            "output_tensor_sha256": file_sha256(out_path),
            "imagefolder_root": str(root.resolve()),
            "image_size": int(args.image_size),
            "num_classes": int(len(dataset.classes)),
            "exported_samples": int(latents_all.shape[0]),
            "latents_shape": list(latents_all.shape),
            "latents_dtype_saved": str(payload["images"].dtype),
            "labels_shape": list(labels_all.shape),
            "pre_encode_transforms": transform_descriptor,
            "transform_ordering": "preprocess_before_sd_vae_encode",
            "device": str(device),
            "env_fingerprint": environment_fingerprint(),
            "paths": {
                "env_snapshot_json": str(run_root / "env_snapshot.json"),
                "codebase_fingerprint_json": str(run_root / "codebase_fingerprint.json"),
                "env_fingerprint_json": str(run_root / "env_fingerprint.json"),
            },
        }

    if args.summary_json_path is not None:
        write_json(Path(args.summary_json_path), summary)
    print(json.dumps(summary, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ImageFolder images to SD-VAE latents tensor-file dataset")
    parser.add_argument("--imagefolder-root", type=str, required=True)
    parser.add_argument("--output-tensor-path", type=str, default=None, help="Single-file output path (non-sharded)")
    parser.add_argument("--output-shards-dir", type=str, default=None, help="Directory to write shard_*.pt files")
    parser.add_argument(
        "--shard-size",
        type=int,
        default=0,
        help="If >0, write sharded outputs of this many samples per shard (requires --output-shards-dir).",
    )
    parser.add_argument("--summary-json-path", type=str, default=None)
    add_device_argument(parser, default="auto")
    parser.add_argument("--seed", type=int, default=None, help="Only used when --latent-sampling sample")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--disable-center-crop", action="store_true")
    parser.add_argument("--max-samples", type=int, default=-1, help="<=0 means all")

    parser.add_argument("--model-id", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--subfolder", type=str, default=None)
    parser.add_argument("--revision", type=str, default="31f26fdeee1355a5c34592e401dd41e45d25a493")
    parser.add_argument("--scaling-factor", type=float, default=0.18215)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--save-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--latent-sampling", choices=("mean", "sample"), default="mean")
    return parser.parse_args()


def _resolve_run_root(args: argparse.Namespace) -> Path:
    if args.output_shards_dir is not None and str(args.output_shards_dir).strip():
        return Path(args.output_shards_dir)
    if args.output_tensor_path is not None and str(args.output_tensor_path).strip():
        return Path(args.output_tensor_path).expanduser().resolve().parent
    if args.summary_json_path is not None and str(args.summary_json_path).strip():
        return Path(args.summary_json_path).expanduser().resolve().parent
    return Path.cwd()


def _required_path(value: str | None, flag: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"{flag} is required")
    return str(value)


def _write_shard(
    *,
    latents: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
    root: Path,
    dataset_classes: int,
    sd_cfg: SDVAEConfig,
    pre_encode_transforms: list[str],
    shard_index: int,
) -> dict[str, object]:
    if args.output_shards_dir is None:
        raise ValueError("--output-shards-dir is required when --shard-size > 0")
    if latents.shape[0] != labels.shape[0]:
        raise RuntimeError("latent/label count mismatch (shard)")

    out_dir = Path(args.output_shards_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"shard_{shard_index:08d}.pt"
    payload = {
        "images": latents.to(dtype=torch.float16 if args.save_dtype == "fp16" else torch.float32),
        "labels": labels,
        "meta": {
            "source": "sd_vae_latents_tensor_shard",
            "imagefolder_root": str(root.resolve()),
            "image_size": int(args.image_size),
            "latent_sampling": str(args.latent_sampling),
            "sd_vae": sd_cfg.__dict__,
            "pre_encode_transforms": pre_encode_transforms,
            "transform_ordering": "preprocess_before_sd_vae_encode",
            "num_classes": int(dataset_classes),
        },
    }
    torch.save(payload, shard_path)

    record = {
        "path": shard_path.name,
        "sha256": file_sha256(shard_path),
        "count": int(payload["images"].shape[0]),
        "images_shape": list(payload["images"].shape),
        "images_dtype": str(payload["images"].dtype),
    }
    return record


def _finalize_sharded_export(
    *,
    args: argparse.Namespace,
    imagefolder_root: Path,
    dataset_classes: int,
    device: torch.device,
    sd_cfg: SDVAEConfig,
    shard_records: list[dict[str, object]],
    pre_encode_transforms: list[str],
) -> dict[str, object]:
    out_dir = Path(_required_path(args.output_shards_dir, "--output-shards-dir"))
    manifest_path = out_dir / "manifest.json"
    total = int(sum(int(s["count"]) for s in shard_records))
    latent_item_shape = None
    latents_dtype_saved = None
    if shard_records:
        shape = shard_records[0].get("images_shape")
        if isinstance(shape, list) and len(shape) >= 2:
            latent_item_shape = shape[1:]
        latents_dtype_saved = shard_records[0].get("images_dtype")
    manifest_payload = {
        "kind": "sd_vae_latent_shards",
        "imagefolder_root": str(imagefolder_root.resolve()),
        "image_size": int(args.image_size),
        "latent_sampling": str(args.latent_sampling),
        "sd_vae": sd_cfg.__dict__,
        "pre_encode_transforms": pre_encode_transforms,
        "transform_ordering": "preprocess_before_sd_vae_encode",
        "num_classes": int(dataset_classes),
        "shard_size": int(args.shard_size),
        "exported_samples": total,
        "latent_item_shape": latent_item_shape,
        "latents_dtype_saved": latents_dtype_saved,
        "shards": shard_records,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return {
        "output_shards_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "manifest_shards": int(len(shard_records)),
        "exported_samples": total,
        "imagefolder_root": str(imagefolder_root.resolve()),
        "image_size": int(args.image_size),
        "num_classes": int(dataset_classes),
        "pre_encode_transforms": pre_encode_transforms,
        "transform_ordering": "preprocess_before_sd_vae_encode",
        "device": str(device),
        "env_fingerprint": environment_fingerprint(),
        "paths": {
            "env_snapshot_json": str(out_dir / "env_snapshot.json"),
            "codebase_fingerprint_json": str(out_dir / "codebase_fingerprint.json"),
            "env_fingerprint_json": str(out_dir / "env_fingerprint.json"),
        },
    }


if __name__ == "__main__":
    main()
