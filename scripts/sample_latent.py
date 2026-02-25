from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import fields

import torch

from drifting_models.models import DiTLikeConfig, DiTLikeGenerator
from drifting_models.features.vae import LatentDecoderConfig, build_latent_decoder
from drifting_models.sampling import postprocess_images, sample_pixel_generator, write_imagefolder
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
    if args.write_imagefolder and args.decode_mode == "sd_vae" and (args.sd_vae_model_id is None or not args.sd_vae_model_id.strip()):
        raise ValueError("--sd-vae-model-id must be provided when --write-imagefolder and --decode-mode sd_vae")
    device = resolve_device(args.device)
    seed_everything(args.seed)

    checkpoint_path = Path(args.checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    config_hash = file_sha256(Path(args.config)) if args.config is not None else None
    model_config_source = "args"
    model_config = _model_config_from_checkpoint(payload)
    if model_config is not None:
        model_config_source = "checkpoint"
    else:
        model_config = _build_model_config(args)
    generator = DiTLikeGenerator(model_config).to(device)
    _check_config_compatibility(
        checkpoint_payload=payload,
        config_path=None if args.config is None else Path(args.config),
        config_hash=config_hash,
        allow_mismatch=bool(args.allow_config_mismatch),
    )
    generator.load_state_dict(state_dict)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tensor_path = out_root / "latents.pt"
    repo_root = Path(__file__).resolve().parents[1]
    write_json(out_root / "env_snapshot.json", environment_snapshot(paths=[out_root]))
    write_json(out_root / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))
    write_json(out_root / "env_fingerprint.json", environment_fingerprint())

    remaining = int(args.n_samples)
    produced_total = 0
    chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    while remaining > 0:
        batch = min(int(args.batch_size), remaining)
        noise = torch.randn(
            batch,
            int(generator.config.in_channels),
            int(generator.config.image_size),
            int(generator.config.image_size),
            device=device,
        )
        class_labels = torch.randint(0, int(generator.config.num_classes), (batch,), device=device, dtype=torch.long)
        alpha = torch.full((batch,), float(args.alpha), device=device, dtype=torch.float32)
        latents = sample_pixel_generator(
            generator=generator,
            noise=noise,
            class_labels=class_labels,
            alpha=alpha,
            style_indices=None,
        )
        chunks.append(latents.detach().cpu())
        label_chunks.append(class_labels.detach().cpu())
        produced_total += batch
        remaining -= batch

    all_latents = torch.cat(chunks, dim=0)
    all_labels = torch.cat(label_chunks, dim=0)
    torch.save({"latents": all_latents, "labels": all_labels}, tensor_path)

    summary = {
        "device": str(device),
        "seed": int(args.seed),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config_path": args.config,
        "config_hash": config_hash,
        "model_config": generator.config.__dict__,
        "model_config_source": model_config_source,
        "output_root": str(out_root),
        "tensor_path": str(tensor_path),
        "n_samples": int(args.n_samples),
        "batch_size": int(args.batch_size),
        "alpha": float(args.alpha),
        "class_sampling": "uniform",
        "env_fingerprint": environment_fingerprint(),
        "paths": {
            "env_snapshot_json": str(out_root / "env_snapshot.json"),
            "codebase_fingerprint_json": str(out_root / "codebase_fingerprint.json"),
            "env_fingerprint_json": str(out_root / "env_fingerprint.json"),
        },
    }

    if args.write_imagefolder:
        images_dir = out_root / "images"
        decoder = build_latent_decoder(
            LatentDecoderConfig(
                mode=args.decode_mode,
                latent_channels=int(all_latents.shape[1]),
                out_channels=int(args.decode_out_channels),
                image_size=int(args.decode_image_size),
                hidden_channels=int(args.decode_hidden_channels),
                sd_vae_model_id=args.sd_vae_model_id,
                sd_vae_subfolder=args.sd_vae_subfolder,
                sd_vae_revision=args.sd_vae_revision,
                sd_vae_scaling_factor=float(args.sd_vae_scaling_factor),
                sd_vae_dtype=args.sd_vae_dtype,
            )
        )
        decoder = decoder.to(device=device)
        total_saved = 0
        total_attempted = 0
        start_index = 0
        for latents_chunk, labels_chunk in zip(chunks, label_chunks):
            decoded = decoder(latents_chunk.to(device=device)).detach().cpu()
            images = postprocess_images(decoded, mode=args.postprocess_mode)
            attempted = int(images.shape[0])
            saved = write_imagefolder(
                images,
                labels_chunk,
                root=images_dir,
                start_index=start_index,
                image_format=args.image_format,
                overwrite=args.overwrite,
            )
            total_attempted += attempted
            total_saved += int(saved)
            start_index += attempted
        summary["images_dir"] = str(images_dir)
        summary["attempted_images"] = int(total_attempted)
        summary["saved_images"] = int(total_saved)
        summary["decode"] = {
            "mode": args.decode_mode,
            "out_channels": int(args.decode_out_channels),
            "image_size": int(args.decode_image_size),
            "hidden_channels": int(args.decode_hidden_channels),
            "sd_vae_model_id": args.sd_vae_model_id,
            "sd_vae_subfolder": args.sd_vae_subfolder,
            "sd_vae_revision": args.sd_vae_revision,
            "sd_vae_scaling_factor": float(args.sd_vae_scaling_factor),
            "sd_vae_dtype": args.sd_vae_dtype,
        }
        if args.decode_mode == "sd_vae":
            summary["decode"]["sd_vae_provenance"] = _resolve_sd_vae_provenance(
                model_id=args.sd_vae_model_id,
                subfolder=args.sd_vae_subfolder,
                revision=args.sd_vae_revision,
            )

    write_json(out_root / "sample_summary.json", summary)
    print(json.dumps(summary, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample latent-space generator checkpoint to tensor file")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="Optional simple key:value config file")
    parser.add_argument("--allow-config-mismatch", action="store_true")
    add_device_argument(parser, default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--write-imagefolder", action="store_true")
    parser.add_argument("--decode-mode", choices=("identity", "conv", "sd_vae"), default="conv")
    parser.add_argument("--decode-out-channels", type=int, default=3)
    parser.add_argument("--decode-image-size", type=int, default=32)
    parser.add_argument("--decode-hidden-channels", type=int, default=64)
    parser.add_argument("--sd-vae-model-id", type=str, default=None)
    parser.add_argument("--sd-vae-subfolder", type=str, default=None)
    parser.add_argument("--sd-vae-revision", type=str, default="31f26fdeee1355a5c34592e401dd41e45d25a493")
    parser.add_argument("--sd-vae-scaling-factor", type=float, default=0.18215)
    parser.add_argument("--sd-vae-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--postprocess-mode", choices=("clamp_0_1", "tanh_to_0_1", "sigmoid", "identity"), default="identity")
    parser.add_argument("--image-format", choices=("png", "jpg", "jpeg"), default="png")
    parser.add_argument("--overwrite", action="store_true")

    # Model config mirrors train_latent defaults.
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--register-tokens", type=int, default=16)
    parser.add_argument("--style-vocab-size", type=int, default=64)
    parser.add_argument("--style-token-count", type=int, default=32)
    parser.add_argument("--norm-type", type=str, default="layernorm")
    parser.add_argument("--use-qk-norm", action="store_true")
    parser.add_argument("--use-rope", action="store_true")
    return parser.parse_args()


def _build_model_config(args: argparse.Namespace) -> DiTLikeConfig:
    if args.config is not None:
        overrides = _load_simple_kv_config(Path(args.config))
        for key, raw_value in overrides.items():
            attr = key.replace("-", "_")
            if not hasattr(args, attr):
                continue
            current = getattr(args, attr)
            setattr(args, attr, _coerce_like(raw_value, current))
    return DiTLikeConfig(
        image_size=args.image_size,
        in_channels=args.channels,
        out_channels=args.channels,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=args.num_classes,
        register_tokens=args.register_tokens,
        style_vocab_size=args.style_vocab_size,
        style_token_count=args.style_token_count,
        norm_type=args.norm_type,
        use_qk_norm=args.use_qk_norm,
        use_rope=args.use_rope,
    )


def _model_config_from_checkpoint(payload: object) -> DiTLikeConfig | None:
    if not isinstance(payload, dict):
        return None
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        return None
    raw = extra.get("model_config")
    if not isinstance(raw, dict):
        return None
    allowed = {field.name for field in fields(DiTLikeConfig)}
    filtered = {key: raw[key] for key in raw.keys() if key in allowed}
    if not filtered:
        return None
    return DiTLikeConfig(**filtered)


def _load_simple_kv_config(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid config line: {raw_line}")
        key, value = line.split(":", 1)
        entries[key.strip()] = value.strip()
    return entries


def _coerce_like(raw_value: str, template: object) -> object:
    if isinstance(template, bool):
        lowered = raw_value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean: {raw_value}")
    if isinstance(template, int):
        return int(raw_value)
    if isinstance(template, float):
        return float(raw_value)
    if isinstance(template, str):
        return raw_value
    raise ValueError(f"Unsupported type: {type(template).__name__}")


def _resolve_sd_vae_provenance(*, model_id: str | None, subfolder: str | None, revision: str | None) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_id": model_id,
        "subfolder": subfolder,
        "requested_revision": revision,
        "resolved_commit_hash": None,
    }
    if model_id is None or not model_id.strip():
        return payload
    try:
        from huggingface_hub import model_info

        info = model_info(repo_id=model_id, revision=revision)
        payload["resolved_commit_hash"] = getattr(info, "sha", None)
    except Exception as error:  # pragma: no cover - optional dependency/network
        payload["resolve_error"] = str(error)
    return payload


def _check_config_compatibility(
    *,
    checkpoint_payload: object,
    config_path: Path | None,
    config_hash: str | None,
    allow_mismatch: bool,
) -> None:
    if allow_mismatch:
        return
    if config_path is None or config_hash is None:
        return
    if not isinstance(checkpoint_payload, dict):
        return
    extra = checkpoint_payload.get("extra")
    if not isinstance(extra, dict):
        return
    expected = extra.get("config_hash")
    if expected is None:
        return
    if not isinstance(expected, str):
        return
    if expected != config_hash:
        raise ValueError(f"Config hash mismatch for {config_path}: expected {expected}, got {config_hash}")


if __name__ == "__main__":
    main()
