from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch

from drifting_models.models import DiTLikeConfig, DiTLikeGenerator
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
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    write_json(out_root / "env_snapshot.json", environment_snapshot(paths=[out_root]))
    write_json(out_root / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))
    write_json(out_root / "env_fingerprint.json", environment_fingerprint())

    existing_max = _existing_max_index(images_dir)
    start_index = args.start_index if args.start_index is not None else existing_max + 1

    target_new_saves = int(args.n_samples)
    attempted_total = 0
    saved_total = 0
    class_file_labels = None
    style_file_indices = None
    if args.class_sampling == "from-file":
        if args.class_file_path is None:
            raise ValueError("--class-file-path is required when --class-sampling from-file")
        class_file_labels = _load_class_labels(Path(args.class_file_path), device=device)
        if class_file_labels.numel() == 0:
            raise ValueError("class file contains no labels")
    if args.style_mode == "from-file":
        if args.style_file_path is None:
            raise ValueError("--style-file-path is required when --style-mode from-file")
        style_file_indices = _load_style_indices(Path(args.style_file_path), device=device)
        if style_file_indices.numel() == 0:
            raise ValueError("style file contains no entries")
    while saved_total < target_new_saves:
        batch = min(int(args.batch_size), target_new_saves - saved_total)
        global_indices = torch.arange(
            start_index + attempted_total,
            start_index + attempted_total + batch,
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn(
            batch,
            int(generator.config.in_channels),
            int(generator.config.image_size),
            int(generator.config.image_size),
            device=device,
        )
        if args.class_sampling == "from-file":
            class_labels = _select_from_file(class_file_labels, global_indices=global_indices).to(
                device=device, dtype=torch.long
            )
        else:
            class_labels = _sample_class_labels(
                args,
                batch=batch,
                device=device,
                num_classes=int(generator.config.num_classes),
            )
        alpha = _build_alpha(args, batch=batch, device=device, global_indices=global_indices)
        if args.style_mode == "from-file":
            style_indices = _select_from_file(style_file_indices, global_indices=global_indices).to(
                device=device, dtype=torch.long
            )
        else:
            style_indices = _build_style_indices(
                args,
                batch=batch,
                device=device,
                style_vocab_size=int(generator.config.style_vocab_size),
                style_token_count=int(generator.config.style_token_count),
            )
        generated = sample_pixel_generator(
            generator=generator,
            noise=noise,
            class_labels=class_labels,
            alpha=alpha,
            style_indices=style_indices,
        )
        processed = postprocess_images(generated, mode=args.postprocess_mode)
        labels_cpu = class_labels.detach().cpu()
        saved = write_imagefolder(
            processed.detach().cpu(),
            labels_cpu,
            root=images_dir,
            start_index=start_index + attempted_total,
            image_format=args.image_format,
            overwrite=args.overwrite,
        )
        attempted_total += batch
        saved_total += saved

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
        "images_dir": str(images_dir),
        "n_samples": int(args.n_samples),
        "saved_images": int(saved_total),
        "attempted_images": int(attempted_total),
        "batch_size": int(args.batch_size),
        "alpha": float(args.alpha),
        "alpha_schedule": args.alpha_schedule,
        "alpha_start": None if args.alpha_start is None else float(args.alpha_start),
        "alpha_end": None if args.alpha_end is None else float(args.alpha_end),
        "alpha_values": list(map(float, args.alpha_values)),
        "class_sampling": args.class_sampling,
        "fixed_class": None if args.fixed_class is None else int(args.fixed_class),
        "class_file_path": args.class_file_path,
        "style_mode": args.style_mode,
        "style_file_path": args.style_file_path,
        "postprocess_mode": args.postprocess_mode,
        "image_format": args.image_format,
        "start_index": int(start_index),
        "end_index_exclusive": int(start_index + attempted_total),
        "overwrite": bool(args.overwrite),
        "env_fingerprint": environment_fingerprint(),
        "paths": {
            "env_snapshot_json": str(out_root / "env_snapshot.json"),
            "codebase_fingerprint_json": str(out_root / "codebase_fingerprint.json"),
            "env_fingerprint_json": str(out_root / "env_fingerprint.json"),
        },
    }
    write_json(out_root / "sample_summary.json", summary)
    print(json.dumps(summary, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample pixel-space generator checkpoint to ImageFolder")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="Optional simple key:value config file")
    parser.add_argument("--allow-config-mismatch", action="store_true")
    add_device_argument(parser, default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--alpha-schedule", choices=("constant", "linear", "list"), default="constant")
    parser.add_argument("--alpha-start", type=float, default=None)
    parser.add_argument("--alpha-end", type=float, default=None)
    parser.add_argument("--alpha-values", nargs="*", type=float, default=[])
    parser.add_argument("--postprocess-mode", choices=("clamp_0_1", "tanh_to_0_1", "sigmoid", "identity"), default="clamp_0_1")
    parser.add_argument("--image-format", choices=("png", "jpg", "jpeg"), default="png")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-index", type=int, default=None)

    parser.add_argument("--class-sampling", choices=("uniform", "fixed", "from-file"), default="uniform")
    parser.add_argument("--fixed-class", type=int, default=None)
    parser.add_argument("--class-file-path", type=str, default=None)
    parser.add_argument("--style-mode", choices=("zeros", "random", "from-file"), default="zeros")
    parser.add_argument("--style-file-path", type=str, default=None)

    # Model config mirrors train_pixel defaults.
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--register-tokens", type=int, default=16)
    parser.add_argument("--norm-type", type=str, default="layernorm")
    parser.add_argument("--use-qk-norm", action="store_true")
    parser.add_argument("--use-rope", action="store_true")
    return parser.parse_args()


def _build_model_config(args: argparse.Namespace) -> DiTLikeConfig:
    if args.config is None:
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
            norm_type=args.norm_type,
            use_qk_norm=args.use_qk_norm,
            use_rope=args.use_rope,
        )
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


def _sample_class_labels(
    args: argparse.Namespace,
    *,
    batch: int,
    device: torch.device,
    num_classes: int,
) -> torch.Tensor:
    if args.class_sampling == "uniform":
        return torch.randint(0, int(num_classes), (batch,), device=device, dtype=torch.long)
    if args.class_sampling == "fixed":
        if args.fixed_class is None:
            raise ValueError("--fixed-class is required when --class-sampling fixed")
        value = int(args.fixed_class)
        if value < 0 or value >= int(num_classes):
            raise ValueError("--fixed-class must be within [0, num_classes)")
        return torch.full((batch,), value, device=device, dtype=torch.long)
    if args.class_sampling == "from-file":
        if args.class_file_path is None:
            raise ValueError("--class-file-path is required when --class-sampling from-file")
        labels = _load_class_labels(Path(args.class_file_path), device=device)
        if labels.numel() == 0:
            raise ValueError("class file contains no labels")
        # Selection is made outside this function (needs global indices).
        raise RuntimeError("from-file labels must be selected with global indices")
    raise ValueError(f"Unknown class_sampling: {args.class_sampling}")


def _build_style_indices(
    args: argparse.Namespace,
    *,
    batch: int,
    device: torch.device,
    style_vocab_size: int,
    style_token_count: int,
) -> torch.Tensor | None:
    if args.style_mode == "zeros":
        return None
    if args.style_mode == "random":
        return torch.randint(0, style_vocab_size, (batch, style_token_count), device=device, dtype=torch.long)
    if args.style_mode == "from-file":
        if args.style_file_path is None:
            raise ValueError("--style-file-path is required when --style-mode from-file")
        styles = _load_style_indices(Path(args.style_file_path), device=device)
        if styles.ndim != 2 or styles.shape[1] != style_token_count:
            raise ValueError(f"Expected styles [N, {style_token_count}], got {tuple(styles.shape)}")
        raise RuntimeError("from-file styles must be selected with global indices")
    raise ValueError(f"Unknown style_mode: {args.style_mode}")


def _build_alpha(args: argparse.Namespace, *, batch: int, device: torch.device, global_indices: torch.Tensor) -> torch.Tensor:
    if args.alpha_schedule == "constant":
        return torch.full((batch,), float(args.alpha), device=device, dtype=torch.float32)
    if args.alpha_schedule == "linear":
        if args.alpha_start is None or args.alpha_end is None:
            raise ValueError("--alpha-start and --alpha-end are required for --alpha-schedule linear")
        denom = max(int(args.n_samples) - 1, 1)
        t = global_indices.float() / float(denom)
        return (float(args.alpha_start) + t * (float(args.alpha_end) - float(args.alpha_start))).to(device=device)
    if args.alpha_schedule == "list":
        if not args.alpha_values:
            raise ValueError("--alpha-values is required for --alpha-schedule list")
        values = torch.tensor(list(map(float, args.alpha_values)), device=device, dtype=torch.float32)
        choice = global_indices % values.numel()
        return values[choice]
    raise ValueError(f"Unsupported alpha_schedule: {args.alpha_schedule}")


def _select_from_file(values: torch.Tensor, *, global_indices: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        choice = global_indices % values.numel()
        return values[choice]
    if values.ndim == 2:
        choice = global_indices % values.shape[0]
        return values[choice]
    raise ValueError("values must be 1D or 2D")


def _load_class_labels(path: Path, *, device: torch.device) -> torch.Tensor:
    if path.suffix.lower() in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            for key in ("labels", "class_labels"):
                if key in payload and isinstance(payload[key], torch.Tensor):
                    return payload[key].to(device=device, dtype=torch.long).view(-1)
        if isinstance(payload, torch.Tensor):
            return payload.to(device=device, dtype=torch.long).view(-1)
        raise ValueError("Unsupported class .pt payload; expected Tensor or dict with labels")
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return torch.empty(0, device=device, dtype=torch.long)
    labels = [int(line.strip()) for line in raw.splitlines() if line.strip()]
    return torch.tensor(labels, device=device, dtype=torch.long)


def _load_style_indices(path: Path, *, device: torch.device) -> torch.Tensor:
    if path.suffix.lower() in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            for key in ("style_indices", "styles"):
                if key in payload and isinstance(payload[key], torch.Tensor):
                    return payload[key].to(device=device, dtype=torch.long)
        if isinstance(payload, torch.Tensor):
            return payload.to(device=device, dtype=torch.long)
        raise ValueError("Unsupported style .pt payload; expected Tensor or dict with style_indices")
    raise ValueError("style-file-path must be a .pt file")


def _existing_max_index(images_root: Path) -> int:
    max_index = -1
    if not images_root.exists():
        return max_index
    for class_dir in images_root.iterdir():
        if not class_dir.is_dir():
            continue
        for file in class_dir.iterdir():
            if not file.is_file():
                continue
            stem = file.stem
            if not stem.isdigit():
                continue
            max_index = max(max_index, int(stem))
    return max_index


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
