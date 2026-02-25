from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from dataclasses import replace
import time

import torch
from torch.nn import functional as F

from drifting_models.features import LatentResNetMAE, LatentResNetMAEConfig
from drifting_models.data import RealBatchProvider, RealBatchProviderConfig
from drifting_models.train import MAETrainConfig, run_mae_pretrain
from drifting_models.utils import (
    add_device_argument,
    codebase_fingerprint,
    environment_fingerprint,
    environment_snapshot,
    file_sha256,
    load_training_checkpoint,
    resolve_device,
    save_training_checkpoint,
    write_json,
)
from drifting_models.utils.run_md import write_run_md


def main() -> None:
    args = _parse_args()
    args = _apply_config_overrides(args)
    if args.output_dir is not None and args.export_encoder_path is None:
        args.export_encoder_path = str(Path(args.output_dir) / "mae_encoder.pt")
    if args.output_dir is not None and args.cls_ft_steps > 0 and args.export_cls_ft_encoder_path is None:
        args.export_cls_ft_encoder_path = str(Path(args.output_dir) / "mae_encoder_clsft.pt")
    config_hash = None if args.config is None else file_sha256(Path(args.config))
    env_fingerprint = environment_fingerprint()
    repo_root = Path(__file__).resolve().parents[1]
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "env_snapshot.json", environment_snapshot(paths=[output_dir]))
        write_json(output_dir / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))
    device = resolve_device(args.device)
    config = MAETrainConfig(
        seed=args.seed,
        steps=args.steps,
        log_every=args.log_every,
        batch_size=args.batch_size,
        image_size=args.image_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        stages=args.stages,
        encoder_arch=args.encoder_arch,
        blocks_per_stage=args.blocks_per_stage,
        norm_groups=args.norm_groups,
        mask_ratio=args.mask_ratio,
        mask_patch_size=args.mask_patch_size,
        mask_schedule=args.mask_schedule,
        mask_warmup_steps=args.mask_warmup_steps,
        val_batch_size=args.val_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        scheduler=args.scheduler,
        warmup_steps=args.warmup_steps,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
    )
    if args.real_loader_batch_size > 0 and int(args.real_loader_batch_size) != int(config.batch_size):
        raise ValueError("--real-loader-batch-size must match --batch-size for MAE training (set 0 to auto)")
    model = LatentResNetMAE(
        LatentResNetMAEConfig(
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            stages=config.stages,
            encoder_arch=config.encoder_arch,
            blocks_per_stage=config.blocks_per_stage,
            norm_groups=config.norm_groups,
            mask_ratio=config.mask_ratio,
            mask_patch_size=config.mask_patch_size,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )
    start_step = 0
    if args.resume_from is not None:
        payload = load_training_checkpoint(
            path=Path(args.resume_from),
            model=model,
            optimizer=optimizer,
            map_location=device,
        )
        start_step = int(payload.get("step", 0))

    real_batch_provider, real_provider_config = _build_real_batch_provider(args=args, config=config)
    real_provider_sanity_report = None
    if args.real_sanity_sample_batches > 0 and real_provider_config is not None:
        sanity_provider = RealBatchProvider(
            replace(
                real_provider_config,
                seed=int(real_provider_config.seed + 9999),
                shuffle=True,
            )
        )
        real_provider_sanity_report = _build_real_provider_sanity_report(
            provider=sanity_provider,
            sample_batches=int(args.real_sanity_sample_batches),
            num_classes=int(config.num_classes),
            device=device,
        )

    def _on_step_end(step_value: int, current_model: LatentResNetMAE, current_optimizer: torch.optim.Optimizer) -> None:
        if args.checkpoint_path is None:
            return
        if args.save_every <= 0:
            return
        if step_value % args.save_every != 0:
            return
        save_training_checkpoint(
            path=Path(args.checkpoint_path),
            model=current_model,
            optimizer=current_optimizer,
            step=step_value,
            extra={"config_path": args.config, "config_hash": config_hash},
        )

    summary = run_mae_pretrain(
        config=config,
        device=device,
        model=model,
        optimizer=optimizer,
        start_step=start_step,
        on_step_end=_on_step_end,
        real_batch_provider=real_batch_provider,
    )
    if args.checkpoint_path is not None:
        save_training_checkpoint(
            path=Path(args.checkpoint_path),
            model=model,
            optimizer=optimizer,
            step=config.steps,
            extra={"config_path": args.config, "config_hash": config_hash},
        )
    if args.export_encoder_path is not None:
        _export_encoder(
            model=model,
            config=config,
            config_path=args.config,
            config_hash=config_hash,
            export_path=Path(args.export_encoder_path),
        )
    cls_ft_summary = None
    if args.cls_ft_steps > 0:
        cls_ft_summary = _run_classifier_finetune(
            model=model,
            config=config,
            args=args,
            device=device,
            real_provider_config=real_provider_config,
        )
        if args.checkpoint_path is not None:
            save_training_checkpoint(
                path=Path(args.checkpoint_path),
                model=model,
                optimizer=optimizer,
                step=config.steps,
                extra={
                    "config_path": args.config,
                    "config_hash": config_hash,
                    "cls_ft_steps": int(args.cls_ft_steps),
                },
            )
        if args.export_cls_ft_encoder_path is not None:
            _export_encoder(
                model=model,
                config=config,
                config_path=args.config,
                config_hash=config_hash,
                export_path=Path(args.export_cls_ft_encoder_path),
                extra={"cls_ft": cls_ft_summary},
            )
    summary["config_path"] = args.config
    summary["config_hash"] = config_hash
    summary["env_fingerprint"] = env_fingerprint
    summary["resume_from"] = args.resume_from
    summary["resume_step"] = float(start_step)
    summary["checkpoint_path"] = args.checkpoint_path
    summary["export_encoder_path"] = args.export_encoder_path
    summary["export_cls_ft_encoder_path"] = args.export_cls_ft_encoder_path
    if cls_ft_summary is not None:
        summary["cls_ft"] = cls_ft_summary
    if real_batch_provider is not None:
        summary["real_batch_provider"] = {
            "source": real_batch_provider.config.source,
            "manifest_fingerprint": real_batch_provider.manifest_fingerprint,
            "config": real_batch_provider.config.__dict__,
        }
    if real_provider_sanity_report is not None:
        summary["real_provider_sanity_report"] = real_provider_sanity_report
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        write_json(output_dir / "mae_summary.json", summary)
        write_json(output_dir / "env_fingerprint.json", env_fingerprint)
        write_run_md(
            output_dir / "RUN.md",
            {
                "output_root": str(output_dir),
                "paths": {
                    "mae_summary_json": str(output_dir / "mae_summary.json"),
                    "env_snapshot_json": str(output_dir / "env_snapshot.json"),
                    "codebase_fingerprint_json": str(output_dir / "codebase_fingerprint.json"),
                    "env_fingerprint_json": str(output_dir / "env_fingerprint.json"),
                    "checkpoint_path": args.checkpoint_path,
                    "export_encoder_path": args.export_encoder_path,
                    "export_cls_ft_encoder_path": args.export_cls_ft_encoder_path,
                },
                "args": vars(args),
                "commands": {"train": {"argv": list(sys.argv), "returncode": 0}},
            },
        )
    print(json.dumps(summary, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent-space MAE feature encoder scaffold")
    parser.add_argument("--config", type=str, default=None, help="Optional simple key:value config file")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional artifact output directory")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--export-encoder-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    add_device_argument(parser, default="auto")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument(
        "--encoder-arch",
        choices=("resnet_unet", "legacy_conv", "paper_resnet34_unet"),
        default="resnet_unet",
    )
    parser.add_argument("--blocks-per-stage", type=int, default=2)
    parser.add_argument("--norm-groups", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument("--mask-patch-size", type=int, default=2)
    parser.add_argument("--mask-schedule", choices=("fixed", "linear_warmup", "cosine"), default="fixed")
    parser.add_argument("--mask-warmup-steps", type=int, default=0)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--scheduler", choices=("none", "cosine", "warmup_cosine"), default="none")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--cls-ft-steps", type=int, default=0)
    parser.add_argument("--cls-ft-log-every", type=int, default=10)
    parser.add_argument("--cls-ft-batch-size", type=int, default=0)
    parser.add_argument("--cls-ft-learning-rate", type=float, default=1e-4)
    parser.add_argument("--cls-ft-weight-decay", type=float, default=0.01)
    parser.add_argument("--cls-ft-freeze-encoder", action="store_true")
    parser.add_argument("--cls-ft-val-batches", type=int, default=1)
    parser.add_argument("--export-cls-ft-encoder-path", type=str, default=None)

    parser.add_argument("--real-batch-source", type=str, default="synthetic_dataset")
    parser.add_argument("--real-dataset-size", type=int, default=4096)
    parser.add_argument(
        "--real-loader-batch-size",
        type=int,
        default=0,
        help="If >0, override the real batch provider DataLoader batch size; otherwise use --batch-size.",
    )
    parser.add_argument("--real-num-workers", type=int, default=0)
    parser.add_argument("--real-sanity-sample-batches", type=int, default=0)
    parser.add_argument("--real-imagefolder-root", type=str, default=None)
    parser.add_argument("--real-webdataset-urls", type=str, default=None)
    parser.add_argument("--real-tensor-file-path", type=str, default=None)
    parser.add_argument("--real-tensor-shards-manifest-path", type=str, default=None)
    parser.add_argument("--real-transform-resize", type=int, default=None)
    parser.add_argument("--disable-real-center-crop", action="store_true")
    parser.add_argument("--real-horizontal-flip", action="store_true")
    parser.add_argument("--real-transform-normalize", action="store_true")
    return parser.parse_args()


def _apply_config_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        return args
    config_path = Path(args.config)
    entries = _load_simple_kv_config(config_path)
    for key, raw_value in entries.items():
        attr = key.replace("-", "_")
        if not hasattr(args, attr):
            raise ValueError(f"Unknown config key '{key}' in {config_path}")
        current = getattr(args, attr)
        setattr(args, attr, _coerce_like(raw_value, current))
    return args


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
    if template is None:
        return raw_value
    if isinstance(template, bool):
        return _parse_bool(raw_value)
    if isinstance(template, int):
        return int(raw_value)
    if isinstance(template, float):
        return float(raw_value)
    if isinstance(template, str):
        return raw_value
    raise ValueError(f"Unsupported config value type: {type(template).__name__}")


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _build_real_batch_provider(
    *, args: argparse.Namespace, config: MAETrainConfig
) -> tuple[RealBatchProvider | None, RealBatchProviderConfig | None]:
    if args.real_batch_source == "synthetic_dataset":
        return None, None
    provider_config = RealBatchProviderConfig(
        source=str(args.real_batch_source),
        dataset_size=int(args.real_dataset_size),
        batch_size=int(args.real_loader_batch_size) if int(args.real_loader_batch_size) > 0 else int(config.batch_size),
        shuffle=True,
        num_workers=int(args.real_num_workers),
        seed=int(args.seed),
        channels=int(config.in_channels),
        image_size=int(config.image_size),
        num_classes=int(config.num_classes),
        imagefolder_root=args.real_imagefolder_root,
        webdataset_urls=args.real_webdataset_urls,
        tensor_file_path=args.real_tensor_file_path,
        tensor_shards_manifest_path=args.real_tensor_shards_manifest_path,
        transform_resize=args.real_transform_resize,
        transform_center_crop=not bool(args.disable_real_center_crop),
        transform_horizontal_flip=bool(args.real_horizontal_flip),
        transform_normalize=bool(args.real_transform_normalize),
    )
    return RealBatchProvider(provider_config), provider_config


def _build_real_provider_sanity_report(
    *,
    provider: RealBatchProvider,
    sample_batches: int,
    num_classes: int,
    device: torch.device,
) -> dict[str, object]:
    if sample_batches <= 0:
        raise ValueError("sample_batches must be > 0")
    labels: list[torch.Tensor] = []
    images_dtype = None
    for _ in range(sample_batches):
        images, y = provider.next_batch(device=device)
        labels.append(y.detach().cpu().long())
        if images_dtype is None:
            images_dtype = str(images.dtype)
    labels_all = torch.cat(labels, dim=0)
    counts = torch.bincount(labels_all, minlength=int(num_classes))
    covered = int((counts > 0).sum().item())
    non_zero = counts[counts > 0]
    return {
        "sample_batches": int(sample_batches),
        "samples": int(labels_all.numel()),
        "images_dtype": images_dtype,
        "label_min": int(labels_all.min().item()) if labels_all.numel() else None,
        "label_max": int(labels_all.max().item()) if labels_all.numel() else None,
        "covered_classes": int(covered),
        "min_class_count_nonzero": int(non_zero.min().item()) if non_zero.numel() else 0,
        "max_class_count": int(counts.max().item()) if counts.numel() else 0,
        "mean_class_count": float(counts.float().mean().item()) if counts.numel() else 0.0,
    }


def _export_encoder(
    *,
    model: LatentResNetMAE,
    config: MAETrainConfig,
    config_path: str | None,
    config_hash: str | None,
    export_path: Path,
    extra: dict[str, object] | None = None,
) -> None:
    encoder_prefixes = ("encoder.", "paper_stem.", "paper_encoder_stages.")
    encoder_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith(encoder_prefixes)
    }
    payload: dict[str, object] = {
        "export_kind": "latent_resnet_mae_encoder",
        "export_version": 2,
        "config": config.__dict__,
        "mae_model_config": {
            "in_channels": int(config.in_channels),
            "base_channels": int(config.base_channels),
            "stages": int(config.stages),
            "encoder_arch": str(config.encoder_arch),
            "blocks_per_stage": int(config.blocks_per_stage),
            "norm_groups": int(config.norm_groups),
            "mask_ratio": float(config.mask_ratio),
            "mask_patch_size": int(config.mask_patch_size),
        },
        "config_path": config_path,
        "config_sha256": config_hash,
        "encoder_state_dict": encoder_state,
    }
    if extra is not None:
        payload.update(extra)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, export_path)


def _run_classifier_finetune(
    *,
    model: LatentResNetMAE,
    config: MAETrainConfig,
    args: argparse.Namespace,
    device: torch.device,
    real_provider_config: RealBatchProviderConfig | None,
) -> dict[str, object]:
    steps = int(args.cls_ft_steps)
    if steps <= 0:
        raise ValueError("cls_ft_steps must be > 0 when classifier fine-tune is enabled")
    batch_size = int(args.cls_ft_batch_size) if int(args.cls_ft_batch_size) > 0 else int(config.batch_size)
    if int(args.cls_ft_val_batches) <= 0:
        raise ValueError("--cls-ft-val-batches must be > 0")
    train_provider = (
        None
        if real_provider_config is None
        else RealBatchProvider(
            replace(
                real_provider_config,
                batch_size=batch_size,
                seed=int(real_provider_config.seed + 12345),
                shuffle=True,
            )
        )
    )
    val_provider = (
        None
        if real_provider_config is None
        else RealBatchProvider(
            replace(
                real_provider_config,
                batch_size=batch_size,
                seed=int(real_provider_config.seed + 22345),
                shuffle=True,
            )
        )
    )
    feature_dim = _infer_feature_dim(
        model=model,
        in_channels=int(config.in_channels),
        image_size=int(config.image_size),
        device=device,
    )
    classifier = torch.nn.Linear(feature_dim, int(config.num_classes)).to(device)
    if args.cls_ft_freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
    trainable_parameters = list(classifier.parameters())
    if not args.cls_ft_freeze_encoder:
        trainable_parameters.extend(model.encoder.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(args.cls_ft_learning_rate),
        betas=(float(config.adam_beta1), float(config.adam_beta2)),
        weight_decay=float(args.cls_ft_weight_decay),
    )
    logs: list[dict[str, float]] = []
    model.train()
    classifier.train()
    for step in range(steps):
        step_start_time = time.perf_counter()
        latents, labels = _next_cls_batch(
            provider=train_provider,
            batch_size=batch_size,
            channels=int(config.in_channels),
            image_size=int(config.image_size),
            num_classes=int(config.num_classes),
            device=device,
        )
        logits = _classifier_logits(model=model, classifier=classifier, latents=latents)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0)
        optimizer.step()
        train_top1 = float((logits.argmax(dim=-1) == labels).float().mean().item())
        if step == 0 or (step + 1) % int(args.cls_ft_log_every) == 0 or step + 1 == steps:
            val_loss, val_top1 = _evaluate_cls_ft(
                model=model,
                classifier=classifier,
                provider=val_provider,
                batch_size=batch_size,
                channels=int(config.in_channels),
                image_size=int(config.image_size),
                num_classes=int(config.num_classes),
                device=device,
                batches=int(args.cls_ft_val_batches),
            )
            logs.append(
                {
                    "step": float(step + 1),
                    "loss": float(loss.item()),
                    "top1": float(train_top1),
                    "val_loss": float(val_loss),
                    "val_top1": float(val_top1),
                    "grad_norm": float(grad_norm.item()),
                    "step_time_s": float(time.perf_counter() - step_start_time),
                }
            )
    return {
        "steps": int(steps),
        "batch_size": int(batch_size),
        "learning_rate": float(args.cls_ft_learning_rate),
        "weight_decay": float(args.cls_ft_weight_decay),
        "freeze_encoder": bool(args.cls_ft_freeze_encoder),
        "logs": logs,
    }


def _evaluate_cls_ft(
    *,
    model: LatentResNetMAE,
    classifier: torch.nn.Linear,
    provider: RealBatchProvider | None,
    batch_size: int,
    channels: int,
    image_size: int,
    num_classes: int,
    device: torch.device,
    batches: int,
) -> tuple[float, float]:
    losses: list[float] = []
    top1_values: list[float] = []
    model.eval()
    classifier.eval()
    with torch.no_grad():
        for _ in range(int(batches)):
            latents, labels = _next_cls_batch(
                provider=provider,
                batch_size=batch_size,
                channels=channels,
                image_size=image_size,
                num_classes=num_classes,
                device=device,
            )
            logits = _classifier_logits(model=model, classifier=classifier, latents=latents)
            losses.append(float(F.cross_entropy(logits, labels).item()))
            top1_values.append(float((logits.argmax(dim=-1) == labels).float().mean().item()))
    model.train()
    classifier.train()
    return float(sum(losses) / len(losses)), float(sum(top1_values) / len(top1_values))


def _next_cls_batch(
    *,
    provider: RealBatchProvider | None,
    batch_size: int,
    channels: int,
    image_size: int,
    num_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if provider is not None:
        images, labels = provider.next_batch(device=device)
        return images, labels.long()
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    noise = torch.randn(batch_size, channels, image_size, image_size, device=device)
    class_scalar = labels.float() / max(1.0, float(num_classes - 1))
    latents = noise + class_scalar.view(batch_size, 1, 1, 1) * 0.2
    return latents, labels


def _classifier_logits(
    *,
    model: LatentResNetMAE,
    classifier: torch.nn.Linear,
    latents: torch.Tensor,
) -> torch.Tensor:
    features = model.encode(latents)[-1]
    pooled = features.mean(dim=(2, 3))
    return classifier(pooled)


def _infer_feature_dim(
    *,
    model: LatentResNetMAE,
    in_channels: int,
    image_size: int,
    device: torch.device,
) -> int:
    with torch.no_grad():
        probe = torch.zeros(1, in_channels, image_size, image_size, device=device)
        features = model.encode(probe)[-1]
    return int(features.shape[1])


if __name__ == "__main__":
    main()
