from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

import torch

from drifting_models.features.mae import (
    LatentResNetMAE,
    LatentResNetMAEConfig,
    masked_reconstruction_loss,
)
from drifting_models.utils import ModelEMA
from drifting_models.utils.runtime import seed_everything


@dataclass(frozen=True)
class MAETrainConfig:
    seed: int = 1337
    steps: int = 100
    log_every: int = 20
    batch_size: int = 32
    image_size: int = 32
    in_channels: int = 4
    num_classes: int = 1000
    base_channels: int = 32
    stages: int = 3
    encoder_arch: str = "resnet_unet"
    blocks_per_stage: int = 2
    norm_groups: int = 8
    mask_ratio: float = 0.6
    mask_patch_size: int = 2
    mask_schedule: str = "fixed"
    mask_warmup_steps: int = 0
    val_batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    scheduler: str = "none"  # {"none","cosine","warmup_cosine"}
    warmup_steps: int = 0
    use_ema: bool = False
    ema_decay: float = 0.999


def run_mae_pretrain(
    *,
    config: MAETrainConfig,
    device: torch.device,
    model: LatentResNetMAE | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    start_step: int = 0,
    on_step_end: Callable[[int, LatentResNetMAE, torch.optim.Optimizer], None] | None = None,
    real_batch_provider: object | None = None,
) -> dict[str, object]:
    _set_seed(config.seed)
    if model is None:
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
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )
    ema_model = ModelEMA.create(model=model, decay=float(config.ema_decay)) if config.use_ema else None

    logs: list[dict[str, float]] = []
    for step in range(start_step, config.steps):
        step_start_time = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        lr_value = _lr_for_step(
            step=step,
            total_steps=config.steps,
            base_lr=config.learning_rate,
            scheduler=config.scheduler,
            warmup_steps=config.warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = float(lr_value)
        mask_ratio = _mask_ratio_for_step(
            step=step,
            total_steps=config.steps,
            base_ratio=config.mask_ratio,
            schedule=config.mask_schedule,
            warmup_steps=config.mask_warmup_steps,
        )
        if real_batch_provider is None:
            class_labels = torch.randint(0, config.num_classes, (config.batch_size,), device=device)
            latents = _sample_synthetic_latents(
                batch_size=config.batch_size,
                channels=config.in_channels,
                image_size=config.image_size,
                class_labels=class_labels,
                num_classes=config.num_classes,
                device=device,
            )
        else:
            latents, class_labels = real_batch_provider.next_batch(device=device)  # type: ignore[attr-defined]
            if latents.ndim != 4:
                raise ValueError("real_batch_provider must yield [B, C, H, W] tensors")
            if int(latents.shape[1]) != int(config.in_channels):
                raise ValueError("real_batch_provider channels mismatch vs config.in_channels")
            if int(latents.shape[-1]) != int(config.image_size) or int(latents.shape[-2]) != int(config.image_size):
                raise ValueError("real_batch_provider image_size mismatch vs config.image_size")
        reconstruction, mask, feature_maps = model(latents, mask_ratio=mask_ratio)
        loss, loss_stats = masked_reconstruction_loss(
            reconstruction=reconstruction,
            target=latents,
            mask=mask,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if ema_model is not None:
            ema_model.update(model)
        feature_norm = float(sum(feature.norm().item() for feature in feature_maps) / max(1, len(feature_maps)))
        step_time_s = float(time.perf_counter() - step_start_time)
        peak_cuda_mem_mb = (
            float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
            if device.type == "cuda"
            else 0.0
        )
        if on_step_end is not None:
            on_step_end(step + 1, model, optimizer)

        if step == 0 or (step + 1) % config.log_every == 0 or step + 1 == config.steps:
            with torch.no_grad():
                val_labels = torch.randint(0, config.num_classes, (config.val_batch_size,), device=device)
                val_latents = _sample_synthetic_latents(
                    batch_size=config.val_batch_size,
                    channels=config.in_channels,
                    image_size=config.image_size,
                    class_labels=val_labels,
                    num_classes=config.num_classes,
                    device=device,
                )
                val_reconstruction, val_mask, _ = model(val_latents, mask_ratio=mask_ratio)
                val_loss, _ = masked_reconstruction_loss(
                    reconstruction=val_reconstruction,
                    target=val_latents,
                    mask=val_mask,
                )
            logs.append(
                {
                    "step": float(step + 1),
                    "loss": float(loss.item()),
                    "val_loss": float(val_loss.item()),
                    "masked_mse": loss_stats["masked_mse"],
                    "unmasked_mse": loss_stats["unmasked_mse"],
                    "mask_ratio_realized": loss_stats["mask_ratio_realized"],
                    "mask_ratio_target": float(mask_ratio),
                    "grad_norm": float(grad_norm.item()),
                    "feature_stages": float(len(feature_maps)),
                    "mean_feature_norm": feature_norm,
                    "step_time_s": step_time_s,
                    "peak_cuda_mem_mb": peak_cuda_mem_mb,
                    "lr": float(lr_value),
                }
            )

    summary = {
        "device": str(device),
        "config": config.__dict__,
        "logs": logs,
        "ema_enabled": bool(config.use_ema),
    }
    if logs:
        summary["perf"] = {
            "mean_step_time_s": float(sum(entry["step_time_s"] for entry in logs) / len(logs)),
            "max_peak_cuda_mem_mb": float(max(entry["peak_cuda_mem_mb"] for entry in logs)),
        }
    return summary


def _sample_synthetic_latents(
    *,
    batch_size: int,
    channels: int,
    image_size: int,
    class_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    noise = torch.randn(batch_size, channels, image_size, image_size, device=device)
    class_scalar = class_labels.float() / max(1.0, float(num_classes - 1))
    class_offset = class_scalar.view(batch_size, 1, 1, 1) * 0.2
    return noise + class_offset


def _set_seed(seed: int) -> None:
    seed_everything(seed)


def _mask_ratio_for_step(
    *,
    step: int,
    total_steps: int,
    base_ratio: float,
    schedule: str,
    warmup_steps: int,
) -> float:
    if schedule == "fixed":
        return float(base_ratio)
    if schedule == "linear_warmup":
        warmup = max(1, warmup_steps)
        scale = min(1.0, float(step + 1) / float(warmup))
        return float(base_ratio * scale)
    if schedule == "cosine":
        progress = float(step) / float(max(1, total_steps - 1))
        cosine = 0.5 * (1.0 - math.cos(progress * math.pi))
        return float(base_ratio * cosine)
    raise ValueError(f"Unsupported mask schedule: {schedule}")


def _lr_for_step(
    *,
    step: int,
    total_steps: int,
    base_lr: float,
    scheduler: str,
    warmup_steps: int,
) -> float:
    if scheduler == "none":
        return float(base_lr)
    if scheduler == "cosine":
        progress = float(step) / float(max(1, total_steps - 1))
        return float(base_lr * 0.5 * (1.0 + math.cos(math.pi * progress)))
    if scheduler == "warmup_cosine":
        warmup = max(1, int(warmup_steps))
        if step < warmup:
            return float(base_lr * float(step + 1) / float(warmup))
        remaining = max(1, total_steps - warmup)
        progress = float(step - warmup) / float(max(1, remaining - 1))
        return float(base_lr * 0.5 * (1.0 + math.cos(math.pi * progress)))
    raise ValueError(f"Unsupported scheduler: {scheduler}")
