from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from drifting_models.drift_field import build_negative_log_weights
from drifting_models.drift_loss import (
    DriftingLossConfig,
    FeatureDriftingConfig,
    drifting_stopgrad_loss,
    drifting_stopgrad_loss_multi_temperature,
    feature_space_drifting_loss,
)
from drifting_models.features.vectorize import extract_feature_maps, vectorize_feature_maps
from drifting_models.train.grouped import infer_grouped_shapes


@dataclass(frozen=True)
class GroupedDriftStepConfig:
    loss_config: DriftingLossConfig
    feature_config: FeatureDriftingConfig | None = None
    drift_temperatures: tuple[float, ...] = ()
    drift_temperature_reduction: str = "mean"
    clip_grad_norm: float | None = 2.0
    run_optimizer_step: bool = True


def grouped_drift_training_step(
    *,
    generator: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    noise_grouped: torch.Tensor,
    class_labels_grouped: torch.Tensor,
    alpha_grouped: torch.Tensor,
    positives_grouped: torch.Tensor,
    style_indices_grouped: torch.Tensor | None,
    unconditional_grouped: torch.Tensor | None = None,
    unconditional_weight_grouped: torch.Tensor | None = None,
    feature_extractor: nn.Module | None = None,
    feature_input_transform: nn.Module | None = None,
    amp_dtype: torch.dtype | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
    backward_when_no_step: bool = False,
    loss_divisor: float = 1.0,
    zero_grad_before_backward: bool = True,
    config: GroupedDriftStepConfig,
) -> dict[str, Any]:
    shapes = infer_grouped_shapes(
        x_grouped=noise_grouped.reshape(noise_grouped.shape[0], noise_grouped.shape[1], -1),
        y_pos_grouped=positives_grouped.reshape(positives_grouped.shape[0], positives_grouped.shape[1], -1),
        y_neg_grouped=noise_grouped.reshape(noise_grouped.shape[0], noise_grouped.shape[1], -1),
    )
    groups = shapes.groups
    negatives_per_group = shapes.negatives_per_group
    if unconditional_grouped is not None and unconditional_grouped.ndim != 5:
        raise ValueError("unconditional_grouped must be [G, U, C, H, W]")
    if unconditional_grouped is not None and unconditional_grouped.shape[0] != groups:
        raise ValueError("unconditional_grouped must share group dimension")
    if unconditional_weight_grouped is not None:
        if unconditional_weight_grouped.ndim != 1:
            raise ValueError("unconditional_weight_grouped must be [G]")
        if unconditional_weight_grouped.shape[0] != groups:
            raise ValueError("unconditional_weight_grouped must share group dimension")

    noise_flat = _flatten_grouped_images(noise_grouped)
    class_labels_flat = class_labels_grouped.repeat_interleave(negatives_per_group)
    alpha_flat = alpha_grouped.repeat_interleave(negatives_per_group)
    styles_flat = None
    if style_indices_grouped is not None:
        styles_flat = style_indices_grouped.reshape(
            groups * negatives_per_group,
            style_indices_grouped.shape[-1],
        )

    if amp_dtype is not None:
        autocast_context = torch.autocast(device_type=noise_grouped.device.type, dtype=amp_dtype)
    else:
        autocast_context = nullcontext()

    with autocast_context:
        generated_flat = generator(
            noise_flat,
            class_labels_flat,
            alpha_flat,
            styles_flat,
        )
        generated_grouped = generated_flat.reshape(
            groups,
            negatives_per_group,
            generated_flat.shape[1],
            generated_flat.shape[2],
            generated_flat.shape[3],
        )
        losses = []
        drift_norms = []
        per_group_stats: list[dict[str, float]] = []
        for group_index in range(groups):
            generated_group = generated_grouped[group_index]
            positives_group = positives_grouped[group_index]
            unconditional_group = None if unconditional_grouped is None else unconditional_grouped[group_index]
            unconditional_weight = (
                None
                if unconditional_weight_grouped is None
                else float(unconditional_weight_grouped[group_index].item())
            )
            if config.feature_config is not None:
                if feature_extractor is None:
                    raise ValueError("feature_extractor must be provided when feature_config is enabled")
                loss, stats = _feature_group_loss(
                    generated_group=generated_group,
                    positives_group=positives_group,
                    unconditional_group=unconditional_group,
                    unconditional_weight=unconditional_weight,
                    feature_extractor=feature_extractor,
                    feature_input_transform=feature_input_transform,
                    config=config,
                )
            else:
                generated_vectors = generated_group.reshape(generated_group.shape[0], -1)
                positive_vectors = positives_group.reshape(positives_group.shape[0], -1)
                negative_vectors = generated_vectors
                negative_log_weights = None
                if unconditional_group is not None:
                    unconditional_vectors = unconditional_group.reshape(unconditional_group.shape[0], -1)
                    negative_vectors = torch.cat([generated_vectors, unconditional_vectors], dim=0)
                    weight = 1.0 if unconditional_weight is None else unconditional_weight
                    negative_log_weights = build_negative_log_weights(
                        n_generated_negatives=generated_vectors.shape[0],
                        n_unconditional_negatives=unconditional_vectors.shape[0],
                        unconditional_weight=weight,
                        device=generated_vectors.device,
                        dtype=generated_vectors.dtype,
                    )
                if config.drift_temperatures:
                    loss, stats = drifting_stopgrad_loss_multi_temperature(
                        x=generated_vectors,
                        y_pos=positive_vectors,
                        y_neg=negative_vectors,
                        temperatures=tuple(config.drift_temperatures),
                        config=config.loss_config,
                        negative_log_weights=negative_log_weights,
                        generated_negative_count=generated_vectors.shape[0],
                        reduction=str(config.drift_temperature_reduction),
                    )
                else:
                    loss, _, stats = drifting_stopgrad_loss(
                        x=generated_vectors,
                        y_pos=positive_vectors,
                        y_neg=negative_vectors,
                        config=config.loss_config,
                        negative_log_weights=negative_log_weights,
                        generated_negative_count=generated_vectors.shape[0],
                    )
            losses.append(loss)
            drift_norms.append(stats["mean_drift_norm"] if "mean_drift_norm" in stats else stats["drift_norm"])
            per_group_stats.append(stats)

        total_loss = torch.stack(losses).mean()

    scaled_loss = total_loss / max(float(loss_divisor), 1e-8)
    if config.run_optimizer_step:
        if optimizer is None:
            raise ValueError("optimizer must be provided when run_optimizer_step is True")
        if zero_grad_before_backward:
            optimizer.zero_grad(set_to_none=True)
        if grad_scaler is not None:
            grad_scaler.scale(scaled_loss).backward()
            grad_norm = None
            if config.clip_grad_norm is not None:
                grad_scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), config.clip_grad_norm)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            scaled_loss.backward()
            grad_norm = None
            if config.clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), config.clip_grad_norm)
            optimizer.step()
    else:
        grad_norm = None
        if backward_when_no_step:
            if grad_scaler is not None:
                grad_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

    aggregated_feature_stats = _aggregate_group_stats(per_group_stats)
    result = {
        "loss": float(total_loss.item()),
        "mean_drift_norm": float(sum(drift_norms) / len(drift_norms)),
        "groups": groups,
        "negatives_per_group": negatives_per_group,
        "alpha_mean": float(alpha_grouped.mean().item()),
        "alpha_min": float(alpha_grouped.min().item()),
        "alpha_max": float(alpha_grouped.max().item()),
        "grad_norm": None if grad_norm is None else float(grad_norm.item()),
    }
    if unconditional_weight_grouped is not None:
        result["mean_unconditional_weight"] = float(unconditional_weight_grouped.mean().item())
        result["mean_unconditional_negative_fraction"] = _mean_unconditional_negative_fraction(
            negatives_per_group=negatives_per_group,
            unconditional_grouped=unconditional_grouped,
            unconditional_weight_grouped=unconditional_weight_grouped,
        )
    result.update(aggregated_feature_stats)
    return result


def _flatten_grouped_images(grouped_images: torch.Tensor) -> torch.Tensor:
    if grouped_images.ndim != 5:
        raise ValueError(f"grouped_images must be [G, N, C, H, W], got {tuple(grouped_images.shape)}")
    groups, items, channels, height, width = grouped_images.shape
    return grouped_images.reshape(groups * items, channels, height, width)


def _feature_group_loss(
    *,
    generated_group: torch.Tensor,
    positives_group: torch.Tensor,
    unconditional_group: torch.Tensor | None,
    unconditional_weight: float | None,
    feature_extractor: nn.Module,
    feature_input_transform: nn.Module | None,
    config: GroupedDriftStepConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    if config.feature_config is None:
        raise ValueError("feature_config is required for feature group loss")
    generated_inputs = (
        generated_group
        if feature_input_transform is None
        else feature_input_transform(generated_group)
    )
    positives_inputs = (
        positives_group
        if feature_input_transform is None
        else feature_input_transform(positives_group)
    )
    unconditional_inputs = (
        None
        if unconditional_group is None
        else (
            unconditional_group
            if feature_input_transform is None
            else feature_input_transform(unconditional_group)
        )
    )
    generated_maps = extract_feature_maps(encoder=feature_extractor, images=generated_inputs)
    if config.feature_config.detach_positive_features:
        with torch.no_grad():
            positive_maps = extract_feature_maps(encoder=feature_extractor, images=positives_inputs)
            unconditional_maps = (
                None
                if unconditional_group is None
                else extract_feature_maps(encoder=feature_extractor, images=unconditional_inputs)
            )
    else:
        positive_maps = extract_feature_maps(encoder=feature_extractor, images=positives_inputs)
        unconditional_maps = (
            None if unconditional_group is None else extract_feature_maps(encoder=feature_extractor, images=unconditional_inputs)
        )

    generated_vectors = vectorize_feature_maps(
        generated_maps,
        config=config.feature_config.vectorization,
        input_images=generated_inputs,
    )
    positive_vectors = vectorize_feature_maps(
        positive_maps,
        config=config.feature_config.vectorization,
        input_images=positives_inputs,
    )
    unconditional_vectors = (
        None
        if unconditional_maps is None
        else vectorize_feature_maps(
            unconditional_maps,
            config=config.feature_config.vectorization,
            input_images=unconditional_inputs,
        )
    )
    feature_loss, feature_stats = feature_space_drifting_loss(
        generated_feature_vectors=generated_vectors,
        positive_feature_vectors=positive_vectors,
        unconditional_feature_vectors=unconditional_vectors,
        base_loss_config=config.loss_config,
        feature_config=config.feature_config,
        unconditional_weight=1.0 if unconditional_weight is None else unconditional_weight,
    )

    raw_loss = None
    raw_stats: dict[str, float] = {}
    if config.feature_config.include_raw_drift_loss:
        raw_loss, raw_stats = _vanilla_group_loss(
            generated_group=generated_group,
            positives_group=positives_group,
            unconditional_group=unconditional_group,
            unconditional_weight=unconditional_weight,
            config=config,
        )
        raw_weight = float(config.feature_config.raw_drift_loss_weight)
        feature_loss = feature_loss + raw_loss * raw_weight
        feature_stats = {
            **feature_stats,
            "feature_loss": float(feature_stats.get("loss", 0.0)),
            "raw_loss": float(raw_stats.get("loss", float(raw_loss.item()))),
            "raw_loss_weight": float(raw_weight),
            "loss": float(feature_loss.item()),
            **{f"raw_{k}": float(v) for k, v in raw_stats.items() if isinstance(v, (int, float)) and k != "loss"},
        }
    return feature_loss, feature_stats


def _vanilla_group_loss(
    *,
    generated_group: torch.Tensor,
    positives_group: torch.Tensor,
    unconditional_group: torch.Tensor | None,
    unconditional_weight: float | None,
    config: GroupedDriftStepConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    generated_vectors = generated_group.reshape(generated_group.shape[0], -1)
    positive_vectors = positives_group.reshape(positives_group.shape[0], -1)
    negative_vectors = generated_vectors
    negative_log_weights = None
    if unconditional_group is not None:
        unconditional_vectors = unconditional_group.reshape(unconditional_group.shape[0], -1)
        negative_vectors = torch.cat([generated_vectors, unconditional_vectors], dim=0)
        weight = 1.0 if unconditional_weight is None else unconditional_weight
        negative_log_weights = build_negative_log_weights(
            n_generated_negatives=generated_vectors.shape[0],
            n_unconditional_negatives=unconditional_vectors.shape[0],
            unconditional_weight=weight,
            device=generated_vectors.device,
            dtype=generated_vectors.dtype,
        )
    if config.drift_temperatures:
        loss, stats = drifting_stopgrad_loss_multi_temperature(
            x=generated_vectors,
            y_pos=positive_vectors,
            y_neg=negative_vectors,
            temperatures=tuple(config.drift_temperatures),
            config=config.loss_config,
            negative_log_weights=negative_log_weights,
            generated_negative_count=generated_vectors.shape[0],
            reduction=str(config.drift_temperature_reduction),
        )
        return loss, stats
    loss, _, stats = drifting_stopgrad_loss(
        x=generated_vectors,
        y_pos=positive_vectors,
        y_neg=negative_vectors,
        config=config.loss_config,
        negative_log_weights=negative_log_weights,
        generated_negative_count=generated_vectors.shape[0],
    )
    return loss, stats


def _aggregate_group_stats(stats_per_group: list[dict[str, float]]) -> dict[str, float]:
    if not stats_per_group:
        return {}
    numeric_values: dict[str, list[float]] = {}
    for entry in stats_per_group:
        for key, value in entry.items():
            if isinstance(value, (int, float)):
                numeric_values.setdefault(key, []).append(float(value))
    aggregated = {}
    for key, values in numeric_values.items():
        if key in {"loss", "drift_norm", "mean_drift_norm"}:
            continue
        aggregated[key] = float(sum(values) / len(values))
    return aggregated


def _mean_unconditional_negative_fraction(
    *,
    negatives_per_group: int,
    unconditional_grouped: torch.Tensor | None,
    unconditional_weight_grouped: torch.Tensor,
) -> float:
    if unconditional_grouped is None or unconditional_grouped.shape[1] == 0:
        return 0.0
    if negatives_per_group <= 1:
        return 0.0
    fractions = []
    for group_index in range(unconditional_weight_grouped.shape[0]):
        weight = float(unconditional_weight_grouped[group_index].item())
        n_generated = negatives_per_group - 1
        n_unconditional = int(unconditional_grouped.shape[1])
        numerator = n_unconditional * weight
        denominator = n_generated + numerator
        fraction = 0.0 if denominator <= 0.0 else numerator / denominator
        fractions.append(fraction)
    return float(sum(fractions) / len(fractions))
