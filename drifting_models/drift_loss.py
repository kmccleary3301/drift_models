from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import sqrt

import torch
import torch.nn.functional as F

from drifting_models.features.vectorize import FeatureVectorizationConfig
from drifting_models.drift_field import (
    DriftFieldConfig,
    build_negative_log_weights,
    compute_drift_components,
)

_COMPILED_MULTI_TEMP_KERNELS: dict[tuple[str, str, bool, bool], object] = {}
_COMPILE_MULTI_TEMP_FAILURES: dict[tuple[str, str, bool, bool], str] = {}


@dataclass(frozen=True)
class DriftingLossConfig:
    drift_field: DriftFieldConfig
    attraction_scale: float = 1.0
    repulsion_scale: float = 1.0
    stopgrad_target: bool = True


def compute_weighted_drift(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    *,
    config: DriftingLossConfig,
    negative_log_weights: torch.Tensor | None = None,
    generated_negative_count: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    drift_pos, drift_neg = compute_drift_components(
        x=x,
        y_pos=y_pos,
        y_neg=y_neg,
        config=config.drift_field,
        negative_log_weights=negative_log_weights,
        generated_negative_count=generated_negative_count,
    )
    drift = (config.attraction_scale * drift_pos) - (config.repulsion_scale * drift_neg)
    stats = {
        "drift_norm": float(drift.norm(dim=-1).mean().item()),
        "drift_pos_norm": float(drift_pos.norm(dim=-1).mean().item()),
        "drift_neg_norm": float(drift_neg.norm(dim=-1).mean().item()),
        "attraction_scale": float(config.attraction_scale),
        "repulsion_scale": float(config.repulsion_scale),
    }
    return drift, stats


def drifting_stopgrad_loss(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    *,
    config: DriftingLossConfig,
    negative_log_weights: torch.Tensor | None = None,
    generated_negative_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    drift, stats = compute_weighted_drift(
        x=x,
        y_pos=y_pos,
        y_neg=y_neg,
        config=config,
        negative_log_weights=negative_log_weights,
        generated_negative_count=generated_negative_count,
    )
    target = x + drift
    if config.stopgrad_target:
        target = target.detach()
    loss = F.mse_loss(x, target)
    stats["loss"] = float(loss.item())
    return loss, drift, stats


def drifting_stopgrad_loss_multi_temperature(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    *,
    temperatures: tuple[float, ...],
    config: DriftingLossConfig,
    negative_log_weights: torch.Tensor | None = None,
    generated_negative_count: int | None = None,
    reduction: str = "mean",
) -> tuple[torch.Tensor, dict[str, float]]:
    if not temperatures:
        raise ValueError("temperatures must be non-empty")
    normalized_reduction = reduction.strip().lower()
    if normalized_reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")

    loss_terms: list[torch.Tensor] = []
    stats: dict[str, float] = {}
    drift_norms: list[float] = []
    drift_pos_norms: list[float] = []
    drift_neg_norms: list[float] = []
    for temperature in temperatures:
        temp = float(temperature)
        loss_cfg = replace(config, drift_field=replace(config.drift_field, temperature=temp))
        loss, _, loss_stats = drifting_stopgrad_loss(
            x=x,
            y_pos=y_pos,
            y_neg=y_neg,
            config=loss_cfg,
            negative_log_weights=negative_log_weights,
            generated_negative_count=generated_negative_count,
        )
        loss_terms.append(loss)
        drift_norms.append(float(loss_stats["drift_norm"]))
        drift_pos_norms.append(float(loss_stats["drift_pos_norm"]))
        drift_neg_norms.append(float(loss_stats["drift_neg_norm"]))
        stats[f"temp_{temp:g}_drift_norm"] = float(loss_stats["drift_norm"])
        stats[f"temp_{temp:g}_drift_pos_norm"] = float(loss_stats["drift_pos_norm"])
        stats[f"temp_{temp:g}_drift_neg_norm"] = float(loss_stats["drift_neg_norm"])

    stacked = torch.stack(loss_terms)
    total_loss = stacked.mean() if normalized_reduction == "mean" else stacked.sum()
    stats["loss"] = float(total_loss.item())
    stats["temperature_count"] = float(len(temperatures))
    stats["mean_drift_norm"] = float(sum(drift_norms) / max(1, len(drift_norms)))
    stats["mean_drift_pos_norm"] = float(sum(drift_pos_norms) / max(1, len(drift_pos_norms)))
    stats["mean_drift_neg_norm"] = float(sum(drift_neg_norms) / max(1, len(drift_neg_norms)))
    return total_loss, stats


@dataclass(frozen=True)
class FeatureDriftingConfig:
    temperatures: tuple[float, ...] = (0.05,)
    vectorization: FeatureVectorizationConfig = field(default_factory=FeatureVectorizationConfig)
    normalize_features: bool = True
    normalize_drifts: bool = True
    scale_temperature_by_sqrt_channels: bool = True
    detach_positive_features: bool = True
    detach_negative_features: bool = True
    share_location_normalization: bool = True
    include_raw_drift_loss: bool = False
    raw_drift_loss_weight: float = 1.0
    temperature_aggregation: str = "per_temperature_mse"  # or "sum_drifts_then_mse"
    loss_term_reduction: str = "sum"  # or "mean"
    compile_drift_kernel: bool = False
    compile_drift_backend: str = "inductor"
    compile_drift_mode: str = "reduce-overhead"
    compile_drift_dynamic: bool = False
    compile_drift_fullgraph: bool = False
    compile_drift_fail_action: str = "warn"  # or "raise"
    eps: float = 1e-8


def feature_space_drifting_loss(
    *,
    generated_feature_vectors: dict[str, torch.Tensor],
    positive_feature_vectors: dict[str, torch.Tensor],
    unconditional_feature_vectors: dict[str, torch.Tensor] | None,
    base_loss_config: DriftingLossConfig,
    feature_config: FeatureDriftingConfig,
    unconditional_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    aggregation = feature_config.temperature_aggregation.strip().lower()
    if aggregation not in {"per_temperature_mse", "sum_drifts_then_mse"}:
        raise ValueError("FeatureDriftingConfig.temperature_aggregation must be 'per_temperature_mse' or 'sum_drifts_then_mse'")
    loss_term_reduction = feature_config.loss_term_reduction.strip().lower()
    if loss_term_reduction not in {"sum", "mean"}:
        raise ValueError("FeatureDriftingConfig.loss_term_reduction must be 'sum' or 'mean'")
    compile_fail_action = feature_config.compile_drift_fail_action.strip().lower()
    if compile_fail_action not in {"warn", "raise"}:
        raise ValueError("FeatureDriftingConfig.compile_drift_fail_action must be 'warn' or 'raise'")
    drift_kernel = _resolve_multi_temp_drift_kernel(feature_config=feature_config)
    shared_keys = sorted(set(generated_feature_vectors.keys()) & set(positive_feature_vectors.keys()))
    if not shared_keys:
        raise ValueError("No shared feature keys between generated and positive features")

    loss_terms: list[torch.Tensor] = []
    drift_norms: list[float] = []
    vector_count = 0
    feature_scales: list[float] = []
    drift_scales: list[float] = []
    temperature_drift_norms: dict[float, list[float]] = {
        float(temperature): [] for temperature in feature_config.temperatures
    }
    temperature_drift_scales: dict[float, list[float]] = {
        float(temperature): [] for temperature in feature_config.temperatures
    }

    for key in shared_keys:
        x_vectors = generated_feature_vectors[key]
        y_vectors = positive_feature_vectors[key]
        if x_vectors.ndim != 3 or y_vectors.ndim != 3:
            raise ValueError("Feature vectors must be [B, V, C]")
        if x_vectors.shape[1] != y_vectors.shape[1] or x_vectors.shape[2] != y_vectors.shape[2]:
            raise ValueError(f"Feature vector shape mismatch for key '{key}'")

        x_current = x_vectors
        y_current = y_vectors
        y_uncurrent = None
        if unconditional_feature_vectors is not None and key in unconditional_feature_vectors:
            y_uncurrent = unconditional_feature_vectors[key]
            if y_uncurrent.ndim != 3:
                raise ValueError("unconditional feature vectors must be [B, V, C]")
            if y_uncurrent.shape[1] != x_current.shape[1] or y_uncurrent.shape[2] != x_current.shape[2]:
                raise ValueError(f"Unconditional feature vector shape mismatch for key '{key}'")

        if feature_config.normalize_features:
            y_neg_for_scale = x_current.detach() if feature_config.detach_negative_features else x_current
            negative_log_weights_for_scale = None
            if y_uncurrent is not None:
                y_uncached = y_uncurrent.detach()
                y_neg_for_scale = torch.cat([y_neg_for_scale, y_uncached], dim=0)
                negative_log_weights_for_scale = build_negative_log_weights(
                    n_generated_negatives=x_current.shape[0],
                    n_unconditional_negatives=y_uncached.shape[0],
                    unconditional_weight=unconditional_weight,
                    device=x_current.device,
                    dtype=x_current.dtype,
                )
            x_current, y_current, scales = _normalize_features(
                x_current,
                y_current,
                y_neg_vectors=y_neg_for_scale,
                negative_log_weights=negative_log_weights_for_scale,
                share_location_normalization=feature_config.share_location_normalization,
                eps=feature_config.eps,
            )
            feature_scales.extend(scales.tolist())
            if y_uncurrent is not None:
                y_uncurrent = _apply_feature_scales(y_uncurrent, scales)

        if feature_config.detach_positive_features:
            y_current = y_current.detach()

        vector_slots = x_current.shape[1]
        vector_count += vector_slots
        drift_sum_full = torch.zeros_like(x_current) if aggregation == "sum_drifts_then_mse" else None
        y_neg_generated = x_current.detach() if feature_config.detach_negative_features else x_current
        y_neg_current = y_neg_generated
        negative_log_weights = None
        if y_uncurrent is not None:
            y_uncached = y_uncurrent.detach()
            y_neg_current = torch.cat([y_neg_generated, y_uncached], dim=0)
            negative_log_weights = build_negative_log_weights(
                n_generated_negatives=y_neg_generated.shape[0],
                n_unconditional_negatives=y_uncached.shape[0],
                unconditional_weight=unconditional_weight,
                device=x_current.device,
                dtype=x_current.dtype,
            )
        drift_tau_pairs = drift_kernel(
            x_vectors=x_current,
            y_pos_vectors=y_current,
            y_neg_vectors=y_neg_current,
            temperatures=feature_config.temperatures,
            base_config=base_loss_config,
            scale_temperature_by_sqrt_channels=feature_config.scale_temperature_by_sqrt_channels,
            negative_log_weights=negative_log_weights,
            generated_negative_count=y_neg_generated.shape[0],
        )
        for temperature_value, drift_tau_full in drift_tau_pairs:
            if drift_tau_full.dtype != x_current.dtype:
                drift_tau_full = drift_tau_full.to(dtype=x_current.dtype)
            if drift_tau_full.device != x_current.device:
                drift_tau_full = drift_tau_full.to(device=x_current.device)
            if drift_tau_full.shape != x_current.shape:
                raise ValueError("slot-batched drift output shape mismatch")
            if not torch.isfinite(drift_tau_full).all():
                raise ValueError("slot-batched drift produced non-finite values")
            if feature_config.normalize_drifts:
                drift_tau_full, scale_tensor = _normalize_drifts(
                    drift_tau_full,
                    share_location_normalization=feature_config.share_location_normalization,
                    eps=feature_config.eps,
                )
                drift_scales.extend(scale_tensor.tolist())
                temperature_drift_scales[temperature_value].extend(scale_tensor.tolist())
            temperature_drift_norms[temperature_value].append(
                float(drift_tau_full.norm(dim=-1).mean().item())
            )
            if drift_sum_full is not None:
                drift_sum_full = drift_sum_full + drift_tau_full
            else:
                for vector_index in range(vector_slots):
                    x_vec = x_current[:, vector_index, :]
                    target = x_vec + drift_tau_full[:, vector_index, :]
                    if base_loss_config.stopgrad_target:
                        target = target.detach()
                    loss_terms.append(F.mse_loss(x_vec, target))
                drift_norms.append(float(drift_tau_full.norm(dim=-1).mean().item()))

        if drift_sum_full is not None:
            for vector_index in range(vector_slots):
                x_vec = x_current[:, vector_index, :]
                target = x_vec + drift_sum_full[:, vector_index, :]
                if base_loss_config.stopgrad_target:
                    target = target.detach()
                loss_terms.append(F.mse_loss(x_vec, target))
            drift_norms.append(float(drift_sum_full.norm(dim=-1).mean().item()))

    stacked_loss_terms = torch.stack(loss_terms)
    total_loss = stacked_loss_terms.mean() if loss_term_reduction == "mean" else stacked_loss_terms.sum()
    stats = {
        "loss": float(total_loss.item()),
        "loss_term_count": float(len(loss_terms)),
        "feature_keys": float(len(shared_keys)),
        "vector_count": float(vector_count),
        "mean_drift_norm": float(sum(drift_norms) / max(1, len(drift_norms))),
        "mean_feature_scale": float(sum(feature_scales) / max(1, len(feature_scales))),
        "mean_drift_scale": float(sum(drift_scales) / max(1, len(drift_scales))),
    }
    for temperature, values in temperature_drift_norms.items():
        key = f"temp_{temperature:g}_drift_norm"
        stats[key] = float(sum(values) / max(1, len(values)))
    for temperature, values in temperature_drift_scales.items():
        key = f"temp_{temperature:g}_drift_scale"
        stats[key] = float(sum(values) / max(1, len(values)))
    return total_loss, stats


def _resolve_multi_temp_drift_kernel(
    *,
    feature_config: FeatureDriftingConfig,
):
    fail_action = feature_config.compile_drift_fail_action.strip().lower()
    if not feature_config.compile_drift_kernel:
        return _compute_weighted_drifts_slot_batched_multi_temperature
    key = (
        str(feature_config.compile_drift_backend),
        str(feature_config.compile_drift_mode),
        bool(feature_config.compile_drift_dynamic),
        bool(feature_config.compile_drift_fullgraph),
    )
    cached = _COMPILED_MULTI_TEMP_KERNELS.get(key)
    if cached is not None:
        return cached
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        message = "torch.compile is unavailable for feature drift kernel compilation"
        if fail_action == "raise":
            raise RuntimeError(message)
        _COMPILE_MULTI_TEMP_FAILURES[key] = message
        _COMPILED_MULTI_TEMP_KERNELS[key] = _compute_weighted_drifts_slot_batched_multi_temperature
        return _COMPILED_MULTI_TEMP_KERNELS[key]
    try:
        compiled = compile_fn(
            _compute_weighted_drifts_slot_batched_multi_temperature,
            backend=str(feature_config.compile_drift_backend),
            mode=str(feature_config.compile_drift_mode),
            dynamic=bool(feature_config.compile_drift_dynamic),
            fullgraph=bool(feature_config.compile_drift_fullgraph),
        )
    except Exception as error:
        if fail_action == "raise":
            raise
        _COMPILE_MULTI_TEMP_FAILURES[key] = str(error)
        compiled = _compute_weighted_drifts_slot_batched_multi_temperature
    _COMPILED_MULTI_TEMP_KERNELS[key] = compiled
    return compiled


def _compute_weighted_drifts_slot_batched_multi_temperature(
    *,
    x_vectors: torch.Tensor,
    y_pos_vectors: torch.Tensor,
    y_neg_vectors: torch.Tensor,
    temperatures: tuple[float, ...],
    base_config: DriftingLossConfig,
    scale_temperature_by_sqrt_channels: bool,
    negative_log_weights: torch.Tensor | None,
    generated_negative_count: int | None,
) -> list[tuple[float, torch.Tensor]]:
    if not temperatures:
        raise ValueError("temperatures must be non-empty")
    if x_vectors.ndim != 3 or y_pos_vectors.ndim != 3 or y_neg_vectors.ndim != 3:
        raise ValueError("slot-batched drift inputs must be [B, V, C]")
    if x_vectors.shape[1:] != y_pos_vectors.shape[1:] or x_vectors.shape[1:] != y_neg_vectors.shape[1:]:
        raise ValueError("slot-batched drift inputs must share [V, C] dimensions")
    if negative_log_weights is not None and negative_log_weights.shape[0] != y_neg_vectors.shape[0]:
        raise ValueError("negative_log_weights size must match y_neg_vectors batch dimension")

    x_slot_major = x_vectors.permute(1, 0, 2)
    y_pos_slot_major = y_pos_vectors.permute(1, 0, 2)
    y_neg_slot_major = y_neg_vectors.permute(1, 0, 2)

    x_compute = x_slot_major
    y_pos_compute = y_pos_slot_major
    y_neg_compute = y_neg_slot_major
    if x_compute.device.type == "cpu" and x_compute.dtype in {torch.float16, torch.bfloat16}:
        x_compute = x_compute.float()
        y_pos_compute = y_pos_compute.float()
        y_neg_compute = y_neg_compute.float()

    dist_pos = torch.cdist(x_compute, y_pos_compute)
    dist_neg = torch.cdist(x_compute, y_neg_compute)

    generated_count = generated_negative_count if generated_negative_count is not None else y_neg_compute.shape[1]
    if base_config.drift_field.mask_self_negatives and generated_count > 0:
        diag_count = min(x_compute.shape[1], generated_count, y_neg_compute.shape[1])
        if diag_count > 0:
            diagonal = torch.arange(diag_count, device=dist_neg.device)
            dist_neg = dist_neg.clone()
            dist_neg[:, diagonal, diagonal] = dist_neg[:, diagonal, diagonal] + base_config.drift_field.self_mask_value

    log_weights = None
    if negative_log_weights is not None:
        log_weights = negative_log_weights
        if log_weights.device != dist_neg.device:
            log_weights = log_weights.to(device=dist_neg.device)
        if log_weights.dtype != dist_neg.dtype:
            log_weights = log_weights.to(dtype=dist_neg.dtype)

    drift_pairs: list[tuple[float, torch.Tensor]] = []
    for temperature in temperatures:
        temperature_value = float(temperature)
        effective_temperature = temperature_value
        if scale_temperature_by_sqrt_channels:
            effective_temperature = temperature_value * sqrt(float(x_vectors.shape[-1]))
        logit_pos = -(dist_pos / effective_temperature)
        logit_neg = -(dist_neg / effective_temperature)
        if log_weights is not None:
            logit_neg = logit_neg + log_weights.view(1, 1, -1)
        logits = torch.cat([logit_pos, logit_neg], dim=2)
        row_affinity = torch.softmax(logits, dim=-1)
        if base_config.drift_field.normalize_over_x:
            col_affinity = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt(torch.clamp(row_affinity * col_affinity, min=base_config.drift_field.eps))
        else:
            affinity = row_affinity

        n_pos = y_pos_compute.shape[1]
        affinity_pos = affinity[:, :, :n_pos]
        affinity_neg = affinity[:, :, n_pos:]
        weight_pos = affinity_pos * affinity_neg.sum(dim=2, keepdim=True)
        weight_neg = affinity_neg * affinity_pos.sum(dim=2, keepdim=True)

        drift_pos = weight_pos @ y_pos_compute
        drift_neg = weight_neg @ y_neg_compute
        drift = (base_config.attraction_scale * drift_pos) - (base_config.repulsion_scale * drift_neg)
        drift = drift.permute(1, 0, 2)
        if drift.dtype != x_vectors.dtype:
            drift = drift.to(dtype=x_vectors.dtype)
        drift_pairs.append((temperature_value, drift))
    return drift_pairs


def _compute_weighted_drift_slot_batched(
    *,
    x_vectors: torch.Tensor,
    y_pos_vectors: torch.Tensor,
    y_neg_vectors: torch.Tensor,
    config: DriftingLossConfig,
    negative_log_weights: torch.Tensor | None,
    generated_negative_count: int | None,
) -> torch.Tensor:
    drift_pairs = _compute_weighted_drifts_slot_batched_multi_temperature(
        x_vectors=x_vectors,
        y_pos_vectors=y_pos_vectors,
        y_neg_vectors=y_neg_vectors,
        temperatures=(float(config.drift_field.temperature),),
        base_config=config,
        scale_temperature_by_sqrt_channels=False,
        negative_log_weights=negative_log_weights,
        generated_negative_count=generated_negative_count,
    )
    return drift_pairs[0][1]


def _normalize_features(
    x_vectors: torch.Tensor,
    y_vectors: torch.Tensor,
    *,
    y_neg_vectors: torch.Tensor | None = None,
    negative_log_weights: torch.Tensor | None = None,
    share_location_normalization: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize feature vectors by a stop-grad distance scale.

    Per slot, the base scale is mean pairwise distance between generated vectors
    `x_vectors` and positive vectors `y_vectors`. When `y_neg_vectors` is provided,
    the scale uses a weighted blend of positive and negative distances:

        D = (N_pos * D_pos + W_neg * D_neg) / (N_pos + W_neg)

    where `W_neg` is either `N_neg` (uniform negatives) or `sum(exp(logw))`
    from `negative_log_weights`. The final slot/shared scale is divided by
    `sqrt(C)` and detached before application.
    """
    channels = x_vectors.shape[-1]
    distance_tensor = _slotwise_mean_feature_distances(x_vectors, y_vectors)
    if y_neg_vectors is not None:
        neg_distance_tensor = _slotwise_mean_feature_distances(
            x_vectors,
            y_neg_vectors,
            negative_log_weights=negative_log_weights,
        )
        positive_weight = float(y_vectors.shape[0])
        if negative_log_weights is None:
            negative_weight = float(y_neg_vectors.shape[0])
        else:
            negative_weight = float(torch.exp(negative_log_weights).sum().item())
        total_weight = max(positive_weight + negative_weight, eps)
        distance_tensor = ((distance_tensor * positive_weight) + (neg_distance_tensor * negative_weight)) / total_weight
    if share_location_normalization:
        distance = distance_tensor.mean()
        scale = torch.clamp(distance / sqrt(float(channels)), min=eps).detach()
        scale_tensor = scale.repeat(x_vectors.shape[1])
        return x_vectors / scale, y_vectors / scale, scale_tensor

    scale_tensor = torch.clamp(distance_tensor / sqrt(float(channels)), min=eps).detach()
    scale_view = scale_tensor.view(1, -1, 1)
    return x_vectors / scale_view, y_vectors / scale_view, scale_tensor


def _slotwise_mean_feature_distances(
    x_vectors: torch.Tensor,
    y_vectors: torch.Tensor,
    *,
    negative_log_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if x_vectors.ndim != 3 or y_vectors.ndim != 3:
        raise ValueError("feature vectors must be [B, V, C]")
    if x_vectors.shape[1:] != y_vectors.shape[1:]:
        raise ValueError("x_vectors and y_vectors must share [V, C] dimensions")
    if negative_log_weights is not None and negative_log_weights.shape[0] != y_vectors.shape[0]:
        raise ValueError("negative_log_weights size must match y_vectors batch dimension")
    x_slot_major = x_vectors.permute(1, 0, 2)
    y_slot_major = y_vectors.permute(1, 0, 2)
    x_compute = x_slot_major
    y_compute = y_slot_major
    if x_compute.device.type == "cpu" and x_compute.dtype in {torch.float16, torch.bfloat16}:
        x_compute = x_compute.float()
        y_compute = y_compute.float()
    distances = torch.cdist(x_compute, y_compute)
    if negative_log_weights is None:
        return distances.mean(dim=(1, 2))
    weights = torch.exp(negative_log_weights)
    if weights.device != distances.device:
        weights = weights.to(device=distances.device)
    if weights.dtype != distances.dtype:
        weights = weights.to(dtype=distances.dtype)
    weight_sum = torch.clamp(weights.sum(), min=torch.finfo(distances.dtype).tiny)
    weighted_per_x = (distances * weights.view(1, 1, -1)).sum(dim=2) / weight_sum
    return weighted_per_x.mean(dim=1)


def _drift_scale(drift: torch.Tensor, *, eps: float) -> torch.Tensor:
    channels = drift.shape[-1]
    scale = torch.sqrt(torch.mean(torch.sum(drift.pow(2), dim=-1) / float(channels)))
    return torch.clamp(scale, min=eps).detach()


def _normalize_drifts(
    drifts: torch.Tensor,
    *,
    share_location_normalization: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if drifts.ndim != 3:
        raise ValueError("drifts must be [B, V, C]")
    if share_location_normalization:
        scale = _drift_scale(drifts.reshape(-1, drifts.shape[-1]), eps=eps)
        scale_tensor = scale.repeat(drifts.shape[1])
        return drifts / scale, scale_tensor

    scales = []
    for vector_index in range(drifts.shape[1]):
        scale = _drift_scale(drifts[:, vector_index, :], eps=eps)
        scales.append(scale)
    scale_tensor = torch.stack(scales, dim=0)
    scale_view = scale_tensor.view(1, -1, 1)
    return drifts / scale_view, scale_tensor


def _apply_feature_scales(vectors: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if vectors.ndim != 3:
        raise ValueError("vectors must be [B, V, C]")
    if scales.ndim != 1:
        raise ValueError("scales must be [V]")
    if vectors.shape[1] != scales.shape[0]:
        raise ValueError("scale count must match vector slots")
    return vectors / scales.view(1, -1, 1)
