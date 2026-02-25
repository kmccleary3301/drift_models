from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DriftFieldConfig:
    temperature: float = 0.05
    normalize_over_x: bool = True
    mask_self_negatives: bool = True
    self_mask_value: float = 1e6
    eps: float = 1e-12


def cfg_alpha_to_unconditional_weight(
    alpha: float,
    n_generated_negatives: int,
    n_unconditional_negatives: int,
) -> float:
    if alpha < 1.0:
        raise ValueError("alpha must be >= 1.0")
    if n_generated_negatives <= 1:
        raise ValueError("n_generated_negatives must be > 1")
    if n_unconditional_negatives <= 0:
        raise ValueError("n_unconditional_negatives must be > 0")
    return ((alpha - 1.0) * (n_generated_negatives - 1)) / n_unconditional_negatives


def build_negative_log_weights(
    n_generated_negatives: int,
    n_unconditional_negatives: int,
    unconditional_weight: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if n_generated_negatives <= 0:
        raise ValueError("n_generated_negatives must be > 0")
    if n_unconditional_negatives < 0:
        raise ValueError("n_unconditional_negatives must be >= 0")
    if unconditional_weight < 0.0:
        raise ValueError("unconditional_weight must be >= 0")

    generated = torch.zeros(n_generated_negatives, device=device, dtype=dtype)
    if n_unconditional_negatives == 0:
        return generated

    if unconditional_weight == 0.0:
        # Use a very negative finite value instead of -inf.
        # Column-wise softmax over all -inf would produce NaNs when
        # normalize_over_x is enabled; finfo.min makes the unconditional
        # logits effectively ignored without destabilizing softmax.
        unconditional = torch.full(
            (n_unconditional_negatives,),
            torch.finfo(dtype).min,
            device=device,
            dtype=dtype,
        )
    else:
        unconditional = torch.full(
            (n_unconditional_negatives,),
            torch.log(torch.tensor(unconditional_weight, device=device, dtype=dtype)),
            device=device,
            dtype=dtype,
        )
    return torch.cat([generated, unconditional], dim=0)


def compute_affinity_matrices(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    *,
    config: DriftFieldConfig,
    negative_log_weights: torch.Tensor | None = None,
    generated_negative_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(
        x=x,
        y_pos=y_pos,
        y_neg=y_neg,
        negative_log_weights=negative_log_weights,
        generated_negative_count=generated_negative_count,
    )

    dist_pos = torch.cdist(x, y_pos)
    dist_neg = torch.cdist(x, y_neg)

    generated_count = generated_negative_count if generated_negative_count is not None else y_neg.shape[0]
    if config.mask_self_negatives and generated_count > 0:
        diag_count = min(x.shape[0], generated_count, y_neg.shape[0])
        diagonal = torch.arange(diag_count, device=x.device)
        dist_neg = dist_neg.clone()
        dist_neg[diagonal, diagonal] = dist_neg[diagonal, diagonal] + config.self_mask_value

    logit_pos = -(dist_pos / config.temperature)
    logit_neg = -(dist_neg / config.temperature)
    if negative_log_weights is not None:
        logit_neg = logit_neg + negative_log_weights.view(1, -1)

    logits = torch.cat([logit_pos, logit_neg], dim=1)
    row_affinity = torch.softmax(logits, dim=-1)

    if config.normalize_over_x:
        col_affinity = torch.softmax(logits, dim=-2)
        affinity = torch.sqrt(torch.clamp(row_affinity * col_affinity, min=config.eps))
    else:
        affinity = row_affinity

    n_pos = y_pos.shape[0]
    return affinity[:, :n_pos], affinity[:, n_pos:]


def compute_v(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    *,
    config: DriftFieldConfig,
    negative_log_weights: torch.Tensor | None = None,
    generated_negative_count: int | None = None,
) -> torch.Tensor:
    drift_pos, drift_neg = compute_drift_components(
        x=x,
        y_pos=y_pos,
        y_neg=y_neg,
        config=config,
        negative_log_weights=negative_log_weights,
        generated_negative_count=generated_negative_count,
    )
    return drift_pos - drift_neg


def compute_drift_components(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    *,
    config: DriftFieldConfig,
    negative_log_weights: torch.Tensor | None = None,
    generated_negative_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    affinity_pos, affinity_neg = compute_affinity_matrices(
        x=x,
        y_pos=y_pos,
        y_neg=y_neg,
        config=config,
        negative_log_weights=negative_log_weights,
        generated_negative_count=generated_negative_count,
    )
    weight_pos = affinity_pos * affinity_neg.sum(dim=1, keepdim=True)
    weight_neg = affinity_neg * affinity_pos.sum(dim=1, keepdim=True)

    drift_pos = weight_pos @ y_pos
    drift_neg = weight_neg @ y_neg
    return drift_pos, drift_neg


def _validate_inputs(
    *,
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    negative_log_weights: torch.Tensor | None,
    generated_negative_count: int | None,
) -> None:
    for name, value in (("x", x), ("y_pos", y_pos), ("y_neg", y_neg)):
        if value.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape {tuple(value.shape)}")
    if x.shape[1] != y_pos.shape[1] or x.shape[1] != y_neg.shape[1]:
        raise ValueError("x, y_pos, y_neg must share feature dimension")
    if negative_log_weights is not None:
        if negative_log_weights.ndim != 1:
            raise ValueError("negative_log_weights must be 1D")
        if negative_log_weights.shape[0] != y_neg.shape[0]:
            raise ValueError("negative_log_weights size must match y_neg count")
    if generated_negative_count is not None:
        if generated_negative_count < 0:
            raise ValueError("generated_negative_count must be >= 0")
        if generated_negative_count > y_neg.shape[0]:
            raise ValueError("generated_negative_count cannot exceed y_neg count")
