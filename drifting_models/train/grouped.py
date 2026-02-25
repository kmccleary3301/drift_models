from __future__ import annotations

from dataclasses import dataclass

import torch

from drifting_models.drift_field import DriftFieldConfig, compute_v


@dataclass(frozen=True)
class GroupedBatchShapes:
    groups: int
    negatives_per_group: int
    positives_per_group: int
    feature_dim: int


def infer_grouped_shapes(
    x_grouped: torch.Tensor,
    y_pos_grouped: torch.Tensor,
    y_neg_grouped: torch.Tensor,
) -> GroupedBatchShapes:
    _validate_grouped_inputs(x_grouped=x_grouped, y_pos_grouped=y_pos_grouped, y_neg_grouped=y_neg_grouped)
    return GroupedBatchShapes(
        groups=x_grouped.shape[0],
        negatives_per_group=x_grouped.shape[1],
        positives_per_group=y_pos_grouped.shape[1],
        feature_dim=x_grouped.shape[2],
    )


def compute_grouped_v(
    x_grouped: torch.Tensor,
    y_pos_grouped: torch.Tensor,
    y_neg_grouped: torch.Tensor,
    *,
    config: DriftFieldConfig,
    negative_log_weights_grouped: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_grouped_inputs(x_grouped=x_grouped, y_pos_grouped=y_pos_grouped, y_neg_grouped=y_neg_grouped)
    if negative_log_weights_grouped is not None:
        if negative_log_weights_grouped.ndim != 2:
            raise ValueError("negative_log_weights_grouped must be [G, N_neg]")
        if negative_log_weights_grouped.shape[:2] != y_neg_grouped.shape[:2]:
            raise ValueError("negative_log_weights_grouped shape mismatch with y_neg_grouped")

    grouped_outputs = []
    group_count = x_grouped.shape[0]
    for index in range(group_count):
        negative_log_weights = None
        if negative_log_weights_grouped is not None:
            negative_log_weights = negative_log_weights_grouped[index]
        grouped_outputs.append(
            compute_v(
                x=x_grouped[index],
                y_pos=y_pos_grouped[index],
                y_neg=y_neg_grouped[index],
                config=config,
                negative_log_weights=negative_log_weights,
            )
        )
    return torch.stack(grouped_outputs, dim=0)


def flatten_grouped(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D grouped tensor, got shape {tuple(tensor.shape)}")
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2])


def _validate_grouped_inputs(
    *,
    x_grouped: torch.Tensor,
    y_pos_grouped: torch.Tensor,
    y_neg_grouped: torch.Tensor,
) -> None:
    for name, value in (
        ("x_grouped", x_grouped),
        ("y_pos_grouped", y_pos_grouped),
        ("y_neg_grouped", y_neg_grouped),
    ):
        if value.ndim != 3:
            raise ValueError(f"{name} must be 3D [G, N, D], got shape {tuple(value.shape)}")
    if x_grouped.shape[0] != y_pos_grouped.shape[0] or x_grouped.shape[0] != y_neg_grouped.shape[0]:
        raise ValueError("Grouped tensors must share group dimension")
    if x_grouped.shape[2] != y_pos_grouped.shape[2] or x_grouped.shape[2] != y_neg_grouped.shape[2]:
        raise ValueError("Grouped tensors must share feature dimension")
