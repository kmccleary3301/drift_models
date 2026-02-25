from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FeatureVectorizationConfig:
    include_per_location: bool = True
    include_global_stats: bool = True
    include_patch2_stats: bool = True
    include_patch4_stats: bool = True
    include_input_x2_mean: bool = False
    selected_stages: tuple[int, ...] | None = None


def extract_feature_maps(
    *,
    encoder: nn.Module,
    images: torch.Tensor,
) -> list[torch.Tensor]:
    features = encoder(images)
    if isinstance(features, torch.Tensor):
        return [features]
    if isinstance(features, (list, tuple)):
        outputs = list(features)
        for feature in outputs:
            if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
                raise ValueError("encoder outputs must be tensor feature maps [B, C, H, W]")
        return outputs
    raise ValueError("encoder must return tensor or list/tuple of tensors")


def vectorize_feature_maps(
    feature_maps: list[torch.Tensor],
    *,
    config: FeatureVectorizationConfig,
    input_images: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    vectors: dict[str, torch.Tensor] = {}
    for stage_index, fmap in enumerate(feature_maps):
        if config.selected_stages is not None and stage_index not in config.selected_stages:
            continue
        if fmap.ndim != 4:
            raise ValueError(f"feature map must be [B, C, H, W], got {tuple(fmap.shape)}")
        stage_prefix = f"stage{stage_index}"
        vectors.update(_vectorize_single_map(fmap, stage_prefix, config))
    if config.include_input_x2_mean:
        if input_images is None:
            raise ValueError("input_images must be provided when include_input_x2_mean is enabled")
        if input_images.ndim != 4:
            raise ValueError("input_images must be [B, C, H, W]")
        input_stat = input_images.pow(2).mean(dim=(2, 3))
        vectors["input.x2mean"] = input_stat.unsqueeze(1)
    return vectors


def _vectorize_single_map(
    fmap: torch.Tensor,
    prefix: str,
    config: FeatureVectorizationConfig,
) -> dict[str, torch.Tensor]:
    batch, channels, height, width = fmap.shape
    outputs: dict[str, torch.Tensor] = {}

    if config.include_per_location:
        per_location = fmap.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        outputs[f"{prefix}.loc"] = per_location

    if config.include_global_stats:
        global_mean = fmap.mean(dim=(2, 3))
        global_std = fmap.std(dim=(2, 3), unbiased=False)
        outputs[f"{prefix}.global"] = torch.stack([global_mean, global_std], dim=1)

    if config.include_patch2_stats and height >= 2 and width >= 2:
        patch2_mean, patch2_std = _patch_stats(fmap, patch_size=2)
        outputs[f"{prefix}.patch2"] = torch.cat([patch2_mean, patch2_std], dim=1)

    if config.include_patch4_stats and height >= 4 and width >= 4:
        patch4_mean, patch4_std = _patch_stats(fmap, patch_size=4)
        outputs[f"{prefix}.patch4"] = torch.cat([patch4_mean, patch4_std], dim=1)

    return outputs


def _patch_stats(fmap: torch.Tensor, *, patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch, channels, height, width = fmap.shape
    patch_h = height // patch_size
    patch_w = width // patch_size
    if patch_h == 0 or patch_w == 0:
        raise ValueError("feature map too small for patch stats")
    cropped_h = patch_h * patch_size
    cropped_w = patch_w * patch_size
    cropped = fmap[:, :, :cropped_h, :cropped_w]
    patches = cropped.reshape(
        batch,
        channels,
        patch_h,
        patch_size,
        patch_w,
        patch_size,
    ).permute(0, 2, 4, 1, 3, 5)
    patches = patches.reshape(batch, patch_h * patch_w, channels, patch_size * patch_size)
    patch_mean = patches.mean(dim=-1)
    patch_std = patches.std(dim=-1, unbiased=False)
    return patch_mean, patch_std
