from __future__ import annotations

import torch


def postprocess_images(images: torch.Tensor, *, mode: str) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"images must be [N, C, H, W], got {tuple(images.shape)}")
    values = images.float()
    if mode == "clamp_0_1":
        return values.clamp(0.0, 1.0)
    if mode == "tanh_to_0_1":
        return ((values.tanh() + 1.0) / 2.0).clamp(0.0, 1.0)
    if mode == "sigmoid":
        return values.sigmoid().clamp(0.0, 1.0)
    if mode == "identity":
        return values
    raise ValueError(f"Unknown postprocess mode: {mode}")

