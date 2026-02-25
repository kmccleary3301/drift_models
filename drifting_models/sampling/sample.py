from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def sample_pixel_generator(
    *,
    generator: nn.Module,
    noise: torch.Tensor,
    class_labels: torch.Tensor,
    alpha: torch.Tensor,
    style_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    generator.eval()
    return generator(noise, class_labels, alpha, style_indices)

