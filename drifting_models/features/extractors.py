from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TinyFeatureEncoderConfig:
    in_channels: int = 4
    base_channels: int = 32
    stages: int = 3


class TinyFeatureEncoder(nn.Module):
    def __init__(self, config: TinyFeatureEncoderConfig) -> None:
        super().__init__()
        self.config = config
        blocks: list[nn.Module] = []
        in_channels = config.in_channels
        for stage in range(config.stages):
            out_channels = config.base_channels * (2**stage)
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=max(1, out_channels // 8), num_channels=out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=max(1, out_channels // 8), num_channels=out_channels),
                    nn.SiLU(),
                )
            )
            in_channels = out_channels
        self.stages = nn.ModuleList(blocks)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        if images.ndim != 4:
            raise ValueError(f"images must be [B, C, H, W], got {tuple(images.shape)}")
        features: list[torch.Tensor] = []
        hidden = images
        for block in self.stages:
            hidden = block(hidden)
            features.append(hidden)
        return features


def freeze_module_parameters(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze_module_parameters(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = True
