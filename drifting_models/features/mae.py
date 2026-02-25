from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LatentResNetMAEConfig:
    in_channels: int = 4
    base_channels: int = 32
    stages: int = 3
    mask_ratio: float = 0.6
    mask_patch_size: int = 2
    encoder_arch: str = "resnet_unet"  # {"resnet_unet", "legacy_conv", "paper_resnet34_unet"}
    blocks_per_stage: int = 2
    norm_groups: int = 8


def _group_count(num_channels: int, max_groups: int) -> int:
    groups = max(1, min(int(max_groups), int(num_channels)))
    while groups > 1 and (num_channels % groups) != 0:
        groups -= 1
    return groups


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int,
        norm_groups: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(
            num_groups=_group_count(out_channels, norm_groups),
            num_channels=out_channels,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=_group_count(out_channels, norm_groups),
            num_channels=out_channels,
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.GroupNorm(
                    num_groups=_group_count(out_channels, norm_groups),
                    num_channels=out_channels,
                ),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        hidden = self.conv1(x)
        hidden = self.norm1(hidden)
        hidden = self.activation(hidden)
        hidden = self.conv2(hidden)
        hidden = self.norm2(hidden)
        return self.activation(hidden + residual)


class _PaperResidualBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int,
        norm_groups: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(
            num_groups=_group_count(out_channels, norm_groups),
            num_channels=out_channels,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=_group_count(out_channels, norm_groups),
            num_channels=out_channels,
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.GroupNorm(
                    num_groups=_group_count(out_channels, norm_groups),
                    num_channels=out_channels,
                ),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        hidden = self.conv1(x)
        hidden = self.norm1(hidden)
        hidden = self.activation(hidden)
        hidden = self.conv2(hidden)
        hidden = self.norm2(hidden)
        return self.activation(hidden + residual)


class _DecoderStage(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        use_skip: bool,
        skip_channels: int | None,
        norm_groups: int,
        final_stage: bool,
    ) -> None:
        super().__init__()
        self.use_skip = bool(use_skip)
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=final_stage,
        )
        if self.use_skip:
            if skip_channels is None:
                raise ValueError("skip_channels must be set when use_skip=True")
            if skip_channels == out_channels:
                self.skip_projection: nn.Module = nn.Identity()
            else:
                self.skip_projection = nn.Conv2d(
                    skip_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
        else:
            self.skip_projection = nn.Identity()
        if final_stage:
            self.refine: nn.Module = nn.Identity()
        else:
            self.refine = _ResidualBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                norm_groups=norm_groups,
            )

    def forward(self, x: torch.Tensor, *, skip: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.up(x)
        if self.use_skip and skip is not None:
            if skip.shape[-2:] != hidden.shape[-2:]:
                skip = F.interpolate(skip, size=hidden.shape[-2:], mode="nearest")
            hidden = hidden + self.skip_projection(skip)
        return self.refine(hidden)


class _PaperDecoderStage(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm_groups: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(
            num_groups=_group_count(out_channels, norm_groups),
            num_channels=out_channels,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=_group_count(out_channels, norm_groups),
            num_channels=out_channels,
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, *, skip: torch.Tensor) -> torch.Tensor:
        hidden = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if hidden.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=hidden.shape[-2:], mode="bilinear", align_corners=False)
        hidden = torch.cat((hidden, skip), dim=1)
        hidden = self.conv1(hidden)
        hidden = self.norm1(hidden)
        hidden = self.activation(hidden)
        hidden = self.conv2(hidden)
        hidden = self.norm2(hidden)
        hidden = self.activation(hidden)
        return hidden


class LatentResNetMAE(nn.Module):
    def __init__(self, config: LatentResNetMAEConfig) -> None:
        super().__init__()
        if config.stages < 1:
            raise ValueError("stages must be >= 1")
        if config.blocks_per_stage < 1:
            raise ValueError("blocks_per_stage must be >= 1")
        if config.encoder_arch not in {"resnet_unet", "legacy_conv", "paper_resnet34_unet"}:
            raise ValueError("encoder_arch must be one of {'resnet_unet', 'legacy_conv', 'paper_resnet34_unet'}")
        self.config = config
        self._uses_skip_connections = False
        self._uses_paper_arch = False

        if config.encoder_arch == "legacy_conv":
            self._build_legacy_conv()
        elif config.encoder_arch == "paper_resnet34_unet":
            self._build_paper_resnet34_unet()
        else:
            self._build_resnet_unet()

    def _build_legacy_conv(self) -> None:
        encoder_blocks: list[nn.Module] = []
        stage_channels: list[int] = []
        in_channels = self.config.in_channels
        for stage in range(self.config.stages):
            out_channels = self.config.base_channels * (2**stage)
            encoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(
                        num_groups=_group_count(out_channels, self.config.norm_groups),
                        num_channels=out_channels,
                    ),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(
                        num_groups=_group_count(out_channels, self.config.norm_groups),
                        num_channels=out_channels,
                    ),
                    nn.SiLU(),
                )
            )
            stage_channels.append(out_channels)
            in_channels = out_channels
        self.encoder = nn.ModuleList(encoder_blocks)

        decoder_blocks: list[nn.Module] = []
        hidden_channels = stage_channels[-1]
        for stage in reversed(range(self.config.stages)):
            out_channels = self.config.in_channels if stage == 0 else stage_channels[stage - 1]
            block_layers: list[nn.Module] = [
                nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=4, stride=2, padding=1)
            ]
            if stage != 0:
                block_layers.extend(
                    [
                        nn.GroupNorm(
                            num_groups=_group_count(out_channels, self.config.norm_groups),
                            num_channels=out_channels,
                        ),
                        nn.SiLU(),
                    ]
                )
            decoder_blocks.append(nn.Sequential(*block_layers))
            hidden_channels = out_channels
        self.decoder = nn.ModuleList(decoder_blocks)
        self._uses_skip_connections = False

    def _build_resnet_unet(self) -> None:
        encoder_blocks: list[nn.Module] = []
        stage_channels: list[int] = []
        in_channels = self.config.in_channels
        for stage in range(self.config.stages):
            out_channels = self.config.base_channels * (2**stage)
            stage_layers: list[nn.Module] = [
                _ResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2,
                    norm_groups=self.config.norm_groups,
                )
            ]
            for _ in range(self.config.blocks_per_stage - 1):
                stage_layers.append(
                    _ResidualBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        stride=1,
                        norm_groups=self.config.norm_groups,
                    )
                )
            encoder_blocks.append(nn.Sequential(*stage_layers))
            stage_channels.append(out_channels)
            in_channels = out_channels
        self.encoder = nn.ModuleList(encoder_blocks)

        decoder_blocks: list[nn.Module] = []
        hidden_channels = stage_channels[-1]
        for stage in reversed(range(self.config.stages)):
            final_stage = stage == 0
            out_channels = self.config.in_channels if final_stage else stage_channels[stage - 1]
            skip_channels = None if final_stage else stage_channels[stage - 1]
            decoder_blocks.append(
                _DecoderStage(
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    use_skip=not final_stage,
                    skip_channels=skip_channels,
                    norm_groups=self.config.norm_groups,
                    final_stage=final_stage,
                )
            )
            hidden_channels = out_channels
        self.decoder = nn.ModuleList(decoder_blocks)
        self._uses_skip_connections = True

    def _build_paper_resnet34_unet(self) -> None:
        if int(self.config.stages) != 4:
            raise ValueError("paper_resnet34_unet requires stages=4")
        channels = int(self.config.base_channels)
        self.paper_stem = nn.Sequential(
            nn.Conv2d(self.config.in_channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=_group_count(channels, self.config.norm_groups), num_channels=channels),
            nn.ReLU(inplace=True),
        )
        stage_block_counts = (3, 4, 6, 3)
        stage_out_channels = (channels, 2 * channels, 4 * channels, 8 * channels)
        stage_strides = (1, 2, 2, 2)
        paper_stages: list[nn.ModuleList] = []
        in_channels = channels
        for out_channels, blocks_count, first_stride in zip(stage_out_channels, stage_block_counts, stage_strides):
            blocks: list[nn.Module] = [
                _PaperResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=first_stride,
                    norm_groups=self.config.norm_groups,
                )
            ]
            for _ in range(blocks_count - 1):
                blocks.append(
                    _PaperResidualBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        stride=1,
                        norm_groups=self.config.norm_groups,
                    )
                )
            paper_stages.append(nn.ModuleList(blocks))
            in_channels = out_channels
        self.paper_encoder_stages = nn.ModuleList(paper_stages)
        self.paper_decoder_stages = nn.ModuleList(
            [
                _PaperDecoderStage(
                    in_channels=8 * channels,
                    skip_channels=4 * channels,
                    out_channels=4 * channels,
                    norm_groups=self.config.norm_groups,
                ),
                _PaperDecoderStage(
                    in_channels=4 * channels,
                    skip_channels=2 * channels,
                    out_channels=2 * channels,
                    norm_groups=self.config.norm_groups,
                ),
                _PaperDecoderStage(
                    in_channels=2 * channels,
                    skip_channels=channels,
                    out_channels=channels,
                    norm_groups=self.config.norm_groups,
                ),
            ]
        )
        self.paper_decoder_output = nn.Conv2d(channels, self.config.in_channels, kernel_size=3, stride=1, padding=1)
        self._uses_skip_connections = True
        self._uses_paper_arch = True

    def encode(self, images: torch.Tensor) -> list[torch.Tensor]:
        if images.ndim != 4:
            raise ValueError(f"images must be [B, C, H, W], got {tuple(images.shape)}")
        if self._uses_paper_arch:
            hidden = self.paper_stem(images)
            stage_outputs: list[torch.Tensor] = []
            for blocks in self.paper_encoder_stages:
                for block in blocks:
                    hidden = block(hidden)
                stage_outputs.append(hidden)
            return stage_outputs
        features: list[torch.Tensor] = []
        hidden = images
        for block in self.encoder:
            hidden = block(hidden)
            features.append(hidden)
        return features

    def encode_feature_taps(self, images: torch.Tensor) -> list[torch.Tensor]:
        if images.ndim != 4:
            raise ValueError(f"images must be [B, C, H, W], got {tuple(images.shape)}")
        if not self._uses_paper_arch:
            return self.encode(images)
        hidden = self.paper_stem(images)
        taps: list[torch.Tensor] = []
        for blocks in self.paper_encoder_stages:
            block_count = len(blocks)
            for idx, block in enumerate(blocks, start=1):
                hidden = block(hidden)
                if (idx % 2 == 0) or (idx == block_count):
                    taps.append(hidden)
        return taps

    def decode(
        self,
        bottleneck: torch.Tensor,
        *,
        encoder_features: Sequence[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self._uses_paper_arch:
            if encoder_features is None:
                raise ValueError("encoder_features are required for paper_resnet34_unet decode")
            if len(encoder_features) != 4:
                raise ValueError("paper_resnet34_unet decode expects 4 stage encoder features")
            hidden = encoder_features[-1]
            hidden = self.paper_decoder_stages[0](hidden, skip=encoder_features[2])
            hidden = self.paper_decoder_stages[1](hidden, skip=encoder_features[1])
            hidden = self.paper_decoder_stages[2](hidden, skip=encoder_features[0])
            return self.paper_decoder_output(hidden)
        hidden = bottleneck
        for decode_index, block in enumerate(self.decoder):
            if isinstance(block, _DecoderStage):
                skip = None
                if self._uses_skip_connections and encoder_features is not None:
                    stage_index = self.config.stages - decode_index - 1
                    if stage_index > 0:
                        skip = encoder_features[stage_index - 1]
                hidden = block(hidden, skip=skip)
            else:
                hidden = block(hidden)
        return hidden

    def forward(
        self,
        images: torch.Tensor,
        *,
        mask_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        ratio = self.config.mask_ratio if mask_ratio is None else mask_ratio
        mask = sample_random_mask(images, mask_ratio=ratio, patch_size=self.config.mask_patch_size)
        masked_inputs = images * (1.0 - mask)
        features = self.encode(masked_inputs)
        reconstruction = self.decode(features[-1], encoder_features=features)
        return reconstruction, mask, features


def sample_random_mask(images: torch.Tensor, *, mask_ratio: float, patch_size: int = 1) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"images must be [B, C, H, W], got {tuple(images.shape)}")
    if not (0.0 <= mask_ratio <= 1.0):
        raise ValueError("mask_ratio must be in [0, 1]")
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if mask_ratio == 0.0:
        return torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3], device=images.device, dtype=images.dtype)
    if mask_ratio == 1.0:
        return torch.ones(images.shape[0], 1, images.shape[2], images.shape[3], device=images.device, dtype=images.dtype)
    if patch_size == 1:
        mask = (torch.rand(images.shape[0], 1, images.shape[2], images.shape[3], device=images.device) < mask_ratio).to(images.dtype)
        return mask
    patch_h = max(1, (images.shape[2] + patch_size - 1) // patch_size)
    patch_w = max(1, (images.shape[3] + patch_size - 1) // patch_size)
    patch_mask = (torch.rand(images.shape[0], 1, patch_h, patch_w, device=images.device) < mask_ratio).to(images.dtype)
    mask = patch_mask.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    mask = mask[:, :, : images.shape[2], : images.shape[3]]
    return mask


def masked_reconstruction_loss(
    *,
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    if reconstruction.shape != target.shape:
        raise ValueError("reconstruction and target must match shape")
    if mask.ndim != 4 or mask.shape[0] != target.shape[0] or mask.shape[2:] != target.shape[2:]:
        raise ValueError("mask must be [B, 1, H, W] with matching spatial size")
    squared_error = (reconstruction - target).pow(2)
    masked_error = squared_error * mask
    mask_denom = torch.clamp(mask.sum() * target.shape[1], min=eps)
    loss = masked_error.sum() / mask_denom

    unmasked = 1.0 - mask
    unmasked_denom = torch.clamp(unmasked.sum() * target.shape[1], min=eps)
    unmasked_mse = (squared_error * unmasked).sum() / unmasked_denom
    stats = {
        "loss": float(loss.item()),
        "mask_ratio_realized": float(mask.mean().item()),
        "masked_mse": float(loss.item()),
        "unmasked_mse": float(unmasked_mse.item()),
    }
    return loss, stats


def mae_feature_maps(
    encoder: LatentResNetMAE,
    images: torch.Tensor,
) -> list[torch.Tensor]:
    return encoder.encode_feature_taps(images)
