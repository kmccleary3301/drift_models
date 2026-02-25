from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LatentDecoderConfig:
    mode: str = "identity"
    latent_channels: int = 4
    out_channels: int = 3
    image_size: int = 32
    hidden_channels: int = 64
    sd_vae_model_id: str | None = None
    sd_vae_subfolder: str | None = None
    sd_vae_revision: str | None = None
    sd_vae_scaling_factor: float = 0.18215
    sd_vae_dtype: str = "fp16"


class IdentityLatentDecoder(nn.Module):
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return latents


class ConvLatentDecoder(nn.Module):
    def __init__(self, config: LatentDecoderConfig) -> None:
        super().__init__()
        if config.mode != "conv":
            raise ValueError("ConvLatentDecoder requires mode='conv'")
        self.config = config
        self.net = nn.Sequential(
            nn.Conv2d(config.latent_channels, config.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(config.hidden_channels, config.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(config.hidden_channels, config.out_channels, kernel_size=1),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        decoded = self.net(latents)
        if decoded.shape[-1] != self.config.image_size or decoded.shape[-2] != self.config.image_size:
            decoded = torch.nn.functional.interpolate(
                decoded,
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return decoded


class SDVAELatentDecoder(nn.Module):
    def __init__(self, config: LatentDecoderConfig) -> None:
        super().__init__()
        if config.mode != "sd_vae":
            raise ValueError("SDVAELatentDecoder requires mode='sd_vae'")
        if config.sd_vae_model_id is None or not config.sd_vae_model_id.strip():
            raise ValueError("sd_vae_model_id must be provided when mode='sd_vae'")
        if config.out_channels != 3:
            raise ValueError("SD-VAE decode only supports out_channels=3")
        self.config = config
        from drifting_models.features.sd_vae import decode_latents_to_images, resolve_dtype

        self._decode_latents_to_images = decode_latents_to_images
        self._target_dtype = resolve_dtype(config.sd_vae_dtype)
        self.vae: torch.nn.Module | None = None

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            from drifting_models.features.sd_vae import SDVAEConfig, load_sd_vae

            self.vae = load_sd_vae(
                config=SDVAEConfig(
                    scaling_factor=self.config.sd_vae_scaling_factor,
                    model_id=str(self.config.sd_vae_model_id),
                    subfolder=self.config.sd_vae_subfolder,
                    revision=self.config.sd_vae_revision,
                    dtype=self.config.sd_vae_dtype,
                )
            )
            self.vae.to(device=latents.device, dtype=self._target_dtype)
        decoded = self._decode_latents_to_images(
            vae=self.vae,
            latents=latents,
            scaling_factor=float(self.config.sd_vae_scaling_factor),
        )
        if decoded.shape[-1] != self.config.image_size or decoded.shape[-2] != self.config.image_size:
            decoded = torch.nn.functional.interpolate(
                decoded,
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return decoded


class DecoderWrappedFeatureExtractor(nn.Module):
    def __init__(self, *, decoder: nn.Module, feature_extractor: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder
        self.feature_extractor = feature_extractor

    def forward(self, latents: torch.Tensor) -> list[torch.Tensor]:
        decoded = self.decoder(latents)
        return self.feature_extractor(decoded)


def build_latent_decoder(config: LatentDecoderConfig) -> nn.Module:
    if config.mode == "identity":
        return IdentityLatentDecoder()
    if config.mode == "conv":
        return ConvLatentDecoder(config)
    if config.mode == "sd_vae":
        return SDVAELatentDecoder(config)
    raise ValueError(f"Unsupported decoder mode: {config.mode}")
