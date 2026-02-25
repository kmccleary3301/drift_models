from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SDVAEConfig:
    # Standard "SD-VAE tokenizer" latent scaling used in Stable Diffusion.
    # The paper references SD-VAE; this is the common scaling constant.
    scaling_factor: float = 0.18215

    # VAE weights source. For the common VAE-only repo, `subfolder` should be None.
    model_id: str = "stabilityai/sd-vae-ft-mse"
    subfolder: str | None = None
    revision: str | None = None

    # Desired parameter dtype once moved to the target device.
    # Keep CPU loads in fp32 for compatibility; cast on device.
    dtype: str = "fp16"  # {"fp16","bf16","fp32"}


def resolve_dtype(dtype: str) -> torch.dtype:
    lowered = dtype.strip().lower()
    if lowered in {"fp16", "float16"}:
        return torch.float16
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported SD-VAE dtype: {dtype}")


def load_sd_vae(*, config: SDVAEConfig) -> torch.nn.Module:
    try:
        from diffusers import AutoencoderKL
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "SD-VAE support requires optional dependencies. "
            "Install with: `uv sync --extra sdvae`"
        ) from error

    kwargs: dict[str, object] = {
        "revision": config.revision,
    }
    if config.subfolder is not None:
        kwargs["subfolder"] = config.subfolder
    # Always load on CPU in fp32 for portability; cast when placed on the target device.
    vae = AutoencoderKL.from_pretrained(config.model_id, torch_dtype=torch.float32, **kwargs)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad = False
    return vae


@torch.no_grad()
def encode_images_to_latents(
    *,
    vae: torch.nn.Module,
    images_0_1: torch.Tensor,
    scaling_factor: float,
    sample: bool,
) -> torch.Tensor:
    if images_0_1.ndim != 4 or images_0_1.shape[1] != 3:
        raise ValueError("images_0_1 must be [N, 3, H, W]")
    images = images_0_1 * 2.0 - 1.0
    vae_param = next(vae.parameters(), None)
    if vae_param is not None:
        images = images.to(dtype=vae_param.dtype)
    encoded = vae.encode(images)  # type: ignore[attr-defined]
    latent_dist = encoded.latent_dist
    latents = latent_dist.sample() if sample else latent_dist.mean
    return latents * float(scaling_factor)


@torch.no_grad()
def decode_latents_to_images(
    *,
    vae: torch.nn.Module,
    latents: torch.Tensor,
    scaling_factor: float,
) -> torch.Tensor:
    if latents.ndim != 4:
        raise ValueError("latents must be [N, C, H, W]")
    vae_param = next(vae.parameters(), None)
    if vae_param is not None:
        latents = latents.to(dtype=vae_param.dtype)
    decoded = vae.decode(latents / float(scaling_factor)).sample  # type: ignore[attr-defined]
    images_0_1 = (decoded + 1.0) * 0.5
    return images_0_1.clamp(0.0, 1.0)
