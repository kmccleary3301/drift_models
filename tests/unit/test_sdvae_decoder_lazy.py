from __future__ import annotations

import importlib.util

import pytest
import torch

from drifting_models.features.vae import LatentDecoderConfig, build_latent_decoder


def test_sdvae_decoder_is_lazy_and_errors_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = LatentDecoderConfig(
        mode="sd_vae",
        latent_channels=4,
        out_channels=3,
        image_size=256,
        sd_vae_model_id="stabilityai/sd-vae-ft-mse",
        sd_vae_dtype="fp16",
    )

    if importlib.util.find_spec("diffusers") is None:
        decoder = build_latent_decoder(cfg)
        latents = torch.randn(1, 4, 32, 32)
        with pytest.raises(RuntimeError, match=r"uv sync --extra sdvae"):
            _ = decoder(latents)
        return

    import diffusers  # type: ignore[import-not-found]

    called = {"count": 0}

    def _fake_from_pretrained(*args, **kwargs):
        called["count"] += 1
        raise RuntimeError("sentinel")

    # Decoder construction should not attempt to download weights.
    monkeypatch.setattr(diffusers.AutoencoderKL, "from_pretrained", staticmethod(_fake_from_pretrained))
    decoder = build_latent_decoder(cfg)
    assert getattr(decoder, "vae") is None
    assert called["count"] == 0
