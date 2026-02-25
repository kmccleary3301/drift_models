from pathlib import Path

import torch

from drifting_models.models import DiTLikeConfig, DiTLikeGenerator
from drifting_models.utils import load_pixel_generator_from_checkpoint


def test_load_pixel_generator_from_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = DiTLikeConfig(image_size=16, in_channels=3, out_channels=3, patch_size=4, hidden_dim=32, depth=1, num_heads=4)
    model = DiTLikeGenerator(config)
    checkpoint_path = tmp_path / "ckpt.pt"
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "step": 0,
        "extra": {"model_config": config.__dict__},
    }
    torch.save(payload, checkpoint_path)

    loaded = load_pixel_generator_from_checkpoint(checkpoint_path=checkpoint_path, device=torch.device("cpu"))
    assert isinstance(loaded, DiTLikeGenerator)
    assert loaded.config.image_size == 16
