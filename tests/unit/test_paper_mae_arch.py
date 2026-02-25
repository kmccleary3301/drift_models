from __future__ import annotations

import torch

from drifting_models.features import LatentResNetMAE, LatentResNetMAEConfig, mae_feature_maps


def test_paper_mae_encoder_stage_shapes() -> None:
    model = LatentResNetMAE(
        LatentResNetMAEConfig(
            in_channels=4,
            base_channels=8,
            stages=4,
            encoder_arch="paper_resnet34_unet",
            norm_groups=4,
            mask_ratio=0.0,
        )
    )
    images = torch.randn(2, 4, 16, 16)
    features = model.encode(images)
    assert len(features) == 4
    assert features[0].shape == (2, 8, 16, 16)
    assert features[1].shape == (2, 16, 8, 8)
    assert features[2].shape == (2, 32, 4, 4)
    assert features[3].shape == (2, 64, 2, 2)


def test_paper_mae_feature_tap_count_and_shapes() -> None:
    model = LatentResNetMAE(
        LatentResNetMAEConfig(
            in_channels=4,
            base_channels=8,
            stages=4,
            encoder_arch="paper_resnet34_unet",
            norm_groups=4,
            mask_ratio=0.0,
        )
    )
    images = torch.randn(2, 4, 16, 16)
    taps = model.encode_feature_taps(images)
    assert len(taps) == 9
    assert taps[0].shape == (2, 8, 16, 16)
    assert taps[1].shape == (2, 8, 16, 16)
    assert taps[2].shape == (2, 16, 8, 8)
    assert taps[3].shape == (2, 16, 8, 8)
    assert taps[4].shape == (2, 32, 4, 4)
    assert taps[5].shape == (2, 32, 4, 4)
    assert taps[6].shape == (2, 32, 4, 4)
    assert taps[7].shape == (2, 64, 2, 2)
    assert taps[8].shape == (2, 64, 2, 2)


def test_paper_mae_forward_shapes() -> None:
    model = LatentResNetMAE(
        LatentResNetMAEConfig(
            in_channels=4,
            base_channels=8,
            stages=4,
            encoder_arch="paper_resnet34_unet",
            norm_groups=4,
            mask_ratio=0.5,
            mask_patch_size=2,
        )
    )
    images = torch.randn(2, 4, 16, 16)
    reconstruction, mask, features = model(images)
    assert reconstruction.shape == images.shape
    assert mask.shape == (2, 1, 16, 16)
    assert len(features) == 4


def test_mae_feature_maps_uses_paper_taps() -> None:
    model = LatentResNetMAE(
        LatentResNetMAEConfig(
            in_channels=4,
            base_channels=8,
            stages=4,
            encoder_arch="paper_resnet34_unet",
            norm_groups=4,
            mask_ratio=0.0,
        )
    )
    images = torch.randn(1, 4, 16, 16)
    maps = mae_feature_maps(model, images)
    assert len(maps) == 9

