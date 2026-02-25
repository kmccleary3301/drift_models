import torch

from drifting_models.features import (
    LatentResNetMAE,
    LatentResNetMAEConfig,
    masked_reconstruction_loss,
    sample_random_mask,
)


def test_latent_mae_forward_shapes() -> None:
    model = LatentResNetMAE(
        LatentResNetMAEConfig(
            in_channels=4,
            base_channels=8,
            stages=3,
            mask_ratio=0.5,
        )
    )
    images = torch.randn(2, 4, 32, 32, requires_grad=True)
    reconstruction, mask, features = model(images)
    assert reconstruction.shape == images.shape
    assert mask.shape == (2, 1, 32, 32)
    assert len(features) == 3
    assert features[0].shape[-2:] == (16, 16)
    assert features[-1].shape[-2:] == (4, 4)


def test_masked_reconstruction_loss_backward() -> None:
    model = LatentResNetMAE(
        LatentResNetMAEConfig(
            in_channels=4,
            base_channels=8,
            stages=3,
            mask_ratio=0.6,
        )
    )
    images = torch.randn(2, 4, 32, 32, requires_grad=True)
    reconstruction, mask, _ = model(images)
    loss, stats = masked_reconstruction_loss(
        reconstruction=reconstruction,
        target=images,
        mask=mask,
    )
    loss.backward()
    assert loss.item() >= 0.0
    assert 0.0 <= stats["mask_ratio_realized"] <= 1.0
    assert images.grad is not None


def test_patchwise_mask_sampling() -> None:
    images = torch.randn(1, 4, 8, 8)
    mask = sample_random_mask(images, mask_ratio=0.5, patch_size=2)
    assert mask.shape == (1, 1, 8, 8)
    for y in range(0, 8, 2):
        for x in range(0, 8, 2):
            patch = mask[0, 0, y : y + 2, x : x + 2]
            assert torch.allclose(patch, torch.full_like(patch, patch[0, 0]))


def test_legacy_mae_forward_shapes() -> None:
    model = LatentResNetMAE(
        LatentResNetMAEConfig(
            in_channels=4,
            base_channels=8,
            stages=3,
            encoder_arch="legacy_conv",
            mask_ratio=0.5,
        )
    )
    images = torch.randn(2, 4, 32, 32)
    reconstruction, mask, features = model(images)
    assert reconstruction.shape == images.shape
    assert mask.shape == (2, 1, 32, 32)
    assert len(features) == 3


def test_resnet_unet_mae_has_more_capacity_than_legacy() -> None:
    legacy = LatentResNetMAE(
        LatentResNetMAEConfig(
            in_channels=4,
            base_channels=8,
            stages=3,
            encoder_arch="legacy_conv",
            mask_ratio=0.5,
        )
    )
    resnet = LatentResNetMAE(
        LatentResNetMAEConfig(
            in_channels=4,
            base_channels=8,
            stages=3,
            encoder_arch="resnet_unet",
            blocks_per_stage=2,
            mask_ratio=0.5,
        )
    )
    legacy_params = sum(parameter.numel() for parameter in legacy.parameters())
    resnet_params = sum(parameter.numel() for parameter in resnet.parameters())
    assert resnet_params > legacy_params
