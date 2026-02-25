import torch

from drifting_models.models import DiTLikeConfig, DiTLikeGenerator


def test_dit_like_forward_shape() -> None:
    config = DiTLikeConfig(
        image_size=32,
        in_channels=4,
        out_channels=4,
        patch_size=2,
        hidden_dim=128,
        depth=2,
        num_heads=8,
        register_tokens=8,
    )
    model = DiTLikeGenerator(config)
    noise = torch.randn(3, 4, 32, 32)
    class_labels = torch.randint(0, config.num_classes, (3,))
    alpha = torch.ones(3)
    style_indices = torch.randint(0, config.style_vocab_size, (3, config.style_token_count))

    output = model(noise, class_labels, alpha, style_indices)
    assert output.shape == noise.shape


def test_dit_like_backward_and_no_register_tokens() -> None:
    config = DiTLikeConfig(
        image_size=16,
        in_channels=4,
        out_channels=4,
        patch_size=4,
        hidden_dim=64,
        depth=2,
        num_heads=4,
        register_tokens=0,
    )
    model = DiTLikeGenerator(config)
    noise = torch.randn(2, 4, 16, 16)
    class_labels = torch.randint(0, config.num_classes, (2,))
    alpha = torch.tensor([1.0, 2.0])

    loss = model(noise, class_labels, alpha).pow(2).mean()
    loss.backward()
    assert model.patch_embed.weight.grad is not None


def test_dit_like_rmsnorm_option_backward() -> None:
    config = DiTLikeConfig(
        image_size=16,
        in_channels=4,
        out_channels=4,
        patch_size=4,
        hidden_dim=64,
        depth=2,
        num_heads=4,
        register_tokens=2,
        norm_type="rmsnorm",
    )
    model = DiTLikeGenerator(config)
    noise = torch.randn(2, 4, 16, 16)
    class_labels = torch.randint(0, config.num_classes, (2,))
    alpha = torch.tensor([1.5, 2.0])
    loss = model(noise, class_labels, alpha).abs().mean()
    loss.backward()
    assert model.patch_embed.weight.grad is not None


def test_dit_like_qknorm_option_backward() -> None:
    config = DiTLikeConfig(
        image_size=16,
        in_channels=4,
        out_channels=4,
        patch_size=4,
        hidden_dim=64,
        depth=2,
        num_heads=4,
        register_tokens=2,
        use_qk_norm=True,
    )
    model = DiTLikeGenerator(config)
    noise = torch.randn(2, 4, 16, 16)
    class_labels = torch.randint(0, config.num_classes, (2,))
    alpha = torch.tensor([1.0, 1.2])
    loss = model(noise, class_labels, alpha).pow(2).mean()
    loss.backward()
    assert model.patch_embed.weight.grad is not None


def test_dit_like_rope_option_backward() -> None:
    config = DiTLikeConfig(
        image_size=16,
        in_channels=4,
        out_channels=4,
        patch_size=4,
        hidden_dim=64,
        depth=2,
        num_heads=4,
        register_tokens=2,
        use_rope=True,
    )
    model = DiTLikeGenerator(config)
    noise = torch.randn(2, 4, 16, 16)
    class_labels = torch.randint(0, config.num_classes, (2,))
    alpha = torch.tensor([1.0, 2.0])
    loss = model(noise, class_labels, alpha).abs().mean()
    loss.backward()
    assert model.patch_embed.weight.grad is not None
