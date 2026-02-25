import torch

from drifting_models.models import DiTLikeConfig, DiTLikeGenerator


def test_dit_like_forward_checksum_fixed_seed() -> None:
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
        use_qk_norm=True,
        use_rope=True,
    )
    torch.manual_seed(123)
    model_a = DiTLikeGenerator(config)
    torch.manual_seed(123)
    model_b = DiTLikeGenerator(config)

    torch.manual_seed(999)
    noise = torch.randn(2, 4, 16, 16)
    class_labels = torch.tensor([1, 5], dtype=torch.long)
    alpha = torch.tensor([1.0, 2.0], dtype=torch.float32)
    style_indices = torch.randint(0, config.style_vocab_size, (2, config.style_token_count))

    out_a = model_a(noise, class_labels, alpha, style_indices)
    out_b = model_b(noise, class_labels, alpha, style_indices)
    assert torch.allclose(out_a, out_b, atol=0.0, rtol=0.0)
