import torch

from drifting_models.models import DiTLikeConfig, DiTLikeGenerator


def test_dit_like_cpu_bfloat16_autocast_smoke() -> None:
    model = DiTLikeGenerator(
        DiTLikeConfig(
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
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    noise = torch.randn(2, 4, 16, 16)
    class_labels = torch.randint(0, 1000, (2,))
    alpha = torch.tensor([1.0, 1.2], dtype=torch.float32)
    style = torch.randint(0, 64, (2, 32))

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(noise, class_labels, alpha, style)
        loss = output.float().pow(2).mean()
    loss.backward()
    optimizer.step()
    assert loss.item() >= 0.0
