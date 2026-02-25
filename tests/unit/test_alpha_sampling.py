import torch

from drifting_models.utils.alpha import sample_alpha


def test_table8_alpha_distribution_has_half_point_mass() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    groups = 20000
    alpha = sample_alpha(
        groups=groups,
        device=device,
        alpha_fixed=None,
        alpha_min=1.0,
        alpha_max=4.0,
        alpha_dist="table8_l2_latent",
        alpha_power=3.0,
        alpha_point=1.0,
        alpha_point_prob=0.5,
    )
    assert alpha.shape == (groups,)
    assert float(alpha.min().item()) >= 1.0
    assert float(alpha.max().item()) <= 4.0

    point_mass = float((alpha == 1.0).float().mean().item())
    assert 0.45 <= point_mass <= 0.55

    tail = alpha[alpha > 1.0]
    assert tail.numel() > 0

    low = float(((tail >= 1.0) & (tail < 2.0)).float().mean().item())
    high = float(((tail >= 3.0) & (tail <= 4.0)).float().mean().item())
    assert low > high

