from __future__ import annotations

import math

import torch


def sample_alpha(
    *,
    groups: int,
    device: torch.device,
    alpha_fixed: float | None,
    alpha_min: float,
    alpha_max: float,
    alpha_dist: str = "uniform",
    alpha_power: float = 3.0,
    alpha_point: float = 1.0,
    alpha_point_prob: float = 0.5,
) -> torch.Tensor:
    if groups <= 0:
        raise ValueError("groups must be > 0")
    if alpha_fixed is not None:
        alpha = float(alpha_fixed)
        if alpha < 1.0:
            raise ValueError("alpha_fixed must be >= 1.0")
        return torch.full((groups,), alpha, device=device, dtype=torch.float32)

    alpha_min = float(alpha_min)
    alpha_max = float(alpha_max)
    if alpha_min < 1.0:
        raise ValueError("alpha_min must be >= 1.0")
    if alpha_max < alpha_min:
        raise ValueError("alpha_max must be >= alpha_min")
    if alpha_max == alpha_min:
        return torch.full((groups,), alpha_min, device=device, dtype=torch.float32)

    dist = str(alpha_dist).strip().lower()
    if dist == "uniform":
        return torch.rand(groups, device=device, dtype=torch.float32) * (alpha_max - alpha_min) + alpha_min
    if dist == "powerlaw":
        return _sample_powerlaw(
            groups=groups,
            device=device,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            power=float(alpha_power),
        )
    if dist in ("mixture_point_powerlaw", "table8_l2_latent"):
        # Table 8 (DiT-L/2 latent): 50% alpha=1, 50% p(alpha) ∝ alpha^-3 on [1, 4].
        point = float(alpha_point)
        point_prob = float(alpha_point_prob)
        if not (0.0 <= point_prob <= 1.0):
            raise ValueError("alpha_point_prob must be in [0, 1]")
        if not (alpha_min <= point <= alpha_max):
            raise ValueError("alpha_point must be within [alpha_min, alpha_max]")
        choose_point = torch.rand(groups, device=device, dtype=torch.float32) < point_prob
        sample = _sample_powerlaw(
            groups=groups,
            device=device,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            power=float(alpha_power),
        )
        return torch.where(
            choose_point,
            torch.full((groups,), point, device=device, dtype=torch.float32),
            sample,
        )
    raise ValueError(f"Unsupported alpha_dist: {alpha_dist!r}")


def _sample_powerlaw(
    *,
    groups: int,
    device: torch.device,
    alpha_min: float,
    alpha_max: float,
    power: float,
) -> torch.Tensor:
    # p(alpha) ∝ alpha^-power on [alpha_min, alpha_max]
    if not math.isfinite(power) or power <= 0.0:
        raise ValueError("power must be finite and > 0")

    u = torch.rand(groups, device=device, dtype=torch.float32)
    if abs(power - 1.0) < 1e-8:
        # p(a) ∝ 1/a
        ratio = alpha_max / alpha_min
        return alpha_min * torch.pow(torch.tensor(ratio, device=device, dtype=torch.float32), u)

    one_minus = 1.0 - power
    a0 = alpha_min**one_minus
    a1 = alpha_max**one_minus
    base = u * (a1 - a0) + a0
    return torch.pow(base, torch.tensor(1.0 / one_minus, device=device, dtype=torch.float32))

