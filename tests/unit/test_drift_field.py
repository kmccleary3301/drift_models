import torch

from drifting_models import (
    DriftFieldConfig,
    build_negative_log_weights,
    cfg_alpha_to_unconditional_weight,
    compute_affinity_matrices,
    compute_v,
)


def test_compute_v_shape() -> None:
    config = DriftFieldConfig(temperature=0.05, normalize_over_x=True, mask_self_negatives=False)
    x = torch.randn(8, 3)
    y_pos = torch.randn(10, 3)
    y_neg = torch.randn(12, 3)

    v = compute_v(x, y_pos, y_neg, config=config)
    assert v.shape == x.shape


def test_equilibrium_zero_for_identical_sets_without_self_mask() -> None:
    config = DriftFieldConfig(temperature=0.1, normalize_over_x=True, mask_self_negatives=False)
    x = torch.randn(16, 4)

    v = compute_v(x, x, x, config=config)
    assert torch.allclose(v, torch.zeros_like(v), atol=1e-6, rtol=1e-6)


def test_equilibrium_zero_for_identical_sets_without_x_normalization() -> None:
    config = DriftFieldConfig(temperature=0.1, normalize_over_x=False, mask_self_negatives=False)
    x = torch.randn(16, 4)
    v = compute_v(x, x, x, config=config)
    assert torch.allclose(v, torch.zeros_like(v), atol=1e-6, rtol=1e-6)


def test_compute_v_is_antisymmetric_in_pos_neg_sets() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 4)
    y_a = torch.randn(12, 4)
    y_b = torch.randn(12, 4)

    for normalize_over_x in (False, True):
        config = DriftFieldConfig(
            temperature=0.05,
            normalize_over_x=normalize_over_x,
            mask_self_negatives=False,
        )
        v_ab = compute_v(x, y_a, y_b, config=config)
        v_ba = compute_v(x, y_b, y_a, config=config)
        assert torch.allclose(v_ab, -v_ba, atol=1e-7, rtol=0.0)


def test_self_mask_suppresses_diagonal_negative_affinity() -> None:
    x = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    )
    y_pos = x.clone()
    y_neg = x.clone()

    no_mask_cfg = DriftFieldConfig(temperature=0.05, normalize_over_x=False, mask_self_negatives=False)
    mask_cfg = DriftFieldConfig(temperature=0.05, normalize_over_x=False, mask_self_negatives=True)

    _, neg_no_mask = compute_affinity_matrices(x, y_pos, y_neg, config=no_mask_cfg)
    _, neg_mask = compute_affinity_matrices(x, y_pos, y_neg, config=mask_cfg)

    diag = torch.arange(x.shape[0])
    assert torch.all(neg_mask[diag, diag] < neg_no_mask[diag, diag])


def test_cfg_alpha_to_weight_matches_formula() -> None:
    n_neg = 64
    n_unc = 16
    for alpha in (1.0, 1.25, 1.5, 2.0, 3.0):
        expected = ((alpha - 1.0) * (n_neg - 1)) / n_unc
        actual = cfg_alpha_to_unconditional_weight(alpha, n_neg, n_unc)
        assert actual == expected


def test_unconditional_negative_log_weights_disable_unconditional_at_alpha_one() -> None:
    x = torch.randn(6, 2)
    y_pos = torch.randn(6, 2)
    y_neg_generated = torch.randn(6, 2)
    y_neg_uncond = torch.randn(3, 2)
    y_neg = torch.cat([y_neg_generated, y_neg_uncond], dim=0)

    config = DriftFieldConfig(temperature=0.1, normalize_over_x=False, mask_self_negatives=False)

    log_weights = build_negative_log_weights(
        n_generated_negatives=y_neg_generated.shape[0],
        n_unconditional_negatives=y_neg_uncond.shape[0],
        unconditional_weight=0.0,
        device=x.device,
        dtype=x.dtype,
    )
    _, neg_affinity = compute_affinity_matrices(
        x=x,
        y_pos=y_pos,
        y_neg=y_neg,
        config=config,
        negative_log_weights=log_weights,
    )
    uncond_mass = neg_affinity[:, y_neg_generated.shape[0] :].sum()
    assert torch.isclose(uncond_mass, torch.tensor(0.0), atol=1e-7, rtol=0.0)


def test_unconditional_weight_zero_is_finite_with_x_normalization() -> None:
    x = torch.randn(6, 2)
    y_pos = torch.randn(6, 2)
    y_neg_generated = torch.randn(6, 2)
    y_neg_uncond = torch.randn(3, 2)
    y_neg = torch.cat([y_neg_generated, y_neg_uncond], dim=0)

    config = DriftFieldConfig(temperature=0.1, normalize_over_x=True, mask_self_negatives=False)

    log_weights = build_negative_log_weights(
        n_generated_negatives=y_neg_generated.shape[0],
        n_unconditional_negatives=y_neg_uncond.shape[0],
        unconditional_weight=0.0,
        device=x.device,
        dtype=x.dtype,
    )
    pos_affinity, neg_affinity = compute_affinity_matrices(
        x=x,
        y_pos=y_pos,
        y_neg=y_neg,
        config=config,
        negative_log_weights=log_weights,
    )
    assert torch.isfinite(pos_affinity).all()
    assert torch.isfinite(neg_affinity).all()
    uncond_mass = neg_affinity[:, y_neg_generated.shape[0] :].sum()
    assert float(uncond_mass.item()) <= 1e-4


def test_partial_self_mask_when_unconditional_negatives_present() -> None:
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    y_pos = x.clone()
    y_neg_generated = x.clone()
    y_neg_unconditional = torch.tensor([[100.0, 100.0], [-100.0, -100.0]])
    y_neg = torch.cat([y_neg_generated, y_neg_unconditional], dim=0)

    config = DriftFieldConfig(temperature=0.05, normalize_over_x=False, mask_self_negatives=True)
    _, neg_affinity = compute_affinity_matrices(
        x=x,
        y_pos=y_pos,
        y_neg=y_neg,
        config=config,
        generated_negative_count=y_neg_generated.shape[0],
    )
    diagonal = torch.arange(x.shape[0])
    assert torch.all(neg_affinity[diagonal, diagonal] < 1e-6)


def _assert_compute_v_is_finite(*, device: torch.device, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    config = DriftFieldConfig(temperature=0.05, normalize_over_x=True, mask_self_negatives=True)
    x = torch.randn(32, 8, device=device, dtype=torch.float32).to(dtype)
    y_pos = torch.randn(48, 8, device=device, dtype=torch.float32).to(dtype)
    y_neg = torch.randn(64, 8, device=device, dtype=torch.float32).to(dtype)
    v = compute_v(x, y_pos, y_neg, config=config)
    assert v.shape == x.shape
    assert torch.isfinite(v.float()).all()
    assert float(v.abs().max().item()) < 1e6


def test_compute_v_is_finite_across_precisions() -> None:
    _assert_compute_v_is_finite(device=torch.device("cpu"), dtype=torch.float32)
    if torch.cuda.is_available():
        _assert_compute_v_is_finite(device=torch.device("cuda"), dtype=torch.float16)
        _assert_compute_v_is_finite(device=torch.device("cuda"), dtype=torch.bfloat16)
    else:
        _assert_compute_v_is_finite(device=torch.device("cpu"), dtype=torch.bfloat16)


def test_compute_v_is_finite_for_extreme_temperatures() -> None:
    x = torch.randn(32, 8)
    y_pos = torch.randn(48, 8)
    y_neg = torch.randn(64, 8)
    for temperature in (1e-4, 1e-2, 1e2):
        config = DriftFieldConfig(temperature=float(temperature), normalize_over_x=True, mask_self_negatives=True)
        v = compute_v(x, y_pos, y_neg, config=config)
        assert torch.isfinite(v).all()
