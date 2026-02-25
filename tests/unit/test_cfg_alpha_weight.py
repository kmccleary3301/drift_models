import pytest

from drifting_models.drift_field import cfg_alpha_to_unconditional_weight


def test_cfg_alpha_weight_alpha_one_is_zero() -> None:
    assert cfg_alpha_to_unconditional_weight(alpha=1.0, n_generated_negatives=2, n_unconditional_negatives=1) == 0.0


def test_cfg_alpha_weight_scales_with_alpha() -> None:
    w2 = cfg_alpha_to_unconditional_weight(alpha=2.0, n_generated_negatives=4, n_unconditional_negatives=2)
    w3 = cfg_alpha_to_unconditional_weight(alpha=3.0, n_generated_negatives=4, n_unconditional_negatives=2)
    assert w3 > w2


def test_cfg_alpha_weight_validates_inputs() -> None:
    with pytest.raises(ValueError):
        cfg_alpha_to_unconditional_weight(alpha=0.9, n_generated_negatives=2, n_unconditional_negatives=1)
    with pytest.raises(ValueError):
        cfg_alpha_to_unconditional_weight(alpha=2.0, n_generated_negatives=1, n_unconditional_negatives=1)
    with pytest.raises(ValueError):
        cfg_alpha_to_unconditional_weight(alpha=2.0, n_generated_negatives=2, n_unconditional_negatives=0)

