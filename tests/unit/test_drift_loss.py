import torch

from drifting_models.drift_field import DriftFieldConfig
from drifting_models.drift_loss import DriftingLossConfig, drifting_stopgrad_loss


def test_drifting_stopgrad_loss_matches_expected_value() -> None:
    config = DriftingLossConfig(
        drift_field=DriftFieldConfig(
            temperature=0.1,
            normalize_over_x=False,
            mask_self_negatives=False,
        ),
        attraction_scale=1.0,
        repulsion_scale=1.0,
        stopgrad_target=True,
    )
    x = torch.randn(12, 3, requires_grad=True)
    y_pos = torch.randn(12, 3)
    y_neg = torch.randn(12, 3)

    loss, drift, _ = drifting_stopgrad_loss(x, y_pos, y_neg, config=config)
    manual = torch.mean(drift.pow(2))
    assert torch.allclose(loss, manual, atol=1e-6, rtol=1e-6)


def test_attraction_only_differs_from_antisymmetric_drift() -> None:
    x = torch.randn(10, 2)
    y_pos = torch.randn(10, 2)
    y_neg = torch.randn(10, 2)
    base_config = DriftFieldConfig(temperature=0.05, normalize_over_x=True, mask_self_negatives=False)

    loss_baseline, drift_baseline, _ = drifting_stopgrad_loss(
        x,
        y_pos,
        y_neg,
        config=DriftingLossConfig(
            drift_field=base_config,
            attraction_scale=1.0,
            repulsion_scale=1.0,
            stopgrad_target=True,
        ),
    )
    loss_attraction_only, drift_attraction_only, _ = drifting_stopgrad_loss(
        x,
        y_pos,
        y_neg,
        config=DriftingLossConfig(
            drift_field=base_config,
            attraction_scale=1.0,
            repulsion_scale=0.0,
            stopgrad_target=True,
        ),
    )

    assert not torch.allclose(drift_baseline, drift_attraction_only)
    assert not torch.isclose(loss_baseline, loss_attraction_only)
