import torch
from torch import nn
import pytest

from drifting_models.drift_field import DriftFieldConfig
from drifting_models.drift_loss import DriftingLossConfig, FeatureDriftingConfig
from drifting_models.features import FeatureVectorizationConfig
from drifting_models.train.stage2 import GroupedDriftStepConfig, grouped_drift_training_step


class _ScaleGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(  # type: ignore[override]
        self,
        noise: torch.Tensor,
        class_labels: torch.Tensor,
        alpha: torch.Tensor,
        styles: torch.Tensor | None,
    ) -> torch.Tensor:
        del class_labels, alpha, styles
        return noise * self.scale


class _IdentityEncoder(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return images


class _ConvEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=1, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.conv(images)


def test_feature_loss_can_optionally_include_raw_drift_loss() -> None:
    torch.manual_seed(0)
    generator = _ScaleGenerator()
    feature_extractor = _IdentityEncoder()

    loss_config = DriftingLossConfig(
        drift_field=DriftFieldConfig(temperature=0.1, normalize_over_x=True, mask_self_negatives=True),
        attraction_scale=1.0,
        repulsion_scale=1.0,
        stopgrad_target=True,
    )
    feature_config = FeatureDriftingConfig(
        temperatures=(0.05,),
        vectorization=FeatureVectorizationConfig(
            include_per_location=True,
            include_global_stats=True,
            include_patch2_stats=False,
            include_patch4_stats=False,
        ),
        include_raw_drift_loss=True,
        raw_drift_loss_weight=1.0,
    )
    step_config = GroupedDriftStepConfig(
        loss_config=loss_config,
        feature_config=feature_config,
        drift_temperatures=(0.1,),
        drift_temperature_reduction="mean",
        run_optimizer_step=False,
        clip_grad_norm=None,
    )

    g, n, p = 1, 4, 4
    noise_grouped = torch.randn(g, n, 1, 4, 4)
    positives_grouped = torch.randn(g, p, 1, 4, 4)
    class_labels_grouped = torch.zeros(g, dtype=torch.long)
    alpha_grouped = torch.ones(g, dtype=torch.float32)

    stats = grouped_drift_training_step(
        generator=generator,
        optimizer=None,
        noise_grouped=noise_grouped,
        class_labels_grouped=class_labels_grouped,
        alpha_grouped=alpha_grouped,
        positives_grouped=positives_grouped,
        style_indices_grouped=None,
        unconditional_grouped=None,
        unconditional_weight_grouped=None,
        feature_extractor=feature_extractor,
        feature_input_transform=None,
        amp_dtype=None,
        grad_scaler=None,
        backward_when_no_step=False,
        loss_divisor=1.0,
        zero_grad_before_backward=True,
        config=step_config,
    )

    assert "feature_loss" in stats
    assert "raw_loss" in stats
    assert stats["loss"] == pytest.approx(stats["feature_loss"] + stats["raw_loss"], rel=1e-6, abs=1e-6)


def test_feature_extractor_stays_frozen_while_generator_gets_gradients() -> None:
    torch.manual_seed(0)
    generator = _ScaleGenerator()
    feature_extractor = _ConvEncoder()
    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False

    loss_config = DriftingLossConfig(
        drift_field=DriftFieldConfig(temperature=0.1, normalize_over_x=True, mask_self_negatives=True),
        attraction_scale=1.0,
        repulsion_scale=1.0,
        stopgrad_target=True,
    )
    feature_config = FeatureDriftingConfig(
        temperatures=(0.05,),
        vectorization=FeatureVectorizationConfig(
            include_per_location=True,
            include_global_stats=False,
            include_patch2_stats=False,
            include_patch4_stats=False,
        ),
        include_raw_drift_loss=False,
    )
    step_config = GroupedDriftStepConfig(
        loss_config=loss_config,
        feature_config=feature_config,
        drift_temperatures=(0.1,),
        drift_temperature_reduction="mean",
        run_optimizer_step=False,
        clip_grad_norm=None,
    )

    g, n, p = 1, 4, 4
    noise_grouped = torch.randn(g, n, 1, 4, 4)
    positives_grouped = torch.randn(g, p, 1, 4, 4)
    class_labels_grouped = torch.zeros(g, dtype=torch.long)
    alpha_grouped = torch.ones(g, dtype=torch.float32)

    _ = grouped_drift_training_step(
        generator=generator,
        optimizer=None,
        noise_grouped=noise_grouped,
        class_labels_grouped=class_labels_grouped,
        alpha_grouped=alpha_grouped,
        positives_grouped=positives_grouped,
        style_indices_grouped=None,
        unconditional_grouped=None,
        unconditional_weight_grouped=None,
        feature_extractor=feature_extractor,
        feature_input_transform=None,
        amp_dtype=None,
        grad_scaler=None,
        backward_when_no_step=True,
        loss_divisor=1.0,
        zero_grad_before_backward=True,
        config=step_config,
    )

    assert generator.scale.grad is not None
    assert torch.isfinite(generator.scale.grad)
    for parameter in feature_extractor.parameters():
        assert parameter.grad is None
