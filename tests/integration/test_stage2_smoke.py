import torch

from drifting_models.drift_field import DriftFieldConfig
from drifting_models.drift_loss import DriftingLossConfig, FeatureDriftingConfig
from drifting_models.features import FeatureVectorizationConfig, TinyFeatureEncoder, TinyFeatureEncoderConfig
from drifting_models.models import DiTLikeConfig, DiTLikeGenerator
from drifting_models.train import GroupedDriftStepConfig, grouped_drift_training_step


def test_grouped_stage2_step_smoke() -> None:
    torch.manual_seed(7)
    model = DiTLikeGenerator(
        DiTLikeConfig(
            image_size=16,
            in_channels=4,
            out_channels=4,
            patch_size=4,
            hidden_dim=64,
            depth=2,
            num_heads=4,
            register_tokens=4,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    step_config = GroupedDriftStepConfig(
        loss_config=DriftingLossConfig(
            drift_field=DriftFieldConfig(
                temperature=0.05,
                normalize_over_x=True,
                mask_self_negatives=True,
            ),
            attraction_scale=1.0,
            repulsion_scale=1.0,
            stopgrad_target=True,
        ),
        clip_grad_norm=2.0,
        run_optimizer_step=True,
    )

    groups = 3
    negatives_per_group = 2
    positives_per_group = 2
    noise_grouped = torch.randn(groups, negatives_per_group, 4, 16, 16)
    positives_grouped = torch.randn(groups, positives_per_group, 4, 16, 16)
    class_labels = torch.randint(0, 1000, (groups,))
    alpha = torch.tensor([1.0, 2.0, 3.0])
    style_indices = torch.randint(0, 64, (groups, negatives_per_group, 32))
    unconditional_grouped = torch.randn(groups, 2, 4, 16, 16)
    unconditional_weight_grouped = torch.tensor([1.0, 2.0, 1.5])

    first_param_before = next(model.parameters()).detach().clone()
    stats = grouped_drift_training_step(
        generator=model,
        optimizer=optimizer,
        noise_grouped=noise_grouped,
        class_labels_grouped=class_labels,
        alpha_grouped=alpha,
        positives_grouped=positives_grouped,
        style_indices_grouped=style_indices,
        unconditional_grouped=unconditional_grouped,
        unconditional_weight_grouped=unconditional_weight_grouped,
        feature_extractor=None,
        config=step_config,
    )
    first_param_after = next(model.parameters()).detach().clone()

    assert stats["groups"] == groups
    assert stats["negatives_per_group"] == negatives_per_group
    assert stats["loss"] >= 0.0
    assert stats["mean_drift_norm"] >= 0.0
    assert stats["alpha_min"] >= 1.0
    assert stats["alpha_max"] >= stats["alpha_min"]
    assert stats["mean_unconditional_weight"] > 0.0
    assert 0.0 <= stats["mean_unconditional_negative_fraction"] <= 1.0
    assert stats["grad_norm"] is not None
    assert not torch.equal(first_param_before, first_param_after)


def test_grouped_stage2_step_smoke_feature_mode() -> None:
    torch.manual_seed(11)
    model = DiTLikeGenerator(
        DiTLikeConfig(
            image_size=16,
            in_channels=4,
            out_channels=4,
            patch_size=4,
            hidden_dim=64,
            depth=2,
            num_heads=4,
            register_tokens=4,
        )
    )
    feature_extractor = TinyFeatureEncoder(
        TinyFeatureEncoderConfig(in_channels=4, base_channels=8, stages=2)
    )
    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False
    feature_extractor.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    step_config = GroupedDriftStepConfig(
        loss_config=DriftingLossConfig(
            drift_field=DriftFieldConfig(
                temperature=0.05,
                normalize_over_x=True,
                mask_self_negatives=True,
            ),
            attraction_scale=1.0,
            repulsion_scale=1.0,
            stopgrad_target=True,
        ),
        feature_config=FeatureDriftingConfig(
            temperatures=(0.05,),
            vectorization=FeatureVectorizationConfig(
                include_per_location=True,
                include_global_stats=True,
                include_patch2_stats=True,
                include_patch4_stats=False,
            ),
            normalize_features=True,
            normalize_drifts=True,
            detach_positive_features=True,
            detach_negative_features=True,
        ),
        clip_grad_norm=2.0,
        run_optimizer_step=True,
    )

    groups = 2
    negatives_per_group = 2
    positives_per_group = 2
    noise_grouped = torch.randn(groups, negatives_per_group, 4, 16, 16)
    positives_grouped = torch.randn(groups, positives_per_group, 4, 16, 16)
    class_labels = torch.randint(0, 1000, (groups,))
    alpha = torch.tensor([1.0, 2.0])
    style_indices = torch.randint(0, 64, (groups, negatives_per_group, 32))
    unconditional_grouped = torch.randn(groups, 2, 4, 16, 16)
    unconditional_weight_grouped = torch.tensor([1.0, 1.5])

    stats = grouped_drift_training_step(
        generator=model,
        optimizer=optimizer,
        noise_grouped=noise_grouped,
        class_labels_grouped=class_labels,
        alpha_grouped=alpha,
        positives_grouped=positives_grouped,
        style_indices_grouped=style_indices,
        unconditional_grouped=unconditional_grouped,
        unconditional_weight_grouped=unconditional_weight_grouped,
        feature_extractor=feature_extractor,
        config=step_config,
    )
    assert stats["loss"] >= 0.0
    assert stats["mean_drift_norm"] >= 0.0
    assert stats["alpha_min"] >= 1.0
    assert stats["alpha_max"] >= stats["alpha_min"]
    assert stats["mean_unconditional_weight"] > 0.0
    assert 0.0 <= stats["mean_unconditional_negative_fraction"] <= 1.0
    assert stats["mean_feature_scale"] > 0.0
    assert stats["mean_drift_scale"] > 0.0
    assert "temp_0.05_drift_norm" in stats
    assert "temp_0.05_drift_scale" in stats
