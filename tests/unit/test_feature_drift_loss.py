import torch
from unittest.mock import patch
import pytest

from drifting_models.drift_field import DriftFieldConfig, build_negative_log_weights
from drifting_models.drift_loss import (
    DriftingLossConfig,
    FeatureDriftingConfig,
    _compute_weighted_drift_slot_batched,
    _compute_weighted_drifts_slot_batched_multi_temperature,
    _normalize_features,
    compute_weighted_drift,
    feature_space_drifting_loss,
)
from drifting_models.features import FeatureVectorizationConfig
import drifting_models.drift_loss as drift_loss_module


def test_feature_space_drifting_loss_backward() -> None:
    base_config = DriftingLossConfig(
        drift_field=DriftFieldConfig(
            temperature=0.05,
            normalize_over_x=True,
            mask_self_negatives=False,
        ),
        attraction_scale=1.0,
        repulsion_scale=1.0,
        stopgrad_target=True,
    )
    feature_config = FeatureDriftingConfig(
        temperatures=(0.02, 0.05),
        vectorization=FeatureVectorizationConfig(
            include_per_location=True,
            include_global_stats=True,
            include_patch2_stats=False,
            include_patch4_stats=False,
        ),
        normalize_features=True,
        normalize_drifts=True,
        detach_positive_features=True,
        detach_negative_features=True,
    )
    generated = {"stage0.loc": torch.randn(4, 3, 6, requires_grad=True)}
    positive = {"stage0.loc": torch.randn(5, 3, 6)}

    loss, stats = feature_space_drifting_loss(
        generated_feature_vectors=generated,
        positive_feature_vectors=positive,
        unconditional_feature_vectors=None,
        base_loss_config=base_config,
        feature_config=feature_config,
    )
    loss.backward()

    assert loss.item() >= 0.0
    assert stats["vector_count"] == 3.0
    assert stats["mean_feature_scale"] > 0.0
    assert stats["mean_drift_scale"] > 0.0
    assert "temp_0.02_drift_norm" in stats
    assert "temp_0.05_drift_norm" in stats
    assert "temp_0.02_drift_scale" in stats
    assert "temp_0.05_drift_scale" in stats
    assert generated["stage0.loc"].grad is not None


def test_feature_space_drifting_loss_supports_unconditional_negatives() -> None:
    base_config = DriftingLossConfig(
        drift_field=DriftFieldConfig(
            temperature=0.05,
            normalize_over_x=True,
            mask_self_negatives=True,
        ),
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
        normalize_features=True,
        normalize_drifts=True,
        detach_positive_features=True,
        detach_negative_features=True,
    )
    generated = {"stage0.loc": torch.randn(3, 2, 5, requires_grad=True)}
    positive = {"stage0.loc": torch.randn(4, 2, 5)}
    unconditional = {"stage0.loc": torch.randn(2, 2, 5)}

    loss, stats = feature_space_drifting_loss(
        generated_feature_vectors=generated,
        positive_feature_vectors=positive,
        unconditional_feature_vectors=unconditional,
        base_loss_config=base_config,
        feature_config=feature_config,
        unconditional_weight=2.0,
    )
    loss.backward()
    assert stats["loss"] >= 0.0
    assert generated["stage0.loc"].grad is not None


def test_feature_space_drifting_share_location_toggle() -> None:
    base_config = DriftingLossConfig(
        drift_field=DriftFieldConfig(temperature=0.05, normalize_over_x=True, mask_self_negatives=True),
        attraction_scale=1.0,
        repulsion_scale=1.0,
        stopgrad_target=True,
    )
    generated = {"stage0.loc": torch.randn(3, 4, 5, requires_grad=True)}
    positive = {"stage0.loc": torch.randn(3, 4, 5)}

    shared_config = FeatureDriftingConfig(
        temperatures=(0.05,),
        vectorization=FeatureVectorizationConfig(
            include_per_location=True,
            include_global_stats=False,
            include_patch2_stats=False,
            include_patch4_stats=False,
        ),
        share_location_normalization=True,
    )
    per_location_config = FeatureDriftingConfig(
        temperatures=(0.05,),
        vectorization=FeatureVectorizationConfig(
            include_per_location=True,
            include_global_stats=False,
            include_patch2_stats=False,
            include_patch4_stats=False,
        ),
        share_location_normalization=False,
    )

    _, shared_stats = feature_space_drifting_loss(
        generated_feature_vectors=generated,
        positive_feature_vectors=positive,
        unconditional_feature_vectors=None,
        base_loss_config=base_config,
        feature_config=shared_config,
    )
    _, per_loc_stats = feature_space_drifting_loss(
        generated_feature_vectors=generated,
        positive_feature_vectors=positive,
        unconditional_feature_vectors=None,
        base_loss_config=base_config,
        feature_config=per_location_config,
    )
    assert shared_stats["mean_feature_scale"] > 0.0
    assert per_loc_stats["mean_feature_scale"] > 0.0
    assert shared_stats["mean_drift_scale"] > 0.0
    assert per_loc_stats["mean_drift_scale"] > 0.0


def test_feature_space_drifting_temperature_sqrt_scaling_toggle() -> None:
    base_config = DriftingLossConfig(
        drift_field=DriftFieldConfig(temperature=0.05, normalize_over_x=True, mask_self_negatives=False),
        attraction_scale=1.0,
        repulsion_scale=1.0,
        stopgrad_target=True,
    )
    generated = {"stage0.loc": torch.randn(3, 4, 16, requires_grad=True)}
    positive = {"stage0.loc": torch.randn(3, 4, 16)}

    with_scaling = FeatureDriftingConfig(
        temperatures=(0.05,),
        vectorization=FeatureVectorizationConfig(
            include_per_location=True,
            include_global_stats=False,
            include_patch2_stats=False,
            include_patch4_stats=False,
        ),
        normalize_drifts=False,
        scale_temperature_by_sqrt_channels=True,
    )
    without_scaling = FeatureDriftingConfig(
        temperatures=(0.05,),
        vectorization=FeatureVectorizationConfig(
            include_per_location=True,
            include_global_stats=False,
            include_patch2_stats=False,
            include_patch4_stats=False,
        ),
        normalize_drifts=False,
        scale_temperature_by_sqrt_channels=False,
    )

    loss_a, _ = feature_space_drifting_loss(
        generated_feature_vectors=generated,
        positive_feature_vectors=positive,
        unconditional_feature_vectors=None,
        base_loss_config=base_config,
        feature_config=with_scaling,
    )
    loss_b, _ = feature_space_drifting_loss(
        generated_feature_vectors=generated,
        positive_feature_vectors=positive,
        unconditional_feature_vectors=None,
        base_loss_config=base_config,
        feature_config=without_scaling,
    )
    assert float(loss_a.item()) != float(loss_b.item())


def test_feature_temperature_sqrt_scaling_stabilizes_channel_padding() -> None:
    torch.manual_seed(0)
    base_config = DriftingLossConfig(
        drift_field=DriftFieldConfig(temperature=0.05, normalize_over_x=True, mask_self_negatives=True),
        attraction_scale=1.0,
        repulsion_scale=1.0,
        stopgrad_target=True,
    )

    b, v, c_small, c_big = 4, 3, 8, 32
    base_generated = torch.randn(b, v, c_small)
    base_positive = torch.randn(b, v, c_small)

    generated_small = {"stage0.loc": base_generated.clone().requires_grad_(True)}
    positive_small = {"stage0.loc": base_positive.clone()}

    pad = c_big - c_small
    generated_big = {
        "stage0.loc": torch.cat([base_generated, torch.zeros(b, v, pad)], dim=-1).clone().requires_grad_(True)
    }
    positive_big = {"stage0.loc": torch.cat([base_positive, torch.zeros(b, v, pad)], dim=-1).clone()}

    def loss_for(scale_by_sqrt: bool, generated: dict[str, torch.Tensor], positive: dict[str, torch.Tensor]) -> float:
        feature_config = FeatureDriftingConfig(
            temperatures=(0.05,),
            vectorization=FeatureVectorizationConfig(
                include_per_location=True,
                include_global_stats=False,
                include_patch2_stats=False,
                include_patch4_stats=False,
            ),
            normalize_features=True,
            normalize_drifts=False,
            scale_temperature_by_sqrt_channels=scale_by_sqrt,
            share_location_normalization=True,
            detach_positive_features=True,
            detach_negative_features=True,
        )
        loss, _ = feature_space_drifting_loss(
            generated_feature_vectors=generated,
            positive_feature_vectors=positive,
            unconditional_feature_vectors=None,
            base_loss_config=base_config,
            feature_config=feature_config,
        )
        return float(loss.item())

    small_scaled = loss_for(True, generated_small, positive_small)
    big_scaled = loss_for(True, generated_big, positive_big)
    small_unscaled = loss_for(False, generated_small, positive_small)
    big_unscaled = loss_for(False, generated_big, positive_big)

    assert abs(small_scaled - big_scaled) < 1e-3
    assert abs(small_unscaled - big_unscaled) > 1e-5


def test_feature_temperature_aggregation_modes_differ() -> None:
    base_config = DriftingLossConfig(
        drift_field=DriftFieldConfig(temperature=0.05, normalize_over_x=True, mask_self_negatives=True),
        attraction_scale=1.0,
        repulsion_scale=1.0,
        stopgrad_target=True,
    )
    generated = {"stage0.loc": torch.randn(4, 3, 8, requires_grad=True)}
    positive = {"stage0.loc": torch.randn(4, 3, 8)}

    per_temp = FeatureDriftingConfig(
        temperatures=(0.02, 0.05, 0.2),
        vectorization=FeatureVectorizationConfig(
            include_per_location=True,
            include_global_stats=False,
            include_patch2_stats=False,
            include_patch4_stats=False,
        ),
        normalize_features=True,
        normalize_drifts=True,
        temperature_aggregation="per_temperature_mse",
    )
    summed = FeatureDriftingConfig(
        temperatures=(0.02, 0.05, 0.2),
        vectorization=FeatureVectorizationConfig(
            include_per_location=True,
            include_global_stats=False,
            include_patch2_stats=False,
            include_patch4_stats=False,
        ),
        normalize_features=True,
        normalize_drifts=True,
        temperature_aggregation="sum_drifts_then_mse",
    )

    loss_a, _ = feature_space_drifting_loss(
        generated_feature_vectors=generated,
        positive_feature_vectors=positive,
        unconditional_feature_vectors=None,
        base_loss_config=base_config,
        feature_config=per_temp,
    )
    loss_b, _ = feature_space_drifting_loss(
        generated_feature_vectors=generated,
        positive_feature_vectors=positive,
        unconditional_feature_vectors=None,
        base_loss_config=base_config,
        feature_config=summed,
    )
    assert float(loss_a.item()) != float(loss_b.item())


def test_feature_normalization_is_scale_invariant_when_enabled() -> None:
    torch.manual_seed(0)
    base_config = DriftingLossConfig(
        drift_field=DriftFieldConfig(temperature=0.05, normalize_over_x=True, mask_self_negatives=True),
        attraction_scale=1.0,
        repulsion_scale=1.0,
        stopgrad_target=True,
    )
    scale = 7.0
    generated_a = {"stage0.loc": torch.randn(5, 4, 8, requires_grad=True)}
    positive_a = {"stage0.loc": torch.randn(6, 4, 8)}
    generated_b = {"stage0.loc": (generated_a["stage0.loc"].detach() * scale).requires_grad_(True)}
    positive_b = {"stage0.loc": positive_a["stage0.loc"] * scale}

    for share_location_normalization in (False, True):
        feature_config = FeatureDriftingConfig(
            temperatures=(0.05,),
            vectorization=FeatureVectorizationConfig(
                include_per_location=True,
                include_global_stats=False,
                include_patch2_stats=False,
                include_patch4_stats=False,
            ),
            normalize_features=True,
            normalize_drifts=True,
            share_location_normalization=share_location_normalization,
        )
        loss_a, stats_a = feature_space_drifting_loss(
            generated_feature_vectors=generated_a,
            positive_feature_vectors=positive_a,
            unconditional_feature_vectors=None,
            base_loss_config=base_config,
            feature_config=feature_config,
        )
        loss_b, stats_b = feature_space_drifting_loss(
            generated_feature_vectors=generated_b,
            positive_feature_vectors=positive_b,
            unconditional_feature_vectors=None,
            base_loss_config=base_config,
            feature_config=feature_config,
        )

        assert abs(float(loss_a.item()) - float(loss_b.item())) < 1e-4

        assert float(stats_a["mean_feature_scale"]) > 0.0
        assert float(stats_b["mean_feature_scale"]) > 0.0
        assert float(stats_a["mean_drift_scale"]) > 0.0
        assert float(stats_b["mean_drift_scale"]) > 0.0

        feature_scale_ratio = float(stats_b["mean_feature_scale"]) / float(stats_a["mean_feature_scale"])
        drift_scale_ratio = float(stats_b["mean_drift_scale"]) / float(stats_a["mean_drift_scale"])
        assert abs(feature_scale_ratio - scale) < 1e-2
        assert abs(drift_scale_ratio - 1.0) < 1e-2


def test_shared_location_normalization_uses_slotwise_average_scale() -> None:
    x = torch.tensor(
        [
            [[0.0, 0.0], [10.0, 0.0]],
            [[1.0, 0.0], [12.0, 0.0]],
        ]
    )
    y = torch.tensor(
        [
            [[0.0, 0.0], [100.0, 0.0]],
            [[2.0, 0.0], [120.0, 0.0]],
        ]
    )

    _, _, shared_scales = _normalize_features(
        x,
        y,
        share_location_normalization=True,
        eps=1e-8,
    )

    channel_sqrt = x.shape[-1] ** 0.5
    slotwise_mean = torch.stack(
        [
            torch.cdist(x[:, 0, :], y[:, 0, :]).mean(),
            torch.cdist(x[:, 1, :], y[:, 1, :]).mean(),
        ]
    ).mean() / channel_sqrt
    flattened_mean = torch.cdist(x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])).mean() / channel_sqrt

    assert torch.allclose(shared_scales, shared_scales[0].repeat(x.shape[1]))
    assert abs(float(shared_scales[0].item()) - float(slotwise_mean.item())) < 1e-6
    assert abs(float(shared_scales[0].item()) - float(flattened_mean.item())) > 1e-3


def test_feature_loss_term_reduction_sum_scales_with_term_count() -> None:
    torch.manual_seed(0)
    base_config = DriftingLossConfig(
        drift_field=DriftFieldConfig(temperature=0.05, normalize_over_x=True, mask_self_negatives=True),
        attraction_scale=1.0,
        repulsion_scale=1.0,
        stopgrad_target=True,
    )
    generated = {
        "stage0.loc": torch.randn(3, 2, 6, requires_grad=True),
        "stage1.loc": torch.randn(3, 2, 6, requires_grad=True),
    }
    positive = {
        "stage0.loc": torch.randn(4, 2, 6),
        "stage1.loc": torch.randn(4, 2, 6),
    }
    sum_config = FeatureDriftingConfig(
        temperatures=(0.05,),
        vectorization=FeatureVectorizationConfig(
            include_per_location=True,
            include_global_stats=False,
            include_patch2_stats=False,
            include_patch4_stats=False,
        ),
        normalize_features=False,
        normalize_drifts=False,
        temperature_aggregation="per_temperature_mse",
        loss_term_reduction="sum",
    )
    mean_config = FeatureDriftingConfig(
        temperatures=(0.05,),
        vectorization=FeatureVectorizationConfig(
            include_per_location=True,
            include_global_stats=False,
            include_patch2_stats=False,
            include_patch4_stats=False,
        ),
        normalize_features=False,
        normalize_drifts=False,
        temperature_aggregation="per_temperature_mse",
        loss_term_reduction="mean",
    )

    loss_sum, stats_sum = feature_space_drifting_loss(
        generated_feature_vectors=generated,
        positive_feature_vectors=positive,
        unconditional_feature_vectors=None,
        base_loss_config=base_config,
        feature_config=sum_config,
    )
    loss_mean, stats_mean = feature_space_drifting_loss(
        generated_feature_vectors=generated,
        positive_feature_vectors=positive,
        unconditional_feature_vectors=None,
        base_loss_config=base_config,
        feature_config=mean_config,
    )

    expected_terms = 4.0  # 2 keys * 2 vector slots * 1 temperature term
    assert stats_sum["loss_term_count"] == expected_terms
    assert stats_mean["loss_term_count"] == expected_terms
    assert torch.allclose(loss_sum, loss_mean * expected_terms, atol=1e-5, rtol=1e-5)


def test_feature_normalization_scale_includes_negative_vectors() -> None:
    x = torch.zeros(4, 2, 3)
    y_pos = torch.ones(5, 2, 3)
    y_neg_close = torch.zeros(4, 2, 3)
    y_neg_far = torch.full((4, 2, 3), 10.0)

    _, _, scales_close = _normalize_features(
        x,
        y_pos,
        y_neg_vectors=y_neg_close,
        share_location_normalization=True,
        eps=1e-8,
    )
    _, _, scales_far = _normalize_features(
        x,
        y_pos,
        y_neg_vectors=y_neg_far,
        share_location_normalization=True,
        eps=1e-8,
    )
    assert float(scales_far.mean().item()) > float(scales_close.mean().item())


def test_feature_normalization_scale_responds_to_unconditional_weight() -> None:
    x = torch.zeros(4, 2, 3)
    y_pos = torch.ones(4, 2, 3)
    y_neg_generated = torch.zeros(4, 2, 3)
    y_neg_unconditional = torch.full((2, 2, 3), 20.0)
    y_neg_all = torch.cat([y_neg_generated, y_neg_unconditional], dim=0)

    weights_off = build_negative_log_weights(
        n_generated_negatives=4,
        n_unconditional_negatives=2,
        unconditional_weight=0.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    weights_on = build_negative_log_weights(
        n_generated_negatives=4,
        n_unconditional_negatives=2,
        unconditional_weight=4.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    _, _, scales_off = _normalize_features(
        x,
        y_pos,
        y_neg_vectors=y_neg_all,
        negative_log_weights=weights_off,
        share_location_normalization=True,
        eps=1e-8,
    )
    _, _, scales_on = _normalize_features(
        x,
        y_pos,
        y_neg_vectors=y_neg_all,
        negative_log_weights=weights_on,
        share_location_normalization=True,
        eps=1e-8,
    )
    assert float(scales_on.mean().item()) > float(scales_off.mean().item())


def test_slot_batched_weighted_drift_matches_per_slot_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 3, 6)
    y_pos = torch.randn(5, 3, 6)
    y_unc = torch.randn(2, 3, 6)
    y_neg_generated = x.clone()
    y_neg = torch.cat([y_neg_generated, y_unc], dim=0)
    negative_log_weights = build_negative_log_weights(
        n_generated_negatives=y_neg_generated.shape[0],
        n_unconditional_negatives=y_unc.shape[0],
        unconditional_weight=2.0,
        device=x.device,
        dtype=x.dtype,
    )

    for normalize_over_x in (False, True):
        config = DriftingLossConfig(
            drift_field=DriftFieldConfig(
                temperature=0.07,
                normalize_over_x=normalize_over_x,
                mask_self_negatives=True,
            ),
            attraction_scale=1.0,
            repulsion_scale=1.0,
            stopgrad_target=True,
        )
        batched = _compute_weighted_drift_slot_batched(
            x_vectors=x,
            y_pos_vectors=y_pos,
            y_neg_vectors=y_neg,
            config=config,
            negative_log_weights=negative_log_weights,
            generated_negative_count=y_neg_generated.shape[0],
        )
        reference = torch.zeros_like(batched)
        for slot_index in range(x.shape[1]):
            drift_slot, _ = compute_weighted_drift(
                x=x[:, slot_index, :],
                y_pos=y_pos[:, slot_index, :],
                y_neg=y_neg[:, slot_index, :],
                config=config,
                negative_log_weights=negative_log_weights,
                generated_negative_count=y_neg_generated.shape[0],
            )
            reference[:, slot_index, :] = drift_slot

        assert torch.allclose(batched, reference, atol=1e-6, rtol=1e-6)


def test_slot_batched_multi_temperature_matches_single_calls() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 3, 6)
    y_pos = torch.randn(5, 3, 6)
    y_unc = torch.randn(2, 3, 6)
    y_neg_generated = x.clone()
    y_neg = torch.cat([y_neg_generated, y_unc], dim=0)
    negative_log_weights = build_negative_log_weights(
        n_generated_negatives=y_neg_generated.shape[0],
        n_unconditional_negatives=y_unc.shape[0],
        unconditional_weight=2.0,
        device=x.device,
        dtype=x.dtype,
    )
    temperatures = (0.02, 0.05, 0.2)

    for normalize_over_x in (False, True):
        for scale_by_sqrt in (False, True):
            base_config = DriftingLossConfig(
                drift_field=DriftFieldConfig(
                    temperature=0.07,
                    normalize_over_x=normalize_over_x,
                    mask_self_negatives=True,
                ),
                attraction_scale=1.0,
                repulsion_scale=1.0,
                stopgrad_target=True,
            )
            multi = _compute_weighted_drifts_slot_batched_multi_temperature(
                x_vectors=x,
                y_pos_vectors=y_pos,
                y_neg_vectors=y_neg,
                temperatures=temperatures,
                base_config=base_config,
                scale_temperature_by_sqrt_channels=scale_by_sqrt,
                negative_log_weights=negative_log_weights,
                generated_negative_count=y_neg_generated.shape[0],
            )
            assert [temp for temp, _ in multi] == [float(temp) for temp in temperatures]
            for temperature, drift in multi:
                effective_temperature = float(temperature)
                if scale_by_sqrt:
                    effective_temperature = effective_temperature * (x.shape[-1] ** 0.5)
                single_config = DriftingLossConfig(
                    drift_field=DriftFieldConfig(
                        temperature=effective_temperature,
                        normalize_over_x=normalize_over_x,
                        mask_self_negatives=True,
                    ),
                    attraction_scale=1.0,
                    repulsion_scale=1.0,
                    stopgrad_target=True,
                )
                single = _compute_weighted_drift_slot_batched(
                    x_vectors=x,
                    y_pos_vectors=y_pos,
                    y_neg_vectors=y_neg,
                    config=single_config,
                    negative_log_weights=negative_log_weights,
                    generated_negative_count=y_neg_generated.shape[0],
                )
                assert torch.allclose(drift, single, atol=1e-6, rtol=1e-6)


def test_slot_batched_multi_temperature_reuses_pairwise_distances() -> None:
    x = torch.randn(3, 2, 4)
    y_pos = torch.randn(5, 2, 4)
    y_neg = torch.randn(4, 2, 4)
    base_config = DriftingLossConfig(
        drift_field=DriftFieldConfig(temperature=0.05, normalize_over_x=True, mask_self_negatives=True),
        attraction_scale=1.0,
        repulsion_scale=1.0,
        stopgrad_target=True,
    )

    with patch("drifting_models.drift_loss.torch.cdist", wraps=torch.cdist) as cdist_mock:
        _ = _compute_weighted_drifts_slot_batched_multi_temperature(
            x_vectors=x,
            y_pos_vectors=y_pos,
            y_neg_vectors=y_neg,
            temperatures=(0.02, 0.05, 0.2),
            base_config=base_config,
            scale_temperature_by_sqrt_channels=True,
            negative_log_weights=None,
            generated_negative_count=x.shape[0],
        )
    assert cdist_mock.call_count == 2


def test_feature_kernel_compile_warn_falls_back_to_eager(monkeypatch) -> None:
    drift_loss_module._COMPILED_MULTI_TEMP_KERNELS.clear()
    drift_loss_module._COMPILE_MULTI_TEMP_FAILURES.clear()

    def _raise_compile(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("compile boom")

    monkeypatch.setattr(drift_loss_module.torch, "compile", _raise_compile)
    base_config = DriftingLossConfig(
        drift_field=DriftFieldConfig(temperature=0.05, normalize_over_x=True, mask_self_negatives=True),
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
        compile_drift_kernel=True,
        compile_drift_fail_action="warn",
    )
    generated = {"stage0.loc": torch.randn(4, 2, 6, requires_grad=True)}
    positive = {"stage0.loc": torch.randn(5, 2, 6)}

    loss, _ = feature_space_drifting_loss(
        generated_feature_vectors=generated,
        positive_feature_vectors=positive,
        unconditional_feature_vectors=None,
        base_loss_config=base_config,
        feature_config=feature_config,
    )
    assert torch.isfinite(loss).all()
    assert drift_loss_module._COMPILE_MULTI_TEMP_FAILURES


def test_feature_kernel_compile_raise_propagates(monkeypatch) -> None:
    drift_loss_module._COMPILED_MULTI_TEMP_KERNELS.clear()
    drift_loss_module._COMPILE_MULTI_TEMP_FAILURES.clear()

    def _raise_compile(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("compile boom")

    monkeypatch.setattr(drift_loss_module.torch, "compile", _raise_compile)
    base_config = DriftingLossConfig(
        drift_field=DriftFieldConfig(temperature=0.05, normalize_over_x=True, mask_self_negatives=True),
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
        compile_drift_kernel=True,
        compile_drift_fail_action="raise",
    )
    generated = {"stage0.loc": torch.randn(4, 2, 6, requires_grad=True)}
    positive = {"stage0.loc": torch.randn(5, 2, 6)}

    try:
        _ = feature_space_drifting_loss(
            generated_feature_vectors=generated,
            positive_feature_vectors=positive,
            unconditional_feature_vectors=None,
            base_loss_config=base_config,
            feature_config=feature_config,
        )
    except RuntimeError as error:
        assert "compile boom" in str(error)
    else:
        raise AssertionError("Expected RuntimeError when compile_drift_fail_action=raise")


def test_compiled_multi_temperature_kernel_numeric_parity_when_available() -> None:
    compile_fn = getattr(drift_loss_module.torch, "compile", None)
    if compile_fn is None:
        pytest.skip("torch.compile unavailable")
    x = torch.randn(3, 2, 4)
    y_pos = torch.randn(4, 2, 4)
    y_neg = torch.randn(5, 2, 4)
    base_config = DriftingLossConfig(
        drift_field=DriftFieldConfig(temperature=0.05, normalize_over_x=True, mask_self_negatives=True),
        attraction_scale=1.0,
        repulsion_scale=1.0,
        stopgrad_target=True,
    )
    temperatures = (0.02, 0.05, 0.2)
    eager = _compute_weighted_drifts_slot_batched_multi_temperature(
        x_vectors=x,
        y_pos_vectors=y_pos,
        y_neg_vectors=y_neg,
        temperatures=temperatures,
        base_config=base_config,
        scale_temperature_by_sqrt_channels=True,
        negative_log_weights=None,
        generated_negative_count=x.shape[0],
    )
    try:
        compiled_kernel = compile_fn(
            _compute_weighted_drifts_slot_batched_multi_temperature,
            backend="inductor",
            mode="reduce-overhead",
            dynamic=False,
            fullgraph=False,
        )
        compiled = compiled_kernel(
            x_vectors=x,
            y_pos_vectors=y_pos,
            y_neg_vectors=y_neg,
            temperatures=temperatures,
            base_config=base_config,
            scale_temperature_by_sqrt_channels=True,
            negative_log_weights=None,
            generated_negative_count=x.shape[0],
        )
    except Exception as error:
        pytest.skip(f"compile backend unavailable in test runtime: {error}")

    assert len(eager) == len(compiled)
    for (_, drift_eager), (_, drift_compiled) in zip(eager, compiled):
        assert torch.allclose(drift_eager, drift_compiled, atol=1e-5, rtol=1e-5)


def test_compiled_multi_temperature_kernel_repeatability_when_available() -> None:
    compile_fn = getattr(drift_loss_module.torch, "compile", None)
    if compile_fn is None:
        pytest.skip("torch.compile unavailable")
    x = torch.randn(3, 2, 4)
    y_pos = torch.randn(4, 2, 4)
    y_neg = torch.randn(5, 2, 4)
    base_config = DriftingLossConfig(
        drift_field=DriftFieldConfig(temperature=0.05, normalize_over_x=True, mask_self_negatives=True),
        attraction_scale=1.0,
        repulsion_scale=1.0,
        stopgrad_target=True,
    )
    temperatures = (0.02, 0.05)
    try:
        compiled_kernel = compile_fn(
            _compute_weighted_drifts_slot_batched_multi_temperature,
            backend="inductor",
            mode="reduce-overhead",
            dynamic=False,
            fullgraph=False,
        )
        first = compiled_kernel(
            x_vectors=x,
            y_pos_vectors=y_pos,
            y_neg_vectors=y_neg,
            temperatures=temperatures,
            base_config=base_config,
            scale_temperature_by_sqrt_channels=True,
            negative_log_weights=None,
            generated_negative_count=x.shape[0],
        )
        second = compiled_kernel(
            x_vectors=x,
            y_pos_vectors=y_pos,
            y_neg_vectors=y_neg,
            temperatures=temperatures,
            base_config=base_config,
            scale_temperature_by_sqrt_channels=True,
            negative_log_weights=None,
            generated_negative_count=x.shape[0],
        )
    except Exception as error:
        pytest.skip(f"compile backend unavailable in test runtime: {error}")

    for (_, drift_first), (_, drift_second) in zip(first, second):
        assert torch.allclose(drift_first, drift_second, atol=1e-6, rtol=1e-6)
