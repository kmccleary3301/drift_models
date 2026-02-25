import torch
from contextlib import nullcontext

from drifting_models.drift_field import DriftFieldConfig
from drifting_models.drift_loss import (
    DriftingLossConfig,
    FeatureDriftingConfig,
    drifting_stopgrad_loss,
    feature_space_drifting_loss,
)
from drifting_models.features import FeatureVectorizationConfig


def _cases() -> list[tuple[torch.device, torch.dtype | None]]:
    cases: list[tuple[torch.device, torch.dtype | None]] = [(torch.device("cpu"), None)]
    if torch.cuda.is_available():
        cases.append((torch.device("cuda"), torch.float16))
        cases.append((torch.device("cuda"), torch.bfloat16))
    return cases


def _maybe_autocast(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def test_drifting_stopgrad_loss_is_finite_across_precisions() -> None:
    torch.manual_seed(0)
    config = DriftingLossConfig(
        drift_field=DriftFieldConfig(temperature=0.05, normalize_over_x=True, mask_self_negatives=True),
        attraction_scale=1.0,
        repulsion_scale=1.0,
        stopgrad_target=True,
    )
    for device, amp_dtype in _cases():
        with _maybe_autocast(device, amp_dtype):
            x = torch.randn(16, 8, device=device, dtype=torch.float32).requires_grad_(True)
            y_pos = torch.randn(24, 8, device=device, dtype=torch.float32)
            y_neg = torch.randn(24, 8, device=device, dtype=torch.float32)
            loss, drift, stats = drifting_stopgrad_loss(x, y_pos, y_neg, config=config)
        assert torch.isfinite(loss.float()).all()
        assert torch.isfinite(drift.float()).all()
        assert isinstance(stats.get("loss"), float)
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad.float()).all()


def test_feature_space_drifting_loss_is_finite_across_precisions() -> None:
    torch.manual_seed(0)
    base_config = DriftingLossConfig(
        drift_field=DriftFieldConfig(temperature=0.05, normalize_over_x=True, mask_self_negatives=True),
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
    for device, amp_dtype in _cases():
        with _maybe_autocast(device, amp_dtype):
            generated = {"stage0.loc": torch.randn(6, 3, 16, device=device, dtype=torch.float32).requires_grad_(True)}
            positive = {"stage0.loc": torch.randn(7, 3, 16, device=device, dtype=torch.float32)}
            unconditional = {"stage0.loc": torch.randn(4, 3, 16, device=device, dtype=torch.float32)}
            loss, stats = feature_space_drifting_loss(
                generated_feature_vectors=generated,
                positive_feature_vectors=positive,
                unconditional_feature_vectors=unconditional,
                base_loss_config=base_config,
                feature_config=feature_config,
                unconditional_weight=2.0,
            )
        assert torch.isfinite(loss.float()).all()
        assert isinstance(stats.get("loss"), float)
        loss.backward()
        assert generated["stage0.loc"].grad is not None
        assert torch.isfinite(generated["stage0.loc"].grad.float()).all()
