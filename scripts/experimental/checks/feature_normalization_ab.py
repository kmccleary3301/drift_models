from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from drifting_models.drift_field import DriftFieldConfig
from drifting_models.drift_loss import DriftingLossConfig, FeatureDriftingConfig, feature_space_drifting_loss
from drifting_models.features import FeatureVectorizationConfig


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

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
        normalize_features=True,
        normalize_drifts=True,
        detach_positive_features=True,
        detach_negative_features=True,
    )

    generated = {"stage0.loc": torch.randn(4, 3, 8, requires_grad=True)}
    positive = {"stage0.loc": torch.randn(5, 3, 8)}
    unconditional = {"stage0.loc": torch.randn(2, 3, 8)}

    loss_off, stats_off = feature_space_drifting_loss(
        generated_feature_vectors=generated,
        positive_feature_vectors=positive,
        unconditional_feature_vectors=unconditional,
        base_loss_config=base_config,
        feature_config=feature_config,
        unconditional_weight=0.0,
    )
    loss_on, stats_on = feature_space_drifting_loss(
        generated_feature_vectors=generated,
        positive_feature_vectors=positive,
        unconditional_feature_vectors=unconditional,
        base_loss_config=base_config,
        feature_config=feature_config,
        unconditional_weight=4.0,
    )

    feature_scale_delta = abs(float(stats_on["mean_feature_scale"]) - float(stats_off["mean_feature_scale"]))
    passed = feature_scale_delta > float(args.min_scale_delta)
    payload = {
        "seed": int(args.seed),
        "loss_unconditional_weight_0": float(loss_off.item()),
        "loss_unconditional_weight_4": float(loss_on.item()),
        "mean_feature_scale_weight_0": float(stats_off["mean_feature_scale"]),
        "mean_feature_scale_weight_4": float(stats_on["mean_feature_scale"]),
        "feature_scale_delta": feature_scale_delta,
        "min_scale_delta": float(args.min_scale_delta),
        "passed": bool(passed),
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rung-2 normalization A/B check for unconditional weighting influence.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-scale-delta", type=float, default=1e-6)
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/stage4_normalization_ab/normalization_ab_check.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
