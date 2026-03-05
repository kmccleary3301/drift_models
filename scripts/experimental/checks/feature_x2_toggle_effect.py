from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from drifting_models.drift_field import DriftFieldConfig
from drifting_models.drift_loss import DriftingLossConfig, FeatureDriftingConfig, feature_space_drifting_loss
from drifting_models.features import FeatureVectorizationConfig, vectorize_feature_maps


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    generated_images = torch.randn(4, 3, 8, 8)
    positive_images = torch.randn(5, 3, 8, 8)
    generated_maps = [torch.randn(4, 6, 4, 4)]
    positive_maps = [torch.randn(5, 6, 4, 4)]

    vectors_without = vectorize_feature_maps(
        generated_maps,
        config=FeatureVectorizationConfig(
            include_per_location=True,
            include_global_stats=True,
            include_patch2_stats=False,
            include_patch4_stats=False,
            include_input_x2_mean=False,
        ),
        input_images=generated_images,
    )
    vectors_with = vectorize_feature_maps(
        generated_maps,
        config=FeatureVectorizationConfig(
            include_per_location=True,
            include_global_stats=True,
            include_patch2_stats=False,
            include_patch4_stats=False,
            include_input_x2_mean=True,
        ),
        input_images=generated_images,
    )
    positive_without = vectorize_feature_maps(
        positive_maps,
        config=FeatureVectorizationConfig(
            include_per_location=True,
            include_global_stats=True,
            include_patch2_stats=False,
            include_patch4_stats=False,
            include_input_x2_mean=False,
        ),
        input_images=positive_images,
    )
    positive_with = vectorize_feature_maps(
        positive_maps,
        config=FeatureVectorizationConfig(
            include_per_location=True,
            include_global_stats=True,
            include_patch2_stats=False,
            include_patch4_stats=False,
            include_input_x2_mean=True,
        ),
        input_images=positive_images,
    )

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
            include_global_stats=True,
            include_patch2_stats=False,
            include_patch4_stats=False,
            include_input_x2_mean=False,
        ),
        normalize_features=False,
        normalize_drifts=False,
    )

    loss_without, stats_without = feature_space_drifting_loss(
        generated_feature_vectors=vectors_without,
        positive_feature_vectors=positive_without,
        unconditional_feature_vectors=None,
        base_loss_config=base_config,
        feature_config=feature_config,
    )
    loss_with, stats_with = feature_space_drifting_loss(
        generated_feature_vectors=vectors_with,
        positive_feature_vectors=positive_with,
        unconditional_feature_vectors=None,
        base_loss_config=base_config,
        feature_config=feature_config,
    )

    vector_delta = float(stats_with["vector_count"] - stats_without["vector_count"])
    loss_delta = abs(float(loss_with.item()) - float(loss_without.item()))
    passed = vector_delta > 0.0 and loss_delta > float(args.min_loss_delta)
    payload = {
        "seed": int(args.seed),
        "keys_without_x2": sorted(vectors_without.keys()),
        "keys_with_x2": sorted(vectors_with.keys()),
        "vector_count_without_x2": float(stats_without["vector_count"]),
        "vector_count_with_x2": float(stats_with["vector_count"]),
        "vector_count_delta": vector_delta,
        "loss_without_x2": float(loss_without.item()),
        "loss_with_x2": float(loss_with.item()),
        "loss_delta": loss_delta,
        "min_loss_delta": float(args.min_loss_delta),
        "passed": bool(passed),
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rung-2 x^2 feature toggle effect check.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-loss-delta", type=float, default=1e-9)
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/stage4_x2_toggle/x2_toggle_check.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
