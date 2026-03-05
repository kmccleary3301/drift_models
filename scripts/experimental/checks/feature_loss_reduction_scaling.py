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
    generated = {
        "stage0.loc": torch.randn(3, 2, 6, requires_grad=True),
        "stage1.loc": torch.randn(3, 2, 6, requires_grad=True),
    }
    positive = {
        "stage0.loc": torch.randn(4, 2, 6),
        "stage1.loc": torch.randn(4, 2, 6),
    }
    common = dict(
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
    )
    sum_cfg = FeatureDriftingConfig(**common, loss_term_reduction="sum")
    mean_cfg = FeatureDriftingConfig(**common, loss_term_reduction="mean")

    loss_sum, stats_sum = feature_space_drifting_loss(
        generated_feature_vectors=generated,
        positive_feature_vectors=positive,
        unconditional_feature_vectors=None,
        base_loss_config=base_config,
        feature_config=sum_cfg,
    )
    loss_mean, stats_mean = feature_space_drifting_loss(
        generated_feature_vectors=generated,
        positive_feature_vectors=positive,
        unconditional_feature_vectors=None,
        base_loss_config=base_config,
        feature_config=mean_cfg,
    )

    expected_terms = float(stats_sum["loss_term_count"])
    observed_terms = float((loss_sum / loss_mean).item()) if float(loss_mean.item()) != 0.0 else float("inf")
    ratio_error = abs(observed_terms - expected_terms)
    passed = ratio_error <= float(args.tolerance)
    payload = {
        "seed": int(args.seed),
        "expected_loss_terms": expected_terms,
        "observed_sum_over_mean_ratio": observed_terms,
        "ratio_error": ratio_error,
        "tolerance": float(args.tolerance),
        "passed": bool(passed),
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check sum-vs-mean feature loss term reduction scaling.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/stage4_reduction_scaling/reduction_scaling_check.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
