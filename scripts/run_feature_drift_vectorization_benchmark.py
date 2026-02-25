from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from drifting_models.drift_field import DriftFieldConfig, build_negative_log_weights
from drifting_models.drift_loss import (
    DriftingLossConfig,
    _compute_weighted_drift_slot_batched,
    compute_weighted_drift,
)
from drifting_models.utils import add_device_argument, resolve_device, seed_everything


def main() -> None:
    args = _parse_args()
    device = resolve_device(args.device)
    seed_everything(args.seed)

    x, y_pos, y_neg, negative_log_weights, generated_negative_count = _build_inputs(args=args, device=device)
    config = DriftingLossConfig(
        drift_field=DriftFieldConfig(
            temperature=float(args.temperature),
            normalize_over_x=not args.disable_normalize_over_x,
            mask_self_negatives=True,
        ),
        attraction_scale=1.0,
        repulsion_scale=1.0,
        stopgrad_target=True,
    )

    def legacy() -> torch.Tensor:
        return _legacy_slot_loop(
            x_vectors=x,
            y_pos_vectors=y_pos,
            y_neg_vectors=y_neg,
            config=config,
            negative_log_weights=negative_log_weights,
            generated_negative_count=generated_negative_count,
        )

    def vectorized() -> torch.Tensor:
        return _compute_weighted_drift_slot_batched(
            x_vectors=x,
            y_pos_vectors=y_pos,
            y_neg_vectors=y_neg,
            config=config,
            negative_log_weights=negative_log_weights,
            generated_negative_count=generated_negative_count,
        )

    legacy_out = legacy()
    vectorized_out = vectorized()
    max_abs_diff = float((legacy_out - vectorized_out).abs().max().item())

    legacy_ms = _time_ms(legacy, iterations=args.iterations, warmup=args.warmup, device=device)
    vectorized_ms = _time_ms(vectorized, iterations=args.iterations, warmup=args.warmup, device=device)
    speedup = legacy_ms / vectorized_ms if vectorized_ms > 0 else float("inf")

    result = {
        "device": str(device),
        "seed": int(args.seed),
        "shape": {
            "generated": [int(x.shape[0]), int(x.shape[1]), int(x.shape[2])],
            "positive": [int(y_pos.shape[0]), int(y_pos.shape[1]), int(y_pos.shape[2])],
            "negative": [int(y_neg.shape[0]), int(y_neg.shape[1]), int(y_neg.shape[2])],
        },
        "normalize_over_x": bool(not args.disable_normalize_over_x),
        "iterations": int(args.iterations),
        "warmup": int(args.warmup),
        "legacy_ms": float(legacy_ms),
        "vectorized_ms": float(vectorized_ms),
        "speedup_x": float(speedup),
        "max_abs_diff": float(max_abs_diff),
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark slot-loop vs vectorized feature drift kernel.")
    add_device_argument(parser, default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--positives", type=int, default=128)
    parser.add_argument("--unconditional", type=int, default=32)
    parser.add_argument("--vectors", type=int, default=16)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--disable-normalize-over-x", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/benchmarks/feature_drift_vectorization/vectorization_benchmark.json",
    )
    return parser.parse_args()


def _build_inputs(
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    x = torch.randn(args.batch, args.vectors, args.channels, device=device, dtype=torch.float32)
    y_pos = torch.randn(args.positives, args.vectors, args.channels, device=device, dtype=torch.float32)
    y_unc = torch.randn(args.unconditional, args.vectors, args.channels, device=device, dtype=torch.float32)
    y_neg = torch.cat([x, y_unc], dim=0)
    negative_log_weights = build_negative_log_weights(
        n_generated_negatives=x.shape[0],
        n_unconditional_negatives=y_unc.shape[0],
        unconditional_weight=2.0,
        device=device,
        dtype=x.dtype,
    )
    return x, y_pos, y_neg, negative_log_weights, int(x.shape[0])


def _legacy_slot_loop(
    *,
    x_vectors: torch.Tensor,
    y_pos_vectors: torch.Tensor,
    y_neg_vectors: torch.Tensor,
    config: DriftingLossConfig,
    negative_log_weights: torch.Tensor | None,
    generated_negative_count: int,
) -> torch.Tensor:
    drift = torch.zeros_like(x_vectors)
    for slot_index in range(x_vectors.shape[1]):
        drift_slot, _ = compute_weighted_drift(
            x=x_vectors[:, slot_index, :],
            y_pos=y_pos_vectors[:, slot_index, :],
            y_neg=y_neg_vectors[:, slot_index, :],
            config=config,
            negative_log_weights=negative_log_weights,
            generated_negative_count=generated_negative_count,
        )
        drift[:, slot_index, :] = drift_slot
    return drift


def _time_ms(fn, *, iterations: int, warmup: int, device: torch.device) -> float:
    for _ in range(max(0, warmup)):
        _ = fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(max(1, iterations)):
        _ = fn()
    _sync(device)
    elapsed = time.perf_counter() - start
    return (elapsed / max(1, iterations)) * 1000.0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
