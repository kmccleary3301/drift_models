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
    _compute_weighted_drifts_slot_batched_multi_temperature,
)
from drifting_models.utils import add_device_argument, resolve_device, seed_everything


def main() -> None:
    args = _parse_args()
    device = resolve_device(args.device)
    seed_everything(args.seed)

    x, y_pos, y_neg, negative_log_weights, generated_negative_count = _build_inputs(args=args, device=device)
    temperatures = tuple(float(value) for value in args.temperatures)
    base_config = DriftingLossConfig(
        drift_field=DriftFieldConfig(
            temperature=0.05,
            normalize_over_x=not args.disable_normalize_over_x,
            mask_self_negatives=True,
        ),
        attraction_scale=1.0,
        repulsion_scale=1.0,
        stopgrad_target=True,
    )

    def loop_impl() -> list[tuple[float, torch.Tensor]]:
        outputs: list[tuple[float, torch.Tensor]] = []
        for temperature in temperatures:
            effective = temperature
            if not args.disable_temperature_sqrt_scaling:
                effective = effective * (x.shape[-1] ** 0.5)
            cfg = DriftingLossConfig(
                drift_field=DriftFieldConfig(
                    temperature=effective,
                    normalize_over_x=base_config.drift_field.normalize_over_x,
                    mask_self_negatives=base_config.drift_field.mask_self_negatives,
                    self_mask_value=base_config.drift_field.self_mask_value,
                    eps=base_config.drift_field.eps,
                ),
                attraction_scale=base_config.attraction_scale,
                repulsion_scale=base_config.repulsion_scale,
                stopgrad_target=base_config.stopgrad_target,
            )
            drift = _compute_weighted_drift_slot_batched(
                x_vectors=x,
                y_pos_vectors=y_pos,
                y_neg_vectors=y_neg,
                config=cfg,
                negative_log_weights=negative_log_weights,
                generated_negative_count=generated_negative_count,
            )
            outputs.append((float(temperature), drift))
        return outputs

    def reused_impl() -> list[tuple[float, torch.Tensor]]:
        return _compute_weighted_drifts_slot_batched_multi_temperature(
            x_vectors=x,
            y_pos_vectors=y_pos,
            y_neg_vectors=y_neg,
            temperatures=temperatures,
            base_config=base_config,
            scale_temperature_by_sqrt_channels=bool(not args.disable_temperature_sqrt_scaling),
            negative_log_weights=negative_log_weights,
            generated_negative_count=generated_negative_count,
        )

    loop_out = loop_impl()
    reuse_out = reused_impl()
    max_abs_diff = _max_abs_diff(loop_out=loop_out, reuse_out=reuse_out)

    loop_ms = _time_ms(loop_impl, iterations=args.iterations, warmup=args.warmup, device=device)
    reuse_ms = _time_ms(reused_impl, iterations=args.iterations, warmup=args.warmup, device=device)

    payload = {
        "device": str(device),
        "seed": int(args.seed),
        "temperatures": [float(value) for value in temperatures],
        "shape": {
            "generated": [int(x.shape[0]), int(x.shape[1]), int(x.shape[2])],
            "positive": [int(y_pos.shape[0]), int(y_pos.shape[1]), int(y_pos.shape[2])],
            "negative": [int(y_neg.shape[0]), int(y_neg.shape[1]), int(y_neg.shape[2])],
        },
        "normalize_over_x": bool(not args.disable_normalize_over_x),
        "scale_temperature_by_sqrt_channels": bool(not args.disable_temperature_sqrt_scaling),
        "iterations": int(args.iterations),
        "warmup": int(args.warmup),
        "legacy_loop_ms": float(loop_ms),
        "distance_reuse_ms": float(reuse_ms),
        "speedup_x": float(loop_ms / reuse_ms if reuse_ms > 0 else float("inf")),
        "max_abs_diff": float(max_abs_diff),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark temperature distance reuse in slot-batched drift.")
    add_device_argument(parser, default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--positives", type=int, default=128)
    parser.add_argument("--unconditional", type=int, default=32)
    parser.add_argument("--vectors", type=int, default=16)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--disable-normalize-over-x", action="store_true")
    parser.add_argument("--disable-temperature-sqrt-scaling", action="store_true")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.02, 0.05, 0.2])
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/benchmarks/feature_drift_temperature_reuse/temperature_reuse_benchmark.json",
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


def _max_abs_diff(
    *,
    loop_out: list[tuple[float, torch.Tensor]],
    reuse_out: list[tuple[float, torch.Tensor]],
) -> float:
    if len(loop_out) != len(reuse_out):
        raise ValueError("outputs length mismatch")
    max_diff = 0.0
    for (temperature_loop, drift_loop), (temperature_reuse, drift_reuse) in zip(loop_out, reuse_out):
        if float(temperature_loop) != float(temperature_reuse):
            raise ValueError("temperature alignment mismatch")
        diff = float((drift_loop - drift_reuse).abs().max().item())
        max_diff = max(max_diff, diff)
    return max_diff


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
