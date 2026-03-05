from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from drifting_models.drift_field import DriftFieldConfig, build_negative_log_weights
from drifting_models.drift_loss import DriftingLossConfig, _compute_weighted_drifts_slot_batched_multi_temperature
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

    def eager_fn() -> list[tuple[float, torch.Tensor]]:
        return _compute_weighted_drifts_slot_batched_multi_temperature(
            x_vectors=x,
            y_pos_vectors=y_pos,
            y_neg_vectors=y_neg,
            temperatures=temperatures,
            base_config=base_config,
            scale_temperature_by_sqrt_channels=not args.disable_temperature_sqrt_scaling,
            negative_log_weights=negative_log_weights,
            generated_negative_count=generated_negative_count,
        )

    compile_fn = getattr(torch, "compile", None)
    compiled_available = compile_fn is not None
    compiled_error = None
    compiled_first_call_ms = None
    compiled_steady_ms = None
    max_abs_diff = None
    if compiled_available:
        try:
            compiled_kernel = compile_fn(
                _compute_weighted_drifts_slot_batched_multi_temperature,
                backend=str(args.compile_backend),
                mode=str(args.compile_mode),
                dynamic=bool(args.compile_dynamic),
                fullgraph=bool(args.compile_fullgraph),
            )
            start = time.perf_counter()
            compiled_out = compiled_kernel(
                x_vectors=x,
                y_pos_vectors=y_pos,
                y_neg_vectors=y_neg,
                temperatures=temperatures,
                base_config=base_config,
                scale_temperature_by_sqrt_channels=not args.disable_temperature_sqrt_scaling,
                negative_log_weights=negative_log_weights,
                generated_negative_count=generated_negative_count,
            )
            _sync(device)
            compiled_first_call_ms = (time.perf_counter() - start) * 1000.0
            eager_out = eager_fn()
            max_abs_diff = _max_abs_diff(a=eager_out, b=compiled_out)

            def compiled_fn() -> list[tuple[float, torch.Tensor]]:
                return compiled_kernel(
                    x_vectors=x,
                    y_pos_vectors=y_pos,
                    y_neg_vectors=y_neg,
                    temperatures=temperatures,
                    base_config=base_config,
                    scale_temperature_by_sqrt_channels=not args.disable_temperature_sqrt_scaling,
                    negative_log_weights=negative_log_weights,
                    generated_negative_count=generated_negative_count,
                )

            compiled_steady_ms = _time_ms(compiled_fn, iterations=args.iterations, warmup=args.warmup, device=device)
        except Exception as error:
            compiled_error = str(error)
    eager_ms = _time_ms(eager_fn, iterations=args.iterations, warmup=args.warmup, device=device)

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
        "compile_backend": str(args.compile_backend),
        "compile_mode": str(args.compile_mode),
        "compile_dynamic": bool(args.compile_dynamic),
        "compile_fullgraph": bool(args.compile_fullgraph),
        "compiled_available": bool(compiled_available),
        "compiled_error": compiled_error,
        "compiled_first_call_ms": None if compiled_first_call_ms is None else float(compiled_first_call_ms),
        "compiled_steady_ms": None if compiled_steady_ms is None else float(compiled_steady_ms),
        "eager_ms": float(eager_ms),
        "speedup_x": None
        if compiled_steady_ms is None or compiled_steady_ms <= 0.0
        else float(eager_ms / compiled_steady_ms),
        "max_abs_diff": max_abs_diff,
        "iterations": int(args.iterations),
        "warmup": int(args.warmup),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark eager vs compiled multi-temperature drift kernel.")
    add_device_argument(parser, default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--positives", type=int, default=128)
    parser.add_argument("--unconditional", type=int, default=32)
    parser.add_argument("--vectors", type=int, default=16)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.02, 0.05, 0.2])
    parser.add_argument("--disable-normalize-over-x", action="store_true")
    parser.add_argument("--disable-temperature-sqrt-scaling", action="store_true")
    parser.add_argument("--compile-backend", type=str, default="inductor")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead")
    parser.add_argument("--compile-dynamic", action="store_true")
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/benchmarks/feature_drift_kernel_compile/kernel_compile_benchmark.json",
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
    a: list[tuple[float, torch.Tensor]],
    b: list[tuple[float, torch.Tensor]],
) -> float:
    if len(a) != len(b):
        raise ValueError("length mismatch")
    max_diff = 0.0
    for (temp_a, drift_a), (temp_b, drift_b) in zip(a, b):
        if float(temp_a) != float(temp_b):
            raise ValueError("temperature mismatch")
        max_diff = max(max_diff, float((drift_a - drift_b).abs().max().item()))
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
