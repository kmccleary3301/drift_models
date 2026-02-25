from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

from drifting_models.utils import (
    add_device_argument,
    codebase_fingerprint,
    detect_runtime_capabilities,
    environment_fingerprint,
    environment_snapshot,
    maybe_compile_callable,
    normalize_compile_fail_action,
    resolve_device,
    write_json,
)


def main() -> None:
    args = _parse_args()
    summary = _build_summary(args=args)
    if args.output_path is not None:
        output_path = Path(args.output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        repo_root = Path(__file__).resolve().parents[1]
        write_json(output_path.parent / "env_snapshot.json", environment_snapshot(paths=[output_path.parent]))
        write_json(output_path.parent / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))
        write_json(output_path.parent / "env_fingerprint.json", environment_fingerprint())
        summary["paths"] = {
            "output_path": str(output_path),
            "env_snapshot_json": str(output_path.parent / "env_snapshot.json"),
            "codebase_fingerprint_json": str(output_path.parent / "codebase_fingerprint.json"),
            "env_fingerprint_json": str(output_path.parent / "env_fingerprint.json"),
        }
        write_json(output_path, summary)
    print(json.dumps(summary, indent=2))
    if bool(args.strict) and int(summary["fail_count"]) > 0:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runtime preflight diagnostics for device/backend readiness.")
    add_device_argument(parser, default="auto")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any check fails.")
    parser.add_argument("--output-path", type=str, default=None, help="Optional path for JSON summary output.")
    parser.add_argument(
        "--check-all-visible-devices",
        action="store_true",
        help="Also run tensor-smoke checks on each visible accelerator index.",
    )
    parser.add_argument(
        "--check-compile",
        action="store_true",
        help="Run a tiny torch.compile smoke check on the selected device.",
    )
    parser.add_argument(
        "--compile-backend",
        type=str,
        default="inductor",
        help="torch.compile backend used for compile smoke checks.",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        help="torch.compile mode used for compile smoke checks.",
    )
    parser.add_argument(
        "--compile-dynamic",
        action="store_true",
        help="Enable dynamic shape mode for compile smoke checks.",
    )
    parser.add_argument(
        "--compile-fullgraph",
        action="store_true",
        help="Enable fullgraph mode for compile smoke checks.",
    )
    parser.add_argument(
        "--compile-fail-action",
        type=str,
        choices=("warn", "raise", "disable"),
        default="warn",
        help="Behavior when compile smoke is unsupported or fails.",
    )
    parser.add_argument(
        "--check-autocast",
        action="store_true",
        help="Run autocast smoke checks for supported dtypes on the selected device.",
    )
    parser.add_argument(
        "--check-torchvision",
        action="store_true",
        help="Import torchvision and report its version.",
    )
    parser.add_argument(
        "--check-sdvae-stack",
        action="store_true",
        help="Import optional SD-VAE stack modules (diffusers/transformers/accelerate/safetensors/huggingface_hub).",
    )
    return parser.parse_args()


def _build_summary(*, args: argparse.Namespace) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    capabilities = detect_runtime_capabilities()
    selected_device = {"value": None}
    _run_check(
        checks=checks,
        name="capabilities.detect",
        fn=lambda: {
            "details": capabilities.to_dict(),
        },
    )
    _run_check(
        checks=checks,
        name=f"device.resolve({args.device})",
        fn=lambda: _resolve_selected_device(args=args, selected_device=selected_device),
    )
    _run_check(
        checks=checks,
        name="tensor.smoke(cpu)",
        fn=lambda: {"details": _tensor_smoke(torch.device("cpu"))},
    )
    if isinstance(selected_device["value"], torch.device):
        resolved = selected_device["value"]
        if resolved.type != "cpu":
            _run_check(
                checks=checks,
                name=f"tensor.smoke(selected:{resolved})",
                fn=lambda: {"details": _tensor_smoke(resolved)},
            )
        else:
            _run_check(
                checks=checks,
                name="tensor.smoke(selected)",
                fn=lambda: {"status": "skip", "details": {"reason": "selected device is cpu (already checked)"},},
            )
    if args.check_all_visible_devices:
        for device_name in _visible_accelerator_devices(capabilities=capabilities):
            _run_check(
                checks=checks,
                name=f"tensor.smoke(visible:{device_name})",
                fn=lambda device_name=device_name: {"details": _tensor_smoke(torch.device(device_name))},
            )
    if args.check_autocast:
        if isinstance(selected_device["value"], torch.device):
            _run_check(
                checks=checks,
                name=f"autocast.smoke({selected_device['value']})",
                fn=lambda: {"details": _autocast_smoke(selected_device["value"])},
            )
        else:
            _run_check(
                checks=checks,
                name="autocast.smoke(selected)",
                fn=lambda: {"status": "skip", "details": {"reason": "selected device did not resolve"}},
            )
    if args.check_compile:
        if isinstance(selected_device["value"], torch.device):
            _run_check(
                checks=checks,
                name=f"compile.smoke({selected_device['value']})",
                fn=lambda: _compile_smoke(
                    selected_device["value"],
                    compile_backend=str(args.compile_backend),
                    compile_mode=str(args.compile_mode),
                    compile_dynamic=bool(args.compile_dynamic),
                    compile_fullgraph=bool(args.compile_fullgraph),
                    compile_fail_action=normalize_compile_fail_action(
                        str(args.compile_fail_action),
                        allow_disable=True,
                    ),
                ),
            )
        else:
            _run_check(
                checks=checks,
                name="compile.smoke(selected)",
                fn=lambda: {"status": "skip", "details": {"reason": "selected device did not resolve"}},
            )
    if args.check_torchvision:
        _run_check(
            checks=checks,
            name="import.torchvision",
            fn=lambda: {"details": _import_module_version("torchvision")},
        )
    if args.check_sdvae_stack:
        for module_name in ("diffusers", "transformers", "accelerate", "safetensors", "huggingface_hub"):
            _run_check(
                checks=checks,
                name=f"import.{module_name}",
                fn=lambda module_name=module_name: {"details": _import_module_version(module_name)},
            )
    pass_count = sum(1 for item in checks if item["status"] == "pass")
    fail_count = sum(1 for item in checks if item["status"] == "fail")
    skip_count = sum(1 for item in checks if item["status"] == "skip")
    summary = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
        },
        "torch": {
            "version": str(getattr(torch, "__version__", "unknown")),
            "git_version": str(getattr(torch.version, "git_version", None)),
        },
        "args": vars(args),
        "checks": checks,
        "pass_count": int(pass_count),
        "fail_count": int(fail_count),
        "skip_count": int(skip_count),
        "status": "pass" if fail_count == 0 else "fail",
        "strict_would_fail": bool(args.strict and fail_count > 0),
        "env_fingerprint": environment_fingerprint(),
    }
    return summary


def _run_check(
    *,
    checks: list[dict[str, object]],
    name: str,
    fn: Callable[[], dict[str, object]],
) -> None:
    started = time.perf_counter()
    try:
        payload = fn()
        status = str(payload.pop("status", "pass"))
        check = {
            "name": name,
            "status": status,
            "duration_ms": float((time.perf_counter() - started) * 1000.0),
        }
        if payload:
            check.update(payload)
        checks.append(check)
    except Exception as error:
        checks.append(
            {
                "name": name,
                "status": "fail",
                "duration_ms": float((time.perf_counter() - started) * 1000.0),
                "error": str(error),
            }
        )


def _resolve_selected_device(*, args: argparse.Namespace, selected_device: dict[str, object]) -> dict[str, object]:
    device = resolve_device(str(args.device))
    selected_device["value"] = device
    return {"details": {"resolved_device": str(device), "type": device.type}}


def _tensor_smoke(device: torch.device) -> dict[str, object]:
    x = torch.randn(16, 16, device=device, dtype=torch.float32, requires_grad=True)
    y = torch.randn(16, 16, device=device, dtype=torch.float32)
    value = (x @ y).sin().mean()
    value.backward()
    _sync_device(device)
    grad_norm = None if x.grad is None else float(x.grad.norm().item())
    return {"device": str(device), "value": float(value.item()), "grad_norm": grad_norm}


def _autocast_smoke(device: torch.device) -> dict[str, object]:
    if device.type == "cpu":
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            x = torch.randn(8, 8, device=device, dtype=torch.float32)
            y = torch.randn(8, 8, device=device, dtype=torch.float32)
            value = (x @ y).mean()
        return {"device": str(device), "dtype": "bfloat16", "value": float(value.float().item())}
    if device.type == "cuda":
        values: dict[str, float] = {}
        failures: dict[str, str] = {}
        for dtype_name, dtype_value in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
            try:
                with torch.autocast(device_type="cuda", dtype=dtype_value):
                    x = torch.randn(8, 8, device=device, dtype=torch.float32)
                    y = torch.randn(8, 8, device=device, dtype=torch.float32)
                    value = (x @ y).mean()
                values[dtype_name] = float(value.float().item())
            except Exception as error:
                failures[dtype_name] = str(error)
        if not values:
            raise RuntimeError(f"autocast failed for all tested dtypes: {failures}")
        payload: dict[str, object] = {"device": str(device), "values": values}
        if failures:
            payload["failures"] = failures
        return payload
    return {"device": str(device), "status": "skip", "reason": f"autocast smoke not implemented for {device.type}"}


def _compile_smoke(
    device: torch.device,
    *,
    compile_backend: str,
    compile_mode: str,
    compile_dynamic: bool,
    compile_fullgraph: bool,
    compile_fail_action: str,
) -> dict[str, object]:
    def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x @ y).sin()

    x = torch.randn(16, 16, device=device, dtype=torch.float32)
    y = torch.randn(16, 16, device=device, dtype=torch.float32)
    eager = fn(x, y)
    eager_mean = float(eager.mean().item())
    compiled_fn, compile_result = maybe_compile_callable(
        fn,
        enabled=True,
        backend=str(compile_backend),
        mode=str(compile_mode),
        dynamic=bool(compile_dynamic),
        fullgraph=bool(compile_fullgraph),
        fail_action=str(compile_fail_action),
        device=device,
        context=f"compile.smoke({device})",
    )
    if not compile_result.enabled:
        return {
            "status": "skip",
            "details": {
                "device": str(device),
                "eager_mean": eager_mean,
                "reason": compile_result.warning or compile_result.error or "compile disabled",
                "compile": compile_result.to_dict(),
            },
        }
    compiled_out = compiled_fn(x, y)
    _sync_device(device)
    max_abs_diff = float((compiled_out - eager).abs().max().item())
    return {
        "details": {
            "device": str(device),
            "eager_mean": eager_mean,
            "compiled_mean": float(compiled_out.mean().item()),
            "max_abs_diff": max_abs_diff,
            "compile": compile_result.to_dict(),
        }
    }


def _import_module_version(module_name: str) -> dict[str, object]:
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", None)
    return {"module": module_name, "version": None if version is None else str(version)}


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return
    if device.type == "xpu":
        xpu = getattr(torch, "xpu", None)
        synchronize = None if xpu is None else getattr(xpu, "synchronize", None)
        if callable(synchronize):
            synchronize()


def _visible_accelerator_devices(*, capabilities) -> list[str]:
    values: list[str] = []
    if bool(capabilities.cuda_available):
        values.extend(f"cuda:{index}" for index in range(int(capabilities.cuda_device_count)))
    if bool(capabilities.xpu_available):
        values.extend(f"xpu:{index}" for index in range(int(capabilities.xpu_device_count)))
    if bool(capabilities.mps_available):
        values.append("mps")
    return values


if __name__ == "__main__":
    main()
