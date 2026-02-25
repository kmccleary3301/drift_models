from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class RuntimeCapabilities:
    cuda_available: bool
    cuda_device_count: int
    xpu_available: bool
    xpu_device_count: int
    mps_available: bool
    available_backends: tuple[str, ...]
    auto_device: str
    compile_available: bool
    compile_supported_backends: tuple[str, ...]
    compile_unsupported_backends: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "auto_device": self.auto_device,
            "available_backends": list(self.available_backends),
            "cuda_available": bool(self.cuda_available),
            "cuda_device_count": int(self.cuda_device_count),
            "xpu_available": bool(self.xpu_available),
            "xpu_device_count": int(self.xpu_device_count),
            "mps_available": bool(self.mps_available),
            "compile_available": bool(self.compile_available),
            "compile_supported_backends": list(self.compile_supported_backends),
            "compile_unsupported_backends": list(self.compile_unsupported_backends),
            "available_device_strings": list(available_device_strings()),
        }


@dataclass(frozen=True)
class CompileAttemptResult:
    enabled: bool
    status: str
    backend: str
    mode: str
    dynamic: bool
    fullgraph: bool
    fail_action: str
    error: str | None = None
    warning: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "enabled": bool(self.enabled),
            "status": self.status,
            "backend": self.backend,
            "mode": self.mode,
            "dynamic": bool(self.dynamic),
            "fullgraph": bool(self.fullgraph),
            "fail_action": self.fail_action,
        }
        if self.error is not None:
            payload["error"] = self.error
        if self.warning is not None:
            payload["warning"] = self.warning
        return payload


def detect_runtime_capabilities() -> RuntimeCapabilities:
    cuda_available = _is_cuda_available()
    xpu_available = _is_xpu_available()
    mps_available = _is_mps_available()
    cuda_device_count = _cuda_device_count() if cuda_available else 0
    xpu_device_count = _xpu_device_count() if xpu_available else 0
    available_backends = tuple(
        backend
        for backend in (
            "cuda" if cuda_available else None,
            "xpu" if xpu_available else None,
            "mps" if mps_available else None,
            "cpu",
        )
        if backend is not None
    )
    auto_device = (
        "cuda"
        if cuda_available
        else (
            "xpu"
            if xpu_available
            else (
                "mps"
                if mps_available
                else "cpu"
            )
        )
    )
    compile_available = callable(getattr(torch, "compile", None))
    compile_supported_backends = tuple(backend for backend in ("cpu", "cuda") if backend in available_backends)
    compile_unsupported_backends = tuple(
        backend
        for backend in available_backends
        if backend not in set(compile_supported_backends)
    )
    return RuntimeCapabilities(
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        xpu_available=xpu_available,
        xpu_device_count=xpu_device_count,
        mps_available=mps_available,
        available_backends=available_backends,
        auto_device=auto_device,
        compile_available=compile_available,
        compile_supported_backends=compile_supported_backends,
        compile_unsupported_backends=compile_unsupported_backends,
    )


def resolve_device(device_arg: str) -> torch.device:
    requested = str(device_arg).strip().lower()
    if requested in {"", "auto"}:
        return torch.device(detect_runtime_capabilities().auto_device)
    if requested in {"gpu", "auto-gpu", "auto_gpu"}:
        capabilities = detect_runtime_capabilities()
        if capabilities.auto_device == "cpu":
            _raise_no_gpu_error(device_arg=requested)
        return torch.device(capabilities.auto_device)
    try:
        device = torch.device(requested)
    except RuntimeError as error:
        options = ", ".join(available_device_strings())
        raise ValueError(f"Invalid --device '{device_arg}'. Supported values include: {options}") from error
    _validate_device_available(device, raw_arg=device_arg)
    return device


def available_device_strings() -> tuple[str, ...]:
    capabilities = detect_runtime_capabilities()
    values: list[str] = ["auto", "cpu"]
    if capabilities.cuda_available:
        values.append("cuda")
        values.extend(f"cuda:{index}" for index in range(capabilities.cuda_device_count))
    if capabilities.xpu_available:
        values.append("xpu")
        values.extend(f"xpu:{index}" for index in range(capabilities.xpu_device_count))
    if capabilities.mps_available:
        values.append("mps")
    return tuple(values)


def device_argument_help(*, default: str = "auto") -> str:
    capabilities = detect_runtime_capabilities()
    available = ", ".join(available_device_strings())
    backends = ", ".join(capabilities.available_backends)
    compile_backends = ", ".join(capabilities.compile_supported_backends) or "none"
    return (
        f"Runtime device selector (default: {default}). "
        "Use 'auto' for prioritized backend selection (cuda > xpu > mps > cpu), "
        "or pass an explicit backend/index like 'cuda:1'. "
        f"Detected backends: {backends}. "
        f"Accepted values now: {available}. "
        f"Compile-supported backends: {compile_backends}."
    )


def add_device_argument(parser: argparse.ArgumentParser, *, default: str = "auto") -> None:
    parser.add_argument(
        "--device",
        type=str,
        default=default,
        help=device_argument_help(default=default),
    )


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if _is_cuda_available():
        torch.cuda.manual_seed_all(seed)
    if _is_xpu_available():
        xpu = getattr(torch, "xpu", None)
        manual_seed_all = None if xpu is None else getattr(xpu, "manual_seed_all", None)
        if callable(manual_seed_all):
            manual_seed_all(seed)


def normalize_compile_fail_action(fail_action: str, *, allow_disable: bool = True) -> str:
    normalized = str(fail_action).strip().lower()
    allowed = {"warn", "raise"}
    if allow_disable:
        allowed.add("disable")
    if normalized not in allowed:
        values = ", ".join(sorted(allowed))
        raise ValueError(f"Invalid compile fail action '{fail_action}'. Expected one of: {values}")
    return normalized


def compile_supported_for_device(device: torch.device) -> tuple[bool, str | None]:
    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        return False, "torch.compile is unavailable in this runtime"
    if device.type in {"cpu", "cuda"}:
        return True, None
    return False, f"torch.compile is not policy-supported on backend '{device.type}' in this repository"


def maybe_compile_callable(
    fn: Callable[..., object],
    *,
    enabled: bool,
    backend: str,
    mode: str,
    dynamic: bool,
    fullgraph: bool,
    fail_action: str,
    device: torch.device,
    context: str,
) -> tuple[Callable[..., object], CompileAttemptResult]:
    normalized_action = normalize_compile_fail_action(fail_action=fail_action, allow_disable=True)
    base = CompileAttemptResult(
        enabled=False,
        status="disabled",
        backend=str(backend),
        mode=str(mode),
        dynamic=bool(dynamic),
        fullgraph=bool(fullgraph),
        fail_action=normalized_action,
    )
    if not enabled:
        return fn, base
    if normalized_action == "disable":
        return fn, CompileAttemptResult(
            enabled=False,
            status="disabled",
            backend=str(backend),
            mode=str(mode),
            dynamic=bool(dynamic),
            fullgraph=bool(fullgraph),
            fail_action=normalized_action,
            warning=f"{context}: compile disabled by policy",
        )
    supported, reason = compile_supported_for_device(device)
    if not supported:
        message = f"{context}: {reason}"
        if normalized_action == "raise":
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return fn, CompileAttemptResult(
            enabled=False,
            status="fallback_eager",
            backend=str(backend),
            mode=str(mode),
            dynamic=bool(dynamic),
            fullgraph=bool(fullgraph),
            fail_action=normalized_action,
            error=message,
            warning=message,
        )
    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        message = f"{context}: torch.compile is unavailable in this runtime"
        if normalized_action == "raise":
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return fn, CompileAttemptResult(
            enabled=False,
            status="fallback_eager",
            backend=str(backend),
            mode=str(mode),
            dynamic=bool(dynamic),
            fullgraph=bool(fullgraph),
            fail_action=normalized_action,
            error=message,
            warning=message,
        )
    try:
        compiled = compile_fn(
            fn,
            backend=str(backend),
            mode=str(mode),
            dynamic=bool(dynamic),
            fullgraph=bool(fullgraph),
        )
        return compiled, CompileAttemptResult(
            enabled=True,
            status="compiled",
            backend=str(backend),
            mode=str(mode),
            dynamic=bool(dynamic),
            fullgraph=bool(fullgraph),
            fail_action=normalized_action,
        )
    except Exception as error:
        message = f"{context}: {error}"
        if normalized_action == "raise":
            raise RuntimeError(message) from error
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return fn, CompileAttemptResult(
            enabled=False,
            status="fallback_eager",
            backend=str(backend),
            mode=str(mode),
            dynamic=bool(dynamic),
            fullgraph=bool(fullgraph),
            fail_action=normalized_action,
            error=str(error),
            warning=message,
        )


def _validate_device_available(device: torch.device, *, raw_arg: str) -> None:
    device_type = device.type
    if device_type == "cpu":
        return
    if device_type == "cuda":
        if not _is_cuda_available():
            _raise_backend_unavailable(raw_arg=raw_arg, backend="cuda")
        index = 0 if device.index is None else int(device.index)
        count = _cuda_device_count()
        if index < 0 or index >= count:
            raise RuntimeError(f"Requested CUDA device index {index}, but only {count} CUDA device(s) detected.")
        return
    if device_type == "xpu":
        if not _is_xpu_available():
            _raise_backend_unavailable(raw_arg=raw_arg, backend="xpu")
        index = 0 if device.index is None else int(device.index)
        count = _xpu_device_count()
        if index < 0 or index >= count:
            raise RuntimeError(f"Requested XPU device index {index}, but only {count} XPU device(s) detected.")
        return
    if device_type == "mps":
        if not _is_mps_available():
            _raise_backend_unavailable(raw_arg=raw_arg, backend="mps")
        if device.index not in (None, 0):
            raise RuntimeError(f"Requested MPS device index {device.index}, but only a single MPS device is supported.")
        return
    options = ", ".join(available_device_strings())
    raise RuntimeError(f"Unsupported device backend '{device_type}' from --device '{raw_arg}'. Try one of: {options}")


def _raise_backend_unavailable(*, raw_arg: str, backend: str) -> None:
    options = ", ".join(available_device_strings())
    raise RuntimeError(f"Requested --device '{raw_arg}' but backend '{backend}' is unavailable. Available: {options}")


def _raise_no_gpu_error(*, device_arg: str) -> None:
    options = ", ".join(available_device_strings())
    raise RuntimeError(f"Requested --device '{device_arg}' but no GPU backend is available. Available: {options}")


def _is_cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cuda_device_count() -> int:
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _is_xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        return False
    is_available = getattr(xpu, "is_available", None)
    if not callable(is_available):
        return False
    try:
        return bool(is_available())
    except Exception:
        return False


def _xpu_device_count() -> int:
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        return 0
    count_fn = getattr(xpu, "device_count", None)
    if not callable(count_fn):
        return 0
    try:
        return int(count_fn())
    except Exception:
        return 0


def _is_mps_available() -> bool:
    backends = getattr(torch, "backends", None)
    if backends is None:
        return False
    mps = getattr(backends, "mps", None)
    if mps is None:
        return False
    is_built = getattr(mps, "is_built", None)
    is_available = getattr(mps, "is_available", None)
    built_value = True
    if callable(is_built):
        try:
            built_value = bool(is_built())
        except Exception:
            built_value = False
    if not callable(is_available):
        return False
    try:
        return built_value and bool(is_available())
    except Exception:
        return False
