from __future__ import annotations

import argparse
import os

import pytest
import torch

from drifting_models.utils import runtime
from drifting_models.utils import device as device_module


def _patch_runtime(
    monkeypatch,
    *,
    cuda_available: bool = False,
    cuda_device_count: int = 0,
    xpu_available: bool = False,
    xpu_device_count: int = 0,
    mps_available: bool = False,
) -> None:
    monkeypatch.setattr(device_module, "_is_cuda_available", lambda: cuda_available)
    monkeypatch.setattr(device_module, "_cuda_device_count", lambda: cuda_device_count)
    monkeypatch.setattr(device_module, "_is_xpu_available", lambda: xpu_available)
    monkeypatch.setattr(device_module, "_xpu_device_count", lambda: xpu_device_count)
    monkeypatch.setattr(device_module, "_is_mps_available", lambda: mps_available)


@pytest.mark.parametrize(
    ("cuda_available", "xpu_available", "mps_available", "expected"),
    [
        (True, True, True, "cuda"),
        (False, True, True, "xpu"),
        (False, False, True, "mps"),
        (False, False, False, "cpu"),
    ],
)
def test_resolve_device_auto_prioritizes_available_accelerators(
    monkeypatch,
    cuda_available: bool,
    xpu_available: bool,
    mps_available: bool,
    expected: str,
) -> None:
    _patch_runtime(
        monkeypatch,
        cuda_available=cuda_available,
        cuda_device_count=2 if cuda_available else 0,
        xpu_available=xpu_available,
        xpu_device_count=2 if xpu_available else 0,
        mps_available=mps_available,
    )
    resolved = runtime.resolve_device("auto")
    assert resolved.type == expected


def test_resolve_device_gpu_alias_raises_when_no_gpu(monkeypatch) -> None:
    _patch_runtime(monkeypatch, cuda_available=False, xpu_available=False, mps_available=False)
    with pytest.raises(RuntimeError, match="no GPU backend is available"):
        runtime.resolve_device("gpu")


def test_resolve_device_cuda_index_validation(monkeypatch) -> None:
    _patch_runtime(monkeypatch, cuda_available=True, cuda_device_count=1)
    with pytest.raises(RuntimeError, match="only 1 CUDA device"):
        runtime.resolve_device("cuda:2")


def test_resolve_device_xpu_index_validation(monkeypatch) -> None:
    _patch_runtime(monkeypatch, xpu_available=True, xpu_device_count=1)
    with pytest.raises(RuntimeError, match="only 1 XPU device"):
        runtime.resolve_device("xpu:3")


def test_resolve_device_mps_index_validation(monkeypatch) -> None:
    _patch_runtime(monkeypatch, mps_available=True)
    with pytest.raises(RuntimeError, match="single MPS device"):
        runtime.resolve_device("mps:1")


def test_resolve_device_invalid_string_shows_supported_values(monkeypatch) -> None:
    _patch_runtime(monkeypatch, cuda_available=False, xpu_available=False, mps_available=False)
    with pytest.raises(ValueError, match="Supported values include: auto, cpu"):
        runtime.resolve_device("not-a-device")


def test_available_device_strings_include_detected_indices(monkeypatch) -> None:
    _patch_runtime(monkeypatch, cuda_available=True, cuda_device_count=2, xpu_available=True, xpu_device_count=1)
    options = runtime.available_device_strings()
    assert options == ("auto", "cpu", "cuda", "cuda:0", "cuda:1", "xpu", "xpu:0")


def test_detect_runtime_capabilities_reports_consistent_summary(monkeypatch) -> None:
    _patch_runtime(monkeypatch, cuda_available=True, cuda_device_count=2, mps_available=True)
    capabilities = runtime.detect_runtime_capabilities()
    assert capabilities.cuda_available is True
    assert capabilities.cuda_device_count == 2
    assert capabilities.mps_available is True
    assert capabilities.available_backends == ("cuda", "mps", "cpu")
    assert capabilities.auto_device == "cuda"


def test_seed_everything_sets_hash_seed_and_calls_torch_seed(monkeypatch) -> None:
    calls: list[int] = []
    cuda_calls: list[int] = []
    _patch_runtime(monkeypatch, cuda_available=True, cuda_device_count=1)
    monkeypatch.setattr(device_module.torch, "manual_seed", lambda seed: calls.append(int(seed)))
    monkeypatch.setattr(device_module.torch.cuda, "manual_seed_all", lambda seed: cuda_calls.append(int(seed)))

    runtime.seed_everything(1234)

    assert os.environ["PYTHONHASHSEED"] == "1234"
    assert calls == [1234]
    assert cuda_calls == [1234]


def test_resolve_device_returns_torch_device(monkeypatch) -> None:
    _patch_runtime(monkeypatch, cuda_available=False, xpu_available=False, mps_available=False)
    resolved = runtime.resolve_device("cpu")
    assert isinstance(resolved, torch.device)
    assert str(resolved) == "cpu"


def test_add_device_argument_sets_default_and_help(monkeypatch) -> None:
    _patch_runtime(monkeypatch, cuda_available=True, cuda_device_count=2, xpu_available=False, mps_available=False)
    parser = argparse.ArgumentParser(prog="device_test", add_help=False)
    runtime.add_device_argument(parser, default="cpu")

    action = next(item for item in parser._actions if item.dest == "device")
    assert action.default == "cpu"
    assert isinstance(action.help, str)
    assert "cuda > xpu > mps > cpu" in action.help
    assert "Detected backends: cuda, cpu" in action.help


def test_add_device_argument_parses_explicit_value(monkeypatch) -> None:
    _patch_runtime(monkeypatch, cuda_available=False, xpu_available=False, mps_available=False)
    parser = argparse.ArgumentParser(prog="device_parse_test", add_help=False)
    runtime.add_device_argument(parser, default="auto")
    args = parser.parse_args(["--device", "cpu"])
    assert args.device == "cpu"


def test_normalize_compile_fail_action_accepts_disable() -> None:
    assert runtime.normalize_compile_fail_action("disable", allow_disable=True) == "disable"
    with pytest.raises(ValueError):
        runtime.normalize_compile_fail_action("disable", allow_disable=False)
