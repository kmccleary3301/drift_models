from __future__ import annotations

import argparse
import importlib

import torch
from torch import nn


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x * self.weight


def _compile_args(*, enabled: bool, fail_action: str = "warn") -> argparse.Namespace:
    return argparse.Namespace(
        compile_generator=enabled,
        compile_backend="inductor",
        compile_mode="reduce-overhead",
        compile_dynamic=False,
        compile_fullgraph=False,
        compile_fail_action=fail_action,
    )


def test_compile_helper_respects_disabled_flag(monkeypatch) -> None:
    for module_name in ("scripts.train_latent", "scripts.train_pixel"):
        module = importlib.import_module(module_name)
        model = _TinyModel()
        info = module._maybe_compile_generator_forward(model=model, args=_compile_args(enabled=False))
        assert info["enabled"] is False
        assert "error" not in info


def test_compile_helper_enables_when_compile_available(monkeypatch) -> None:
    for module_name in ("scripts.train_latent", "scripts.train_pixel"):
        module = importlib.import_module(module_name)
        model = _TinyModel()

        def fake_compile(fn, **kwargs):
            del kwargs
            return fn

        monkeypatch.setattr(module.torch, "compile", fake_compile)
        info = module._maybe_compile_generator_forward(model=model, args=_compile_args(enabled=True))
        assert info["enabled"] is True
        output = model(torch.ones(1))
        assert float(output.item()) == 1.0


def test_compile_helper_warns_when_compile_unavailable(monkeypatch) -> None:
    for module_name in ("scripts.train_latent", "scripts.train_pixel"):
        module = importlib.import_module(module_name)
        model = _TinyModel()
        monkeypatch.setattr(module.torch, "compile", None)
        info = module._maybe_compile_generator_forward(model=model, args=_compile_args(enabled=True, fail_action="warn"))
        assert info["enabled"] is False
        assert "error" in info


def test_compile_helper_disable_action_skips_compile(monkeypatch) -> None:
    for module_name in ("scripts.train_latent", "scripts.train_pixel"):
        module = importlib.import_module(module_name)
        model = _TinyModel()
        called = {"compile": False}

        def fake_compile(fn, **kwargs):
            called["compile"] = True
            return fn

        monkeypatch.setattr(module.torch, "compile", fake_compile)
        info = module._maybe_compile_generator_forward(model=model, args=_compile_args(enabled=True, fail_action="disable"))
        assert info["enabled"] is False
        assert info["status"] == "disabled"
        assert called["compile"] is False
