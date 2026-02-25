from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize("module_name", ["scripts.train_latent", "scripts.train_pixel"])
def test_attach_loss_scale_metrics_adds_per_term_fields(module_name: str) -> None:
    module = importlib.import_module(module_name)
    stats = {"loss": 1557.0, "loss_term_count": 1557.0, "feature_loss": 1557.0}

    module._attach_loss_scale_metrics(stats)

    assert stats["loss_per_term"] == pytest.approx(1.0)
    assert stats["feature_loss_per_term"] == pytest.approx(1.0)


@pytest.mark.parametrize("module_name", ["scripts.train_latent", "scripts.train_pixel"])
def test_attach_loss_scale_metrics_skips_when_term_count_invalid(module_name: str) -> None:
    module = importlib.import_module(module_name)
    stats = {"loss": 1557.0, "loss_term_count": 0.0, "feature_loss": 1557.0}

    module._attach_loss_scale_metrics(stats)

    assert "loss_per_term" not in stats
    assert "feature_loss_per_term" not in stats
