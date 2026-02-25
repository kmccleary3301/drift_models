from __future__ import annotations

from pathlib import Path


_FAITHFUL_TABLE8_CONFIGS: dict[str, Path] = {
    "ablation_default": Path("configs/latent/imagenet1k_sdvae_latents_table8_ablation_default_template.yaml"),
    "b2": Path("configs/latent/imagenet1k_sdvae_latents_table8_b2_template.yaml"),
    "l2": Path("configs/latent/imagenet1k_sdvae_latents_table8_l2_template.yaml"),
}

_EXPECTED_MAE_BASE_WIDTH: dict[str, str] = {
    "ablation_default": "256",
    "b2": "640",
    "l2": "640",
}

_EXPECTED_MAE_PATH_TOKEN: dict[str, str] = {
    "ablation_default": "w256",
    "b2": "w640",
    "l2": "w640",
}


def _parse_simple_kv(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def test_table8_faithful_configs_enforce_feature_vector_contract() -> None:
    for name, path in _FAITHFUL_TABLE8_CONFIGS.items():
        config = _parse_simple_kv(path)
        assert (
            config.get("mae-encoder-arch") == "paper_resnet34_unet"
        ), f"{path} must set mae-encoder-arch=paper_resnet34_unet"
        assert (
            config.get("feature-base-channels") == _EXPECTED_MAE_BASE_WIDTH[name]
        ), f"{path} must set feature-base-channels={_EXPECTED_MAE_BASE_WIDTH[name]}"
        assert _EXPECTED_MAE_PATH_TOKEN[name] in config.get(
            "mae-encoder-path", ""
        ), f"{path} must point mae-encoder-path to a {_EXPECTED_MAE_PATH_TOKEN[name]} export"
        assert config.get("include-input-x2-mean") == "true", f"{path} must set include-input-x2-mean=true"
        assert config.get("include-patch4-stats") == "true", f"{path} must set include-patch4-stats=true"
        assert (
            config.get("feature-include-raw-drift-loss") == "true"
        ), f"{path} must set feature-include-raw-drift-loss=true"
        assert config.get("feature-raw-drift-loss-weight") in {
            "1",
            "1.0",
        }, f"{path} must set feature-raw-drift-loss-weight=1.0"
        assert (
            config.get("feature-temperature-aggregation") == "sum_drifts_then_mse"
        ), f"{path} must set feature-temperature-aggregation=sum_drifts_then_mse"
        assert config.get("feature-loss-term-reduction") == "sum", f"{path} must set feature-loss-term-reduction=sum"


def test_table8_faithful_configs_enforce_queue_contract() -> None:
    for _, path in _FAITHFUL_TABLE8_CONFIGS.items():
        config = _parse_simple_kv(path)
        assert config.get("queue-push-batch") == "64", f"{path} must set queue-push-batch=64"
        assert config.get("queue-global-capacity") == "1000", f"{path} must set queue-global-capacity=1000"
        assert config.get("queue-warmup-mode") == "class_balanced", f"{path} must set queue-warmup-mode=class_balanced"
        assert (
            config.get("queue-strict-without-replacement") == "true"
        ), f"{path} must set queue-strict-without-replacement=true"
