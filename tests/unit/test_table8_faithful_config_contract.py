from __future__ import annotations

from pathlib import Path

from drifting_models.utils.simple_kv import load_simple_kv_config


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

_CLOSEST_FEASIBLE_B2_CONFIG = Path("configs/latent/imagenet1k_sdvae_latents_table8_b2_closest_feasible_single_gpu.yaml")

_EXPECTED_MICROVARIANT_TUPLE: dict[str, str] = {
    "alpha-embedding-type": "mlp",
    "qk-norm-mode": "l2",
    "rope-mode": "2d_axial",
    "disable-patch-positional-embedding": "true",
    "disable-rmsnorm-affine": "true",
}


def _parse_simple_kv(path: Path) -> dict[str, str]:
    return load_simple_kv_config(path)


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
        assert config.get("alpha-embedding-type") == "mlp", f"{path} must set alpha-embedding-type=mlp"
        assert config.get("qk-norm-mode") == "l2", f"{path} must set qk-norm-mode=l2"
        assert config.get("rope-mode") == "2d_axial", f"{path} must set rope-mode=2d_axial"
        assert (
            config.get("disable-patch-positional-embedding") == "true"
        ), f"{path} must set disable-patch-positional-embedding=true"
        assert config.get("disable-rmsnorm-affine") == "true", f"{path} must set disable-rmsnorm-affine=true"


def test_table8_faithful_configs_enforce_queue_contract() -> None:
    for _, path in _FAITHFUL_TABLE8_CONFIGS.items():
        config = _parse_simple_kv(path)
        assert config.get("queue-push-batch") == "64", f"{path} must set queue-push-batch=64"
        assert config.get("queue-global-capacity") == "1000", f"{path} must set queue-global-capacity=1000"
        assert config.get("queue-warmup-mode") == "class_balanced", f"{path} must set queue-warmup-mode=class_balanced"
        assert (
            config.get("queue-strict-without-replacement") == "true"
        ), f"{path} must set queue-strict-without-replacement=true"


def test_closest_feasible_b2_pins_generator_microvariant_tuple() -> None:
    config = _parse_simple_kv(_CLOSEST_FEASIBLE_B2_CONFIG)
    for key, expected in _EXPECTED_MICROVARIANT_TUPLE.items():
        assert config.get(key) == expected, f"{_CLOSEST_FEASIBLE_B2_CONFIG} must set {key}={expected}"
