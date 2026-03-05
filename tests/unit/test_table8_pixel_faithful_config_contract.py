from __future__ import annotations

from pathlib import Path

from drifting_models.utils.simple_kv import load_simple_kv_config


_PIXEL_TABLE8_CONFIGS: dict[str, Path] = {
    "b16": Path("configs/pixel/imagenet256_table8_b16_template.yaml"),
    "l16": Path("configs/pixel/imagenet256_table8_l16_template.yaml"),
}

_EXPECTED_GENERATOR: dict[str, dict[str, str]] = {
    "b16": {
        "hidden-dim": "768",
        "depth": "12",
        "num-heads": "12",
        "learning-rate": "2e-4",
    },
    "l16": {
        "hidden-dim": "1024",
        "depth": "24",
        "num-heads": "16",
        "learning-rate": "4e-4",
    },
}


def _parse_simple_kv(path: Path) -> dict[str, str]:
    return load_simple_kv_config(path)


def test_table8_pixel_templates_pin_generator_and_cfg_contract() -> None:
    for name, path in _PIXEL_TABLE8_CONFIGS.items():
        config = _parse_simple_kv(path)
        assert config.get("image-size") == "256", f"{path} must set image-size=256"
        assert config.get("channels") == "3", f"{path} must set channels=3"
        assert config.get("patch-size") == "16", f"{path} must set patch-size=16"
        assert config.get("register-tokens") == "16", f"{path} must set register-tokens=16"
        assert config.get("norm-type") == "rmsnorm", f"{path} must set norm-type=rmsnorm"
        assert config.get("use-qk-norm") == "true", f"{path} must set use-qk-norm=true"
        assert config.get("use-rope") == "true", f"{path} must set use-rope=true"
        assert config.get("alpha-embedding-type") == "mlp", f"{path} must set alpha-embedding-type=mlp"
        assert config.get("qk-norm-mode") == "l2", f"{path} must set qk-norm-mode=l2"
        assert config.get("rope-mode") == "2d_axial", f"{path} must set rope-mode=2d_axial"
        assert (
            config.get("disable-patch-positional-embedding") == "true"
        ), f"{path} must set disable-patch-positional-embedding=true"
        assert config.get("disable-rmsnorm-affine") == "true", f"{path} must set disable-rmsnorm-affine=true"
        assert config.get("alpha-min") == "1.0", f"{path} must set alpha-min=1.0"
        assert config.get("alpha-max") == "4.0", f"{path} must set alpha-max=4.0"
        assert config.get("alpha-dist") == "powerlaw", f"{path} must set alpha-dist=powerlaw"
        assert config.get("alpha-power") == "5.0", f"{path} must set alpha-power=5.0"
        assert config.get("unconditional-per-group") == "32", f"{path} must set unconditional-per-group=32"
        assert config.get("groups") == "128", f"{path} must set groups=128"
        assert config.get("negatives-per-group") == "64", f"{path} must set negatives-per-group=64"
        assert config.get("positives-per-group") == "128", f"{path} must set positives-per-group=128"
        assert config.get("weight-decay") == "0.01", f"{path} must set weight-decay=0.01"
        assert config.get("warmup-steps") == "10000", f"{path} must set warmup-steps=10000"
        assert config.get("scheduler") == "warmup_cosine", f"{path} must set scheduler=warmup_cosine"
        assert config.get("ema-decay") == "0.9995", f"{path} must set ema-decay=0.9995"
        expected = _EXPECTED_GENERATOR[name]
        for key, value in expected.items():
            assert config.get(key) == value, f"{path} must set {key}={value}"


def test_table8_pixel_templates_pin_paper_closer_feature_encoder_contract() -> None:
    for _, path in _PIXEL_TABLE8_CONFIGS.items():
        config = _parse_simple_kv(path)
        assert config.get("use-feature-loss") == "true", f"{path} must set use-feature-loss=true"
        assert config.get("feature-encoder") == "mae_convnextv2", f"{path} must set feature-encoder=mae_convnextv2"
        assert (
            config.get("mae-encoder-arch") == "paper_resnet34_unet"
        ), f"{path} must set mae-encoder-arch=paper_resnet34_unet"
        assert config.get("mae-input-patchify-size") == "8", f"{path} must set mae-input-patchify-size=8"
        assert config.get("convnextv2-weights") == "none", f"{path} must set convnextv2-weights=none"
        assert config.get("feature-base-channels") == "640", f"{path} must set feature-base-channels=640"
        assert config.get("feature-stages") == "4", f"{path} must set feature-stages=4"
        assert (
            config.get("feature-temperature-aggregation") == "sum_drifts_then_mse"
        ), f"{path} must set feature-temperature-aggregation=sum_drifts_then_mse"
        assert config.get("feature-loss-term-reduction") == "sum", f"{path} must set feature-loss-term-reduction=sum"
        assert config.get("include-input-x2-mean") == "true", f"{path} must set include-input-x2-mean=true"
        assert config.get("include-patch4-stats") == "true", f"{path} must set include-patch4-stats=true"
        assert (
            config.get("feature-include-raw-drift-loss") == "true"
        ), f"{path} must set feature-include-raw-drift-loss=true"
        assert config.get("feature-raw-drift-loss-weight") in {"1", "1.0"}, (
            f"{path} must set feature-raw-drift-loss-weight=1.0"
        )
        assert config.get("queue-strict-without-replacement") == "true", (
            f"{path} must set queue-strict-without-replacement=true"
        )
