from __future__ import annotations

from pathlib import Path


_STRICT_LONGRUN_CONFIGS = (
    Path("configs/latent/imagenet1k_sdvae_latents_ablation_horizon_b2.yaml"),
    Path("configs/latent/imagenet1k_sdvae_latents_ablation_horizon_b2_step400_w64.yaml"),
    Path("configs/latent/imagenet1k_sdvae_latents_ablation_horizon_b2_cuda1_step400_w64.yaml"),
    Path("configs/latent/imagenet1k_sdvae_latents_ablation_horizon_b2_cuda1_step600_w64.yaml"),
    Path("configs/latent/imagenet1k_sdvae_latents_recovery_from550_step700_w64.yaml"),
    Path("configs/latent/imagenet1k_sdvae_latents_recovery_from550_cuda1_step700_w64.yaml"),
    Path("configs/latent/longrun_feature_queue.yaml"),
    Path("configs/latent/rung4_mini_long_protocol.yaml"),
)


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


def test_longrun_queue_configs_pin_strict_no_replacement() -> None:
    for path in _STRICT_LONGRUN_CONFIGS:
        config = _parse_simple_kv(path)
        assert config.get("use-queue") == "true", f"{path} must set use-queue=true"
        assert (
            config.get("queue-strict-without-replacement") == "true"
        ), f"{path} must set queue-strict-without-replacement=true"


def test_longrun_queue_configs_keep_capacity_compatible_with_strict_mode() -> None:
    for path in _STRICT_LONGRUN_CONFIGS:
        config = _parse_simple_kv(path)
        positives = int(config["positives-per-group"])
        unconditional = int(config["unconditional-per-group"])
        per_class_capacity = int(config["queue-per-class-capacity"])
        global_capacity = int(config["queue-global-capacity"])

        assert (
            per_class_capacity >= positives
        ), f"{path} queue-per-class-capacity must be >= positives-per-group for strict mode"
        assert (
            global_capacity >= unconditional
        ), f"{path} queue-global-capacity must be >= unconditional-per-group for strict mode"
