from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from drifting_models.utils import add_device_argument, codebase_fingerprint, environment_fingerprint, environment_snapshot, write_json
from drifting_models.utils.run_md import write_run_md


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    write_json(output_dir / "env_snapshot.json", environment_snapshot(paths=[output_dir]))
    write_json(output_dir / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))
    write_json(output_dir / "env_fingerprint.json", environment_fingerprint())

    variants = _build_variants()
    results: list[dict[str, object]] = []
    for variant in variants:
        result = _run_variant(
            variant=variant,
            output_dir=output_dir,
            device=args.device,
            steps=int(args.steps),
            seed=int(args.seed),
        )
        results.append(result)

    summary = {
        "device": args.device,
        "steps": int(args.steps),
        "seed": int(args.seed),
        "variants": results,
    }
    (output_dir / "generator_arch_ablation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "generator_arch_ablation_summary.md").write_text(
        _to_markdown(summary),
        encoding="utf-8",
    )
    write_run_md(
        output_dir / "RUN.md",
        {
            "output_root": str(output_dir),
            "args": vars(args),
            "paths": {
                "summary_json": str(output_dir / "generator_arch_ablation_summary.json"),
                "summary_markdown": str(output_dir / "generator_arch_ablation_summary.md"),
                "env_snapshot_json": str(output_dir / "env_snapshot.json"),
                "codebase_fingerprint_json": str(output_dir / "codebase_fingerprint.json"),
                "env_fingerprint_json": str(output_dir / "env_fingerprint.json"),
            },
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "variant_count": len(results)}, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run short generator-architecture ablation matrix (RoPE/QK-norm/RMSNorm/style/register tokens)."
    )
    add_device_argument(parser, default="auto")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", type=str, default="outputs/stage2_generator_arch_ablations")
    return parser.parse_args()


def _build_variants() -> list[dict[str, object]]:
    return [
        {
            "name": "baseline_table8_like",
            "description": "RMSNorm + QK-Norm + RoPE + register=16 + style(64x32).",
            "overrides": {},
        },
        {
            "name": "no_rope",
            "description": "Disable RoPE only.",
            "overrides": {"use_rope": False},
        },
        {
            "name": "no_qk_norm",
            "description": "Disable QK-Norm only.",
            "overrides": {"use_qk_norm": False},
        },
        {
            "name": "layernorm",
            "description": "Use LayerNorm instead of RMSNorm.",
            "overrides": {"norm_type": "layernorm"},
        },
        {
            "name": "no_register_tokens",
            "description": "Remove register tokens.",
            "overrides": {"register_tokens": 0},
        },
        {
            "name": "no_style_tokens",
            "description": "Disable style tokens (style_token_count=0).",
            "overrides": {"style_token_count": 0, "style_vocab_size": 1},
        },
    ]


def _base_hparams(*, device: str, steps: int, seed: int) -> dict[str, object]:
    queue_store_device = "cpu"
    if device.startswith("cuda"):
        queue_store_device = "cuda"
    return {
        "device": device,
        "seed": seed,
        "steps": steps,
        "log_every": max(1, steps // 4),
        "groups": 8,
        "negatives_per_group": 4,
        "positives_per_group": 4,
        "unconditional_per_group": 2,
        # Ablation-only profile: keep classes/queue small so we can compare model variants quickly.
        "num_classes": 128,
        "image_size": 32,
        "channels": 4,
        "patch_size": 2,
        "hidden_dim": 384,
        "depth": 8,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "register_tokens": 16,
        "style_vocab_size": 64,
        "style_token_count": 32,
        "norm_type": "rmsnorm",
        "use_qk_norm": True,
        "use_rope": True,
        "temperature": 0.05,
        "learning_rate": 1e-4,
        "precision": "bf16" if device.startswith("cuda") else "fp32",
        "use_feature_loss": False,
        "feature_base_channels": 32,
        "feature_stages": 3,
        "feature_temperatures": [0.02, 0.05, 0.2],
        "feature_selected_stages": [0, 1, 2],
        "include_patch4_stats": True,
        "include_input_x2_mean": True,
        "use_queue": False,
        "queue_prime_samples": 4096,
        "queue_push_batch": 256,
        "queue_per_class_capacity": 64,
        "queue_global_capacity": 4096,
        "queue_store_device": queue_store_device,
        "queue_warmup_mode": "random",
        "queue_warmup_min_per_class": 1,
        "queue_refill_policy": "per_step",
        "queue_refill_every": 1,
        "real_batch_source": "synthetic_dataset",
        "real_dataset_size": 16384,
        "real_loader_batch_size": 256,
    }


def _run_variant(
    *,
    variant: dict[str, object],
    output_dir: Path,
    device: str,
    steps: int,
    seed: int,
) -> dict[str, object]:
    name = str(variant["name"])
    config = _base_hparams(device=device, steps=steps, seed=seed)
    config.update(dict(variant["overrides"]))

    variant_root = output_dir / name
    train_root = variant_root / "train"
    train_root.mkdir(parents=True, exist_ok=True)
    command = _build_command(config=config, output_dir=train_root)

    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(completed.stdout)
    except subprocess.CalledProcessError as exc:
        return {
            "name": name,
            "description": str(variant["description"]),
            "success": False,
            "error": exc.stderr.strip() if exc.stderr else exc.stdout.strip(),
            "command": command,
        }
    except json.JSONDecodeError as exc:
        return {
            "name": name,
            "description": str(variant["description"]),
            "success": False,
            "error": f"Failed to parse train output JSON: {exc}",
            "command": command,
        }

    final = payload["logs"][-1]
    perf = payload.get("perf", {})
    numeric_keys = (
        "loss",
        "mean_drift_norm",
        "grad_norm",
        "step_time_s",
        "generated_images_per_sec",
    )
    finite = all(_is_finite_number(final.get(key)) for key in numeric_keys)

    return {
        "name": name,
        "description": str(variant["description"]),
        "success": True,
        "finite_final_metrics": finite,
        "loss": float(final["loss"]),
        "mean_drift_norm": float(final["mean_drift_norm"]),
        "grad_norm": float(final["grad_norm"]),
        "step_time_s": float(final["step_time_s"]),
        "generated_images_per_sec": float(final["generated_images_per_sec"]),
        "mean_step_time_s": float(perf.get("mean_step_time_s", 0.0)),
        "mean_generated_images_per_sec": float(perf.get("mean_generated_images_per_sec", 0.0)),
        "max_peak_cuda_mem_mb": float(perf.get("max_peak_cuda_mem_mb", 0.0)),
        "queue_underflow_totals": payload.get("queue_underflow_totals"),
        "train_output_dir": str(train_root),
        "command": command,
    }


def _build_command(*, config: dict[str, object], output_dir: Path) -> list[str]:
    command: list[str] = [sys.executable, "scripts/train_latent.py", "--output-dir", str(output_dir)]
    mapping: tuple[tuple[str, str], ...] = (
        ("device", "--device"),
        ("seed", "--seed"),
        ("steps", "--steps"),
        ("log_every", "--log-every"),
        ("groups", "--groups"),
        ("negatives_per_group", "--negatives-per-group"),
        ("positives_per_group", "--positives-per-group"),
        ("unconditional_per_group", "--unconditional-per-group"),
        ("num_classes", "--num-classes"),
        ("image_size", "--image-size"),
        ("channels", "--channels"),
        ("patch_size", "--patch-size"),
        ("hidden_dim", "--hidden-dim"),
        ("depth", "--depth"),
        ("num_heads", "--num-heads"),
        ("mlp_ratio", "--mlp-ratio"),
        ("register_tokens", "--register-tokens"),
        ("style_vocab_size", "--style-vocab-size"),
        ("style_token_count", "--style-token-count"),
        ("norm_type", "--norm-type"),
        ("temperature", "--temperature"),
        ("learning_rate", "--learning-rate"),
        ("precision", "--precision"),
        ("feature_base_channels", "--feature-base-channels"),
        ("feature_stages", "--feature-stages"),
        ("queue_prime_samples", "--queue-prime-samples"),
        ("queue_push_batch", "--queue-push-batch"),
        ("queue_per_class_capacity", "--queue-per-class-capacity"),
        ("queue_global_capacity", "--queue-global-capacity"),
        ("queue_store_device", "--queue-store-device"),
        ("queue_warmup_mode", "--queue-warmup-mode"),
        ("queue_warmup_min_per_class", "--queue-warmup-min-per-class"),
        ("queue_refill_policy", "--queue-refill-policy"),
        ("queue_refill_every", "--queue-refill-every"),
        ("real_batch_source", "--real-batch-source"),
        ("real_dataset_size", "--real-dataset-size"),
        ("real_loader_batch_size", "--real-loader-batch-size"),
    )
    for key, flag in mapping:
        value = config[key]
        command.extend([flag, str(value)])
    if bool(config.get("use_qk_norm", False)):
        command.append("--use-qk-norm")
    if bool(config.get("use_rope", False)):
        command.append("--use-rope")
    if bool(config.get("use_feature_loss", False)):
        command.append("--use-feature-loss")
    if bool(config.get("include_patch4_stats", False)):
        command.append("--include-patch4-stats")
    if bool(config.get("include_input_x2_mean", False)):
        command.append("--include-input-x2-mean")
    if bool(config.get("use_queue", False)):
        command.append("--use-queue")
    feature_temperatures = config.get("feature_temperatures", [])
    if feature_temperatures:
        command.append("--feature-temperatures")
        command.extend([str(value) for value in feature_temperatures])
    feature_selected_stages = config.get("feature_selected_stages", [])
    if feature_selected_stages:
        command.append("--feature-selected-stages")
        command.extend([str(value) for value in feature_selected_stages])
    return command


def _is_finite_number(value: object) -> bool:
    if not isinstance(value, (float, int)):
        return False
    return math.isfinite(float(value))


def _to_markdown(summary: dict[str, object]) -> str:
    rows = [
        "| variant | success | finite | loss | drift | step_time_s | img_per_sec | peak_vram_mb |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in summary["variants"]:
        if not entry.get("success", False):
            rows.append(
                f"| {entry['name']} | 0 | 0 | nan | nan | nan | nan | nan |"
            )
            continue
        rows.append(
            f"| {entry['name']} | 1 | {1 if entry['finite_final_metrics'] else 0} | "
            f"{entry['loss']:.6f} | {entry['mean_drift_norm']:.6f} | "
            f"{entry['mean_step_time_s']:.4f} | {entry['mean_generated_images_per_sec']:.4f} | "
            f"{entry['max_peak_cuda_mem_mb']:.2f} |"
        )
    return "# Generator Architecture Ablation Summary\n\n" + "\n".join(rows) + "\n"


if __name__ == "__main__":
    main()
