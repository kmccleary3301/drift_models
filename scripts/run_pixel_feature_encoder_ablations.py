from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from drifting_models.utils import add_device_argument, codebase_fingerprint, environment_fingerprint, environment_snapshot, write_json
from drifting_models.utils.run_md import write_run_md


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "env_snapshot.json", environment_snapshot(paths=[output_dir]))
    write_json(output_dir / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))
    write_json(output_dir / "env_fingerprint.json", environment_fingerprint())

    variants = _build_variants()
    results: list[dict[str, object]] = []
    commands: dict[str, dict[str, object]] = {}
    for variant in variants:
        payload, argv = _run_variant(device=args.device, extra_args=list(variant["extra_args"]))
        final = payload["logs"][-1]
        results.append(
            {
                "name": variant["name"],
                "description": variant["description"],
                "feature_encoder": payload["train_config"]["feature_encoder"],
                "loss": float(final["loss"]),
                "mean_drift_norm": float(final["mean_drift_norm"]),
                "mean_step_time_s": float(payload["perf"]["mean_step_time_s"]),
                "generated_images_per_sec": float(payload["perf"]["mean_generated_images_per_sec"]),
                "raw_summary": payload,
            }
        )
        commands[f"train_{variant['name']}"] = {"argv": argv, "returncode": 0}

    summary = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "device": args.device,
        "results": results,
    }
    summary_json = output_dir / "pixel_feature_encoder_ablation_summary.json"
    summary_md = output_dir / "pixel_feature_encoder_ablation_summary.md"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(_to_markdown(summary), encoding="utf-8")
    write_run_md(
        output_dir / "RUN.md",
        {
            "output_root": str(output_dir),
            "args": vars(args),
            "paths": {
                "summary_json": str(summary_json),
                "summary_markdown": str(summary_md),
                "env_snapshot_json": str(output_dir / "env_snapshot.json"),
                "codebase_fingerprint_json": str(output_dir / "codebase_fingerprint.json"),
                "env_fingerprint_json": str(output_dir / "env_fingerprint.json"),
            },
            "commands": commands,
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "variant_count": len(results)}, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run short pixel feature-encoder ablation block")
    add_device_argument(parser, default="cpu")
    parser.add_argument("--output-dir", type=str, default="outputs/feature_ablations/pixel_feature_encoders")
    return parser.parse_args()


def _build_variants() -> list[dict[str, object]]:
    return [
        {
            "name": "tiny",
            "description": "Tiny feature encoder baseline",
            "extra_args": ["--feature-encoder", "tiny"],
        },
        {
            "name": "mae",
            "description": "MAE feature encoder path (random init unless export is loaded)",
            "extra_args": [
                "--feature-encoder",
                "mae",
                "--mae-encoder-arch",
                "paper_resnet34_unet",
                "--feature-stages",
                "4",
            ],
        },
        {
            "name": "convnext_tiny",
            "description": "ConvNeXt Tiny feature encoder path",
            "extra_args": ["--feature-encoder", "convnext_tiny", "--convnext-weights", "none"],
        },
    ]


def _run_variant(*, device: str, extra_args: list[str]) -> tuple[dict[str, object], list[str]]:
    argv = [
        sys.executable,
        "scripts/train_pixel.py",
        "--device",
        device,
        "--steps",
        "2",
        "--log-every",
        "1",
        "--groups",
        "1",
        "--negatives-per-group",
        "1",
        "--positives-per-group",
        "1",
        "--num-classes",
        "1000",
        "--image-size",
        "32",
        "--channels",
        "3",
        "--patch-size",
        "4",
        "--hidden-dim",
        "64",
        "--depth",
        "2",
        "--num-heads",
        "4",
        "--register-tokens",
        "8",
        "--temperature",
        "0.05",
        "--learning-rate",
        "1e-4",
        "--scheduler",
        "none",
        "--use-feature-loss",
    ]
    argv.extend(extra_args)
    result = subprocess.run(argv, check=True, capture_output=True, text=True)
    return json.loads(result.stdout), argv


def _to_markdown(summary: dict[str, object]) -> str:
    rows = [
        "| variant | description | feature_encoder | loss | mean_drift_norm | mean_step_time_s | images_per_sec |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for entry in summary["results"]:
        rows.append(
            f"| {entry['name']} | {entry['description']} | {entry['feature_encoder']} | "
            f"{entry['loss']:.6f} | {entry['mean_drift_norm']:.6f} | "
            f"{entry['mean_step_time_s']:.4f} | {entry['generated_images_per_sec']:.4f} |"
        )
    return "# Pixel Feature-Encoder Ablation Summary\n\n" + "\n".join(rows) + "\n"


if __name__ == "__main__":
    main()
