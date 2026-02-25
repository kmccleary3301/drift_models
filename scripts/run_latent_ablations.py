from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
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
    results = []
    commands: dict[str, dict[str, object]] = {}
    for variant in variants:
        payload, argv = _run_variant(variant=variant, device=args.device)
        final = payload["logs"][-1]
        results.append(
            {
                "name": variant["name"],
                "description": variant["description"],
                "loss": float(final["loss"]),
                "mean_drift_norm": float(final["mean_drift_norm"]),
                "mean_step_time_s": float(payload["perf"]["mean_step_time_s"]),
                "generated_images_per_sec": float(payload["perf"]["mean_generated_images_per_sec"]),
                "raw_summary": payload,
            }
        )
        commands[f"train_{variant['name']}"] = {"argv": argv, "returncode": 0}
    summary = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "device": args.device,
        "results": results,
    }
    summary_json = output_dir / "latent_ablation_summary.json"
    summary_md = output_dir / "latent_ablation_summary.md"
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
    parser = argparse.ArgumentParser(description="Run short latent ablation block")
    add_device_argument(parser, default="auto")
    parser.add_argument("--output-dir", type=str, default="outputs/feature_ablations/latent")
    return parser.parse_args()


def _run_variant(*, variant: dict[str, object], device: str) -> tuple[dict[str, object], list[str]]:
    queue_store_device = "cpu"
    if str(device).startswith("cuda") or str(device).strip().lower() in {"auto", "cuda"}:
        queue_store_device = "cuda"
    base = [
        sys.executable,
        "scripts/train_latent.py",
        "--device",
        device,
        "--steps",
        "4",
        "--log-every",
        "1",
        "--groups",
        "2",
        "--negatives-per-group",
        "2",
        "--positives-per-group",
        "2",
        "--unconditional-per-group",
        "2",
        "--num-classes",
        "128",
        "--image-size",
        "16",
        "--channels",
        "4",
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
        "--precision",
        "bf16" if device == "cuda" else "fp32",
        "--grad-accum-steps",
        "1",
        "--scheduler",
        "none",
        "--use-feature-loss",
        "--feature-base-channels",
        "8",
        "--feature-stages",
        "2",
        "--feature-selected-stages",
        "0",
        "1",
        "--include-input-x2-mean",
        "--use-queue",
        "--queue-prime-samples",
        "512",
        "--queue-push-batch",
        "64",
        "--queue-per-class-capacity",
        "64",
        "--queue-global-capacity",
        "1024",
        "--queue-store-device",
        queue_store_device,
        "--queue-warmup-mode",
        "random",
        "--queue-warmup-min-per-class",
        "1",
        "--real-batch-source",
        "synthetic_dataset",
        "--real-dataset-size",
        "5000",
        "--real-loader-batch-size",
        "128",
    ]
    command = base + list(variant["extra_args"])
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout), command


def _build_variants() -> list[dict[str, object]]:
    return [
        {
            "name": "temp_multi",
            "description": "Temperatures {0.02,0.05,0.2}",
            "extra_args": ["--feature-temperatures", "0.02", "0.05", "0.2"],
        },
        {
            "name": "temp_0_02",
            "description": "Single temperature 0.02",
            "extra_args": ["--feature-temperatures", "0.02"],
        },
        {
            "name": "temp_0_05",
            "description": "Single temperature 0.05",
            "extra_args": ["--feature-temperatures", "0.05"],
        },
        {
            "name": "temp_0_2",
            "description": "Single temperature 0.2",
            "extra_args": ["--feature-temperatures", "0.2"],
        },
        {
            "name": "shared_norm_on",
            "description": "Shared location normalization enabled",
            "extra_args": ["--feature-temperatures", "0.02", "0.05", "0.2"],
        },
        {
            "name": "shared_norm_off",
            "description": "Per-location normalization",
            "extra_args": [
                "--feature-temperatures",
                "0.02",
                "0.05",
                "0.2",
                "--disable-shared-location-normalization",
            ],
        },
        {
            "name": "feature_ab",
            "description": "Feature set (a,b)",
            "extra_args": [
                "--feature-temperatures",
                "0.02",
                "0.05",
                "0.2",
                "--disable-patch2-stats",
            ],
        },
        {
            "name": "feature_ac",
            "description": "Feature set (a-c)",
            "extra_args": [
                "--feature-temperatures",
                "0.02",
                "0.05",
                "0.2",
            ],
        },
        {
            "name": "feature_ad",
            "description": "Feature set (a-d)",
            "extra_args": [
                "--feature-temperatures",
                "0.02",
                "0.05",
                "0.2",
                "--include-patch4-stats",
            ],
        },
    ]


def _to_markdown(summary: dict[str, object]) -> str:
    rows = [
        "| variant | description | loss | mean_drift_norm | mean_step_time_s | images_per_sec |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for entry in summary["results"]:
        rows.append(
            f"| {entry['name']} | {entry['description']} | "
            f"{entry['loss']:.6f} | {entry['mean_drift_norm']:.6f} | "
            f"{entry['mean_step_time_s']:.4f} | {entry['generated_images_per_sec']:.4f} |"
        )
    return "# Latent Ablation Summary\n\n" + "\n".join(rows) + "\n"


if __name__ == "__main__":
    main()
