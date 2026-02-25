from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from drifting_models.utils import add_device_argument, codebase_fingerprint, environment_fingerprint, environment_snapshot, write_json
from drifting_models.utils.run_md import write_run_md


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    write_json(output_root / "env_snapshot.json", environment_snapshot(paths=[output_root]))
    write_json(output_root / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))
    write_json(output_root / "env_fingerprint.json", environment_fingerprint())

    config_paths = [Path(path).expanduser().resolve() for path in args.configs]
    benchmark_rows: list[dict[str, object]] = []
    for config_path in config_paths:
        run_dir = output_root / config_path.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        merged_config_path = _write_merged_benchmark_config(args=args, config_path=config_path, run_dir=run_dir)
        argv = _build_train_argv(args=args, merged_config_path=merged_config_path, run_dir=run_dir)
        env = os.environ.copy()
        if str(args.device).strip().lower() == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(
            argv,
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        row = {
            "config_path": str(config_path),
            "run_dir": str(run_dir),
            "argv": argv,
            "returncode": int(result.returncode),
            "stderr_tail": result.stderr[-2000:],
        }
        if result.returncode == 0:
            payload = json.loads(result.stdout)
            perf = payload.get("perf", {})
            row["perf"] = {
                "mean_step_time_s": perf.get("mean_step_time_s"),
                "mean_generated_images_per_sec": perf.get("mean_generated_images_per_sec"),
                "max_peak_cuda_mem_mb": perf.get("max_peak_cuda_mem_mb"),
            }
            row["queue"] = {
                "mode": payload.get("queue_mode"),
                "underflow_missing_labels": payload.get("queue_underflow_missing_labels"),
                "underflow_backfilled_samples": payload.get("queue_underflow_backfilled_samples"),
                "manifest_fingerprint": payload.get("real_batch_manifest_fingerprint"),
            }
        benchmark_rows.append(row)

    summary = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "device": args.device,
        "bench_steps": int(args.bench_steps),
        "proxy_batch": {
            "groups": int(args.proxy_groups),
            "negatives_per_group": int(args.proxy_negatives_per_group),
            "positives_per_group": int(args.proxy_positives_per_group),
            "unconditional_per_group": int(args.proxy_unconditional_per_group),
        },
        "results": benchmark_rows,
    }
    summary_path = output_root / "table8_latent_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown_path = output_root / "table8_latent_benchmark_summary.md"
    markdown_path.write_text(_to_markdown(summary), encoding="utf-8")
    write_run_md(
        output_root / "RUN.md",
        {
            "output_root": str(output_root),
            "args": vars(args),
            "paths": {
                "summary_json": str(summary_path),
                "summary_markdown": str(markdown_path),
                "env_snapshot_json": str(output_root / "env_snapshot.json"),
                "codebase_fingerprint_json": str(output_root / "codebase_fingerprint.json"),
                "env_fingerprint_json": str(output_root / "env_fingerprint.json"),
            },
            "commands": {
                f"benchmark_{Path(row['config_path']).stem}": {"argv": row.get("argv"), "returncode": row.get("returncode")}
                for row in benchmark_rows
                if isinstance(row, dict) and row.get("argv") is not None
            },
        },
    )
    print(json.dumps({"summary_path": str(summary_path), "markdown_path": str(markdown_path)}, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Table-8 latent configs with proxy batch sizing.")
    add_device_argument(parser, default="auto")
    parser.add_argument("--output-root", type=str, default="outputs/imagenet/benchmarks/table8_latent_proxy")
    parser.add_argument("--bench-steps", type=int, default=6)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/latent/imagenet1k_sdvae_latents_table8_b2_template.yaml",
            "configs/latent/imagenet1k_sdvae_latents_table8_l2_template.yaml",
        ],
    )
    parser.add_argument("--proxy-groups", type=int, default=4)
    parser.add_argument("--proxy-negatives-per-group", type=int, default=4)
    parser.add_argument("--proxy-positives-per-group", type=int, default=4)
    parser.add_argument("--proxy-unconditional-per-group", type=int, default=2)
    parser.add_argument("--proxy-queue-prime-samples", type=int, default=2000)
    parser.add_argument("--proxy-real-loader-batch-size", type=int, default=256)
    return parser.parse_args()


def _write_merged_benchmark_config(*, args: argparse.Namespace, config_path: Path, run_dir: Path) -> Path:
    merged = run_dir / "benchmark_config.yaml"
    base_text = config_path.read_text(encoding="utf-8").rstrip()
    overrides = "\n".join(
        [
            "",
            "# Benchmark overrides appended by scripts/run_latent_table8_benchmark.py",
            f"device: {args.device}",
            f"steps: {int(args.bench_steps)}",
            "log-every: 1",
            "save-every: 0",
            f"groups: {int(args.proxy_groups)}",
            f"negatives-per-group: {int(args.proxy_negatives_per_group)}",
            f"positives-per-group: {int(args.proxy_positives_per_group)}",
            f"unconditional-per-group: {int(args.proxy_unconditional_per_group)}",
            f"queue-prime-samples: {int(args.proxy_queue_prime_samples)}",
            f"real-loader-batch-size: {int(args.proxy_real_loader_batch_size)}",
        ]
    )
    merged.write_text(base_text + "\n" + overrides + "\n", encoding="utf-8")
    return merged


def _build_train_argv(*, args: argparse.Namespace, merged_config_path: Path, run_dir: Path) -> list[str]:
    checkpoint_path = run_dir / "checkpoint.pt"
    return [
        sys.executable,
        "scripts/train_latent.py",
        "--config",
        str(merged_config_path),
        "--checkpoint-path",
        str(checkpoint_path),
        "--output-dir",
        str(run_dir),
    ]


def _to_markdown(summary: dict[str, object]) -> str:
    rows = [
        "# Table-8 Latent Proxy Benchmark",
        "",
        f"- Device: `{summary['device']}`",
        f"- Steps per run: `{summary['bench_steps']}`",
        (
            "- Proxy batch: "
            f"`groups={summary['proxy_batch']['groups']}`, "
            f"`neg={summary['proxy_batch']['negatives_per_group']}`, "
            f"`pos={summary['proxy_batch']['positives_per_group']}`, "
            f"`unc={summary['proxy_batch']['unconditional_per_group']}`"
        ),
        "",
        "| config | returncode | step_time_s | img_per_sec | peak_cuda_mem_mb | queue_underflow |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in summary["results"]:
        perf = entry.get("perf") or {}
        queue = entry.get("queue") or {}
        rows.append(
            f"| `{Path(entry['config_path']).name}` | {entry['returncode']} | "
            f"{_fmt(perf.get('mean_step_time_s'))} | {_fmt(perf.get('mean_generated_images_per_sec'))} | "
            f"{_fmt(perf.get('max_peak_cuda_mem_mb'))} | {_fmt(queue.get('underflow_missing_labels'))} |"
        )
    rows.append("")
    return "\n".join(rows)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    if value is None:
        return "n/a"
    return str(value)


if __name__ == "__main__":
    main()
