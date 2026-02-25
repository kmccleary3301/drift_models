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

    run_dir = output_root / "latent_queue_hotpath"
    run_dir.mkdir(parents=True, exist_ok=True)

    argv = _build_latent_train_argv(args=args, run_dir=run_dir)
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

    row: dict[str, object] = {
        "argv": argv,
        "returncode": int(result.returncode),
        "stderr_tail": result.stderr[-4000:],
        "run_dir": str(run_dir),
    }
    if result.returncode == 0 and result.stdout.strip():
        payload = json.loads(result.stdout)
        row["perf"] = payload.get("perf")
        row["queue_warmup_report"] = payload.get("queue_warmup_report")
        row["queue_underflow_totals"] = payload.get("queue_underflow_totals")
        row["real_batch_manifest_fingerprint"] = payload.get("real_batch_manifest_fingerprint")

    summary = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "kind": "queue_hotpath_benchmark",
        "device": args.device,
        "steps": int(args.steps),
        "results": [row],
    }
    summary_path = output_root / "queue_hotpath_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_path = output_root / "queue_hotpath_benchmark_summary.md"
    md_path.write_text(_to_markdown(summary), encoding="utf-8")
    write_run_md(
        output_root / "RUN.md",
        {
            "output_root": str(output_root),
            "args": vars(args),
            "paths": {
                "summary_json": str(summary_path),
                "summary_markdown": str(md_path),
                "run_dir": str(run_dir),
                "env_snapshot_json": str(output_root / "env_snapshot.json"),
                "codebase_fingerprint_json": str(output_root / "codebase_fingerprint.json"),
                "env_fingerprint_json": str(output_root / "env_fingerprint.json"),
            },
            "commands": {
                "train_latent": {
                    "argv": argv,
                    "returncode": int(result.returncode),
                }
            },
        },
    )
    print(json.dumps({"summary_path": str(summary_path), "markdown_path": str(md_path)}, indent=2))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark latent queue hot path (synthetic dataset + queue enabled).")
    add_device_argument(p, default="auto")
    p.add_argument("--output-root", type=str, default="outputs/benchmarks/queue_hotpath")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--groups", type=int, default=4)
    p.add_argument("--negatives-per-group", type=int, default=4)
    p.add_argument("--positives-per-group", type=int, default=4)
    p.add_argument("--unconditional-per-group", type=int, default=2)
    p.add_argument("--num-classes", type=int, default=128)
    p.add_argument("--queue-prime-samples", type=int, default=2048)
    p.add_argument("--queue-per-class-capacity", type=int, default=64)
    p.add_argument("--queue-global-capacity", type=int, default=4096)
    p.add_argument("--queue-push-batch", type=int, default=256)
    p.add_argument("--real-dataset-size", type=int, default=16384)
    p.add_argument("--real-loader-batch-size", type=int, default=256)
    return p.parse_args()


def _build_latent_train_argv(*, args: argparse.Namespace, run_dir: Path) -> list[str]:
    checkpoint_path = run_dir / "checkpoint.pt"
    return [
        sys.executable,
        "scripts/train_latent.py",
        "--device",
        str(args.device),
        "--seed",
        "1337",
        "--steps",
        str(int(args.steps)),
        "--log-every",
        "1",
        "--save-every",
        "0",
        "--groups",
        str(int(args.groups)),
        "--negatives-per-group",
        str(int(args.negatives_per_group)),
        "--positives-per-group",
        str(int(args.positives_per_group)),
        "--unconditional-per-group",
        str(int(args.unconditional_per_group)),
        "--alpha-fixed",
        "3.0",
        "--num-classes",
        str(int(args.num_classes)),
        "--image-size",
        "16",
        "--channels",
        "4",
        "--patch-size",
        "4",
        "--hidden-dim",
        "128",
        "--depth",
        "3",
        "--num-heads",
        "8",
        "--mlp-ratio",
        "4",
        "--register-tokens",
        "8",
        "--style-vocab-size",
        "1",
        "--style-token-count",
        "0",
        "--precision",
        "fp32",
        "--scheduler",
        "none",
        "--learning-rate",
        "1e-4",
        "--use-queue",
        "--queue-prime-samples",
        str(int(args.queue_prime_samples)),
        "--queue-push-batch",
        str(int(args.queue_push_batch)),
        "--queue-per-class-capacity",
        str(int(args.queue_per_class_capacity)),
        "--queue-global-capacity",
        str(int(args.queue_global_capacity)),
        "--queue-warmup-mode",
        "class_balanced",
        "--queue-warmup-min-per-class",
        "2",
        "--queue-report-level",
        "basic",
        "--real-batch-source",
        "synthetic_dataset",
        "--real-dataset-size",
        str(int(args.real_dataset_size)),
        "--real-loader-batch-size",
        str(int(args.real_loader_batch_size)),
        "--output-dir",
        str(run_dir),
        "--checkpoint-path",
        str(checkpoint_path),
    ]


def _to_markdown(summary: dict[str, object]) -> str:
    rows = [
        "# Queue Hot-Path Benchmark (Latent)",
        "",
        f"- Generated at (UTC): `{summary['generated_at_utc']}`",
        f"- Device: `{summary['device']}`",
        f"- Steps: `{summary['steps']}`",
        "",
        "| returncode | mean_step_time_s | mean_img_per_sec | peak_cuda_mem_mb | underflow_missing_labels |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    entry = summary["results"][0]
    perf = entry.get("perf") or {}
    underflow = entry.get("queue_underflow_totals") or {}
    step_time = _fmt(perf.get("mean_step_time_s"))
    img_per_sec = _fmt(perf.get("mean_generated_images_per_sec"))
    peak_mem = _fmt(perf.get("max_peak_cuda_mem_mb"))
    missing = _fmt(underflow.get("missing_labels"))
    rows.append(f"| {entry['returncode']} | {step_time} | {img_per_sec} | {peak_mem} | {missing} |")
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
