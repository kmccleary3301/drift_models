from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from drifting_models.utils import add_device_argument


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_a = _run_once(repo_root=repo_root, seed=int(args.seed), device=str(args.device))
    run_b = _run_once(repo_root=repo_root, seed=int(args.seed), device=str(args.device))

    payload_a = json.loads(run_a)
    payload_b = json.loads(run_b)

    warmup_a = payload_a.get("queue_warmup_report", {})
    warmup_b = payload_b.get("queue_warmup_report", {})
    underflow_a = payload_a.get("queue_underflow_totals", {})
    underflow_b = payload_b.get("queue_underflow_totals", {})
    fingerprint_a = payload_a.get("real_batch_provider", {}).get("manifest_fingerprint")
    fingerprint_b = payload_b.get("real_batch_provider", {}).get("manifest_fingerprint")

    report = {
        "seed": int(args.seed),
        "device": args.device,
        "warmup_equal": warmup_a == warmup_b,
        "underflow_equal": underflow_a == underflow_b,
        "manifest_fingerprint_equal": fingerprint_a == fingerprint_b,
        "warmup_a": warmup_a,
        "warmup_b": warmup_b,
        "underflow_a": underflow_a,
        "underflow_b": underflow_b,
        "manifest_fingerprint_a": fingerprint_a,
        "manifest_fingerprint_b": fingerprint_b,
    }
    report["passed"] = bool(
        report["warmup_equal"] and report["underflow_equal"] and report["manifest_fingerprint_equal"]
    )
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


def _run_once(*, repo_root: Path, seed: int, device: str) -> str:
    command = [
        sys.executable,
        "scripts/train_latent.py",
        "--device",
        device,
        "--steps",
        "2",
        "--log-every",
        "1",
        "--groups",
        "2",
        "--negatives-per-group",
        "2",
        "--positives-per-group",
        "2",
        "--image-size",
        "16",
        "--patch-size",
        "4",
        "--hidden-dim",
        "64",
        "--depth",
        "2",
        "--num-heads",
        "4",
        "--num-classes",
        "32",
        "--seed",
        str(seed),
        "--use-queue",
        "--queue-prime-samples",
        "128",
        "--queue-warmup-mode",
        "class_balanced",
        "--queue-warmup-min-per-class",
        "2",
        "--queue-push-batch",
        "32",
        "--queue-per-class-capacity",
        "32",
        "--queue-global-capacity",
        "512",
        "--real-batch-source",
        "synthetic_dataset",
        "--real-dataset-size",
        "4096",
        "--real-loader-batch-size",
        "128",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two queue-enabled latent runs and compare determinism signals.")
    parser.add_argument("--seed", type=int, default=1337)
    add_device_argument(parser, default="cpu")
    parser.add_argument("--output-path", type=str, default="outputs/stage4_queue_determinism/queue_determinism.json")
    return parser.parse_args()


if __name__ == "__main__":
    main()
