from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from drifting_models.utils import (
    add_device_argument,
    build_stable_run_root,
    codebase_fingerprint,
    discover_repo_root,
    environment_fingerprint,
    environment_snapshot,
    write_json,
)


@dataclass(frozen=True)
class CommandRecord:
    name: str
    argv: list[str]
    returncode: int
    duration_seconds: float
    stdout_path: str
    stderr_path: str


def main() -> None:
    args = _parse_args()
    repo_root = discover_repo_root(Path(__file__))
    config_path = _resolve_path(repo_root=repo_root, value=args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    run_root = _resolve_run_root(repo_root=repo_root, args=args)
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_root / "checkpoints"
    logs_dir = run_root / "pipeline_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    records: list[CommandRecord] = []
    if not args.skip_preflight:
        preflight_cmd = [
            sys.executable,
            "scripts/runtime_preflight.py",
            "--device",
            str(args.device),
            "--check-torchvision",
            "--output-path",
            str(run_root / "runtime_preflight.json"),
        ]
        if not args.preflight_non_strict:
            preflight_cmd.append("--strict")
        records.append(_run(name="runtime_preflight", argv=preflight_cmd, cwd=repo_root, logs_dir=logs_dir))

    train_cmd = [
        sys.executable,
        "scripts/train_latent.py",
        "--config",
        str(config_path),
        "--device",
        str(args.device),
        "--output-dir",
        str(run_root),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-path",
        str(run_root / "checkpoint.pt"),
    ]
    if args.steps is not None:
        train_cmd += ["--steps", str(int(args.steps))]
    records.append(_run(name="train_latent", argv=train_cmd, cwd=repo_root, logs_dir=logs_dir))

    validate_cmd = [
        sys.executable,
        "scripts/validate_run_artifacts.py",
        "--run-root",
        str(run_root),
        "--lane",
        "stable",
        "--output-json",
        str(run_root / "artifact_validation.json"),
    ]
    if not args.require_eval_summaries:
        validate_cmd.append("--allow-missing-eval-summaries")
    records.append(_run(name="validate_run_artifacts", argv=validate_cmd, cwd=repo_root, logs_dir=logs_dir))

    write_json(run_root / "env_snapshot.json", environment_snapshot(paths=[run_root]))
    write_json(run_root / "env_fingerprint.json", environment_fingerprint())
    write_json(run_root / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))

    payload = {
        "status": "ok",
        "run_root": str(run_root),
        "config": str(config_path),
        "device": str(args.device),
        "args": vars(args),
        "commands": [record.__dict__ for record in records],
        "paths": {
            "run_root": str(run_root),
            "checkpoint_path": str(run_root / "checkpoint.pt"),
            "checkpoint_dir": str(checkpoint_dir),
            "validation_json": str(run_root / "artifact_validation.json"),
            "summary_json": str(run_root / "stable_lane_summary.json"),
            "logs_dir": str(logs_dir),
        },
    }
    write_json(run_root / "stable_lane_summary.json", payload)
    print(json.dumps(payload, indent=2))


def _resolve_run_root(*, repo_root: Path, args: argparse.Namespace) -> Path:
    if args.run_root is not None:
        return _resolve_path(repo_root=repo_root, value=args.run_root)
    base_dir = _resolve_path(repo_root=repo_root, value=args.base_dir)
    return build_stable_run_root(base_dir=base_dir, timestamp=args.timestamp)


def _resolve_path(*, repo_root: Path, value: str) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (repo_root / raw).resolve()


def _run(*, name: str, argv: list[str], cwd: Path, logs_dir: Path) -> CommandRecord:
    started = time.perf_counter()
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    duration_seconds = time.perf_counter() - started
    stdout_path = logs_dir / f"{name}.stdout.log"
    stderr_path = logs_dir / f"{name}.stderr.log"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        stderr_tail = _tail_text(result.stderr)
        raise RuntimeError(
            f"{name} failed (code={result.returncode}).\n"
            f"command: {' '.join(map(str, argv))}\n"
            f"stderr tail:\n{stderr_tail}\n"
            f"full logs: {stderr_path}"
        )
    return CommandRecord(
        name=name,
        argv=list(map(str, argv)),
        returncode=int(result.returncode),
        duration_seconds=float(duration_seconds),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def _tail_text(value: str, *, lines: int = 20) -> str:
    rows = value.splitlines()
    if not rows:
        return "<empty>"
    return "\n".join(rows[-lines:])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command stable lane wrapper: preflight -> train_latent -> artifact validation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable/latent_smoke_feature_queue.yaml",
        help="Stable config path. Override for longer claim-facing runs.",
    )
    add_device_argument(parser, default="cpu")
    parser.add_argument("--base-dir", type=str, default="outputs/imagenet")
    parser.add_argument("--run-root", type=str, default=None, help="Explicit run root. Overrides --base-dir/--timestamp.")
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None, help="Optional override for train_latent --steps.")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-non-strict", action="store_true")
    parser.add_argument("--require-eval-summaries", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
