from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from drifting_models.utils import add_device_argument, discover_repo_root, write_json


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
    output_root = _resolve_path(repo_root=repo_root, value=args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    required_paths = [
        repo_root / "scripts/runtime_preflight.py",
        repo_root / "scripts/train_toy.py",
        repo_root / "scripts/runtime_stable_lane.py",
        repo_root / "configs/stable/toy_quick.yaml",
        repo_root / "configs/stable/latent_smoke_feature_queue.yaml",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required newcomer assets:\n" + "\n".join(missing))

    records: list[CommandRecord] = []
    if not args.skip_preflight:
        preflight_cmd = [
            sys.executable,
            "scripts/runtime_preflight.py",
            "--device",
            str(args.device),
            "--check-torchvision",
            "--strict",
            "--output-path",
            str(output_root / "runtime_preflight.json"),
        ]
        records.append(_run(name="runtime_preflight", argv=preflight_cmd, cwd=repo_root, logs_dir=logs_dir))

    toy_dir = output_root / "toy_quick"
    toy_cmd = [
        sys.executable,
        "scripts/train_toy.py",
        "--config",
        "configs/stable/toy_quick.yaml",
        "--output-dir",
        str(toy_dir),
        "--device",
        "cpu",
    ]
    records.append(_run(name="train_toy", argv=toy_cmd, cwd=repo_root, logs_dir=logs_dir))

    stable_lane_cmd = [
        sys.executable,
        "scripts/runtime_stable_lane.py",
        "--config",
        "configs/stable/latent_smoke_feature_queue.yaml",
        "--device",
        str(args.device),
        "--base-dir",
        str(args.stable_base_dir),
        "--skip-preflight",
    ]
    if args.timestamp is not None:
        stable_lane_cmd += ["--timestamp", str(args.timestamp)]
    if args.stable_steps is not None:
        stable_lane_cmd += ["--steps", str(int(args.stable_steps))]
    stable_record = _run(name="runtime_stable_lane", argv=stable_lane_cmd, cwd=repo_root, logs_dir=logs_dir)
    records.append(stable_record)

    stable_summary = json.loads(Path(stable_record.stdout_path).read_text(encoding="utf-8"))
    stable_run_root = stable_summary.get("run_root")
    if not stable_run_root:
        raise RuntimeError("Failed to parse stable run root from runtime_stable_lane output")

    payload = {
        "status": "ok",
        "args": vars(args),
        "paths": {
            "output_root": str(output_root),
            "toy_output_dir": str(toy_dir),
            "stable_run_root": str(stable_run_root),
            "summary_json": str(output_root / "newcomer_smoke_summary.json"),
            "logs_dir": str(logs_dir),
        },
        "commands": [record.__dict__ for record in records],
    }
    write_json(output_root / "newcomer_smoke_summary.json", payload)
    print(json.dumps(payload, indent=2))


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
        description="Newcomer smoke path: strict preflight, toy train, then one-command stable lane smoke."
    )
    add_device_argument(parser, default="cpu")
    parser.add_argument("--output-root", type=str, default="outputs/onboarding/newcomer_smoke")
    parser.add_argument("--stable-base-dir", type=str, default="outputs/imagenet")
    parser.add_argument("--stable-steps", type=int, default=2)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
