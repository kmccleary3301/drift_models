from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from drifting_models.utils import add_device_argument, environment_fingerprint, write_json


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    mae_dir = output_root / "mae"
    pixel_dir = output_root / "pixel"
    mae_dir.mkdir(parents=True, exist_ok=True)
    pixel_dir.mkdir(parents=True, exist_ok=True)

    export_path = mae_dir / "mae_encoder.pt"
    mae_cmd = [
        sys.executable,
        "scripts/train_mae.py",
        "--device",
        args.device,
        "--steps",
        str(args.mae_steps),
        "--log-every",
        str(args.mae_log_every),
        "--batch-size",
        str(args.mae_batch_size),
        "--in-channels",
        str(args.channels),
        "--base-channels",
        str(args.mae_base_channels),
        "--stages",
        str(args.mae_stages),
        "--export-encoder-path",
        str(export_path),
        "--output-dir",
        str(mae_dir),
    ]
    mae_result = _run(mae_cmd, cwd=repo_root)
    if mae_result.returncode != 0:
        raise RuntimeError(mae_result.stderr)
    if not export_path.exists():
        raise FileNotFoundError(f"Missing MAE encoder export: {export_path}")

    pixel_cmd = [
        sys.executable,
        "scripts/train_pixel.py",
        "--device",
        args.device,
        "--steps",
        str(args.pixel_steps),
        "--log-every",
        str(args.pixel_log_every),
        "--mae-encoder-path",
        str(export_path),
        "--output-dir",
        str(pixel_dir),
    ]
    if args.pixel_config is not None:
        pixel_cmd += ["--config", args.pixel_config]
    pixel_result = _run(pixel_cmd, cwd=repo_root)
    if pixel_result.returncode != 0:
        raise RuntimeError(pixel_result.stderr)

    pipeline_summary = {
        "output_root": str(output_root),
        "paths": {
            "mae_dir": str(mae_dir),
            "pixel_dir": str(pixel_dir),
            "export_path": str(export_path),
            "mae_summary_path": str(mae_dir / "mae_summary.json"),
            "pixel_summary_path": str(pixel_dir / "pixel_summary.json"),
        },
        "args": vars(args),
        "commands": {
            "mae": mae_result.__dict__,
            "pixel": pixel_result.__dict__,
        },
        "env_fingerprint": environment_fingerprint(),
    }
    write_json(output_root / "pipeline_summary.json", pipeline_summary)
    print(json.dumps(pipeline_summary, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAE export -> pixel train pipeline")
    parser.add_argument("--output-root", type=str, required=True)
    add_device_argument(parser, default="cpu")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--mae-steps", type=int, default=2)
    parser.add_argument("--mae-log-every", type=int, default=1)
    parser.add_argument("--mae-batch-size", type=int, default=8)
    parser.add_argument("--mae-base-channels", type=int, default=8)
    parser.add_argument("--mae-stages", type=int, default=2)
    parser.add_argument("--pixel-config", type=str, default=None)
    parser.add_argument("--pixel-steps", type=int, default=2)
    parser.add_argument("--pixel-log-every", type=int, default=1)
    return parser.parse_args()


def _run(argv: list[str], *, cwd: Path) -> CommandResult:
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    return CommandResult(argv=list(map(str, argv)), returncode=int(result.returncode), stdout=result.stdout, stderr=result.stderr)


if __name__ == "__main__":
    main()
