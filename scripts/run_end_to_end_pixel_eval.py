from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from drifting_models.utils import add_device_argument, codebase_fingerprint, environment_fingerprint, environment_snapshot, write_json
from drifting_models.utils.experiment_log import (
    append_experiment_log,
    build_entry_from_summary,
    default_run_id,
)
from drifting_models.utils.run_md import write_run_md


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
    train_dir = output_root / "train"
    sample_dir = output_root / "samples"
    eval_dir = output_root / "eval"
    for p in (train_dir, sample_dir, eval_dir):
        p.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "env_snapshot.json", environment_snapshot(paths=[output_root]))
    write_json(output_root / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))

    checkpoint_path = train_dir / "checkpoint.pt"

    train_cmd = [
        sys.executable,
        "scripts/train_pixel.py",
        "--device",
        args.device,
        "--steps",
        str(args.train_steps),
        "--log-every",
        str(args.train_log_every),
        "--checkpoint-path",
        str(checkpoint_path),
        "--save-every",
        str(args.train_save_every),
        "--output-dir",
        str(train_dir),
    ]
    if args.train_config is not None:
        train_cmd += ["--config", args.train_config]
    train_result = _run(train_cmd, cwd=repo_root)
    if train_result.returncode != 0:
        raise RuntimeError(train_result.stderr)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    sample_cmd = [
        sys.executable,
        "scripts/sample_pixel.py",
        "--device",
        args.device,
        "--checkpoint-path",
        str(checkpoint_path),
        "--output-root",
        str(sample_dir),
        "--n-samples",
        str(args.sample_count),
        "--batch-size",
        str(args.sample_batch_size),
    ]
    if args.train_config is not None:
        sample_cmd += ["--config", args.train_config]
    sample_result = _run(sample_cmd, cwd=repo_root)
    if sample_result.returncode != 0:
        raise RuntimeError(sample_result.stderr)

    generated_root = sample_dir / "images"
    if not generated_root.exists():
        raise FileNotFoundError(f"Missing generated ImageFolder: {generated_root}")

    ref_stats_path = eval_dir / "reference_stats.pt"
    eval_cmd = [
        sys.executable,
        "scripts/eval_fid_is.py",
        "--device",
        args.device,
        "--batch-size",
        str(args.eval_batch_size),
        "--inception-weights",
        args.inception_weights,
        "--reference-source",
        "imagefolder",
        "--reference-imagefolder-root",
        args.reference_imagefolder_root,
        "--generated-source",
        "imagefolder",
        "--generated-imagefolder-root",
        str(generated_root),
        "--output-path",
        str(eval_dir / "eval_summary.json"),
    ]
    if args.cache_reference_stats:
        eval_cmd += ["--cache-reference-stats", str(ref_stats_path)]
    eval_result = _run(eval_cmd, cwd=repo_root)
    if eval_result.returncode != 0:
        raise RuntimeError(eval_result.stderr)

    run_summary = {
        "output_root": str(output_root),
        "paths": {
            "train_dir": str(train_dir),
            "checkpoint_path": str(checkpoint_path),
            "sample_dir": str(sample_dir),
            "generated_root": str(generated_root),
            "eval_dir": str(eval_dir),
            "eval_summary_path": str(eval_dir / "eval_summary.json"),
            "reference_stats_path": str(ref_stats_path) if args.cache_reference_stats else None,
        },
        "args": vars(args),
        "commands": {
            "train": train_result.__dict__,
            "sample": sample_result.__dict__,
            "eval": eval_result.__dict__,
        },
        "env_fingerprint": environment_fingerprint(),
    }
    write_json(output_root / "run_summary.json", run_summary)
    write_run_md(output_root / "RUN.md", run_summary)
    if args.append_experiment_log:
        run_id = args.experiment_run_id or default_run_id(args.experiment_run_id_prefix)
        entry = build_entry_from_summary(
            kind="end_to_end_pixel",
            run_id=run_id,
            date=args.experiment_date,
            summary=run_summary,
            notes=args.experiment_notes,
        )
        append_experiment_log(log_path=Path(args.experiment_log_path), entry_md=entry)
    print(json.dumps(run_summary, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train -> sample -> eval end-to-end pixel smoke runner")
    parser.add_argument("--output-root", type=str, required=True)
    add_device_argument(parser, default="cpu")
    parser.add_argument("--train-config", type=str, default=None)
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--train-log-every", type=int, default=1)
    parser.add_argument("--train-save-every", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--sample-batch-size", type=int, default=16)
    parser.add_argument("--reference-imagefolder-root", type=str, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--inception-weights", choices=("none", "pretrained"), default="none")
    parser.add_argument("--cache-reference-stats", action="store_true")
    parser.add_argument("--append-experiment-log", action="store_true")
    parser.add_argument("--experiment-log-path", type=str, default="docs/experiment_log.md")
    parser.add_argument("--experiment-run-id", type=str, default=None)
    parser.add_argument("--experiment-run-id-prefix", type=str, default="EXP-E2E-PIXEL")
    parser.add_argument("--experiment-date", type=str, default=None)
    parser.add_argument("--experiment-notes", type=str, default=None)
    return parser.parse_args()


def _run(argv: list[str], *, cwd: Path) -> CommandResult:
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    return CommandResult(argv=list(map(str, argv)), returncode=int(result.returncode), stdout=result.stdout, stderr=result.stderr)


if __name__ == "__main__":
    main()
