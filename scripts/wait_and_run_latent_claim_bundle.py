from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from drifting_models.utils import add_device_argument, codebase_fingerprint, environment_fingerprint, environment_snapshot, write_json
from drifting_models.utils.run_md import write_run_md


@dataclass(frozen=True)
class CommandRecord:
    name: str
    argv: list[str]
    returncode: int
    duration_seconds: float


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = (repo_root / args.run_dir).resolve()
    config_path = (repo_root / args.config_path).resolve() if args.config_path is not None else run_dir / "config.yaml"
    checkpoint_dir = run_dir / "checkpoints"
    output_root = (repo_root / args.output_root).resolve() if args.output_root is not None else run_dir / "claim_bundle"
    output_root.mkdir(parents=True, exist_ok=True)

    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run_dir: {run_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint dir: {checkpoint_dir}")

    target_step = int(args.target_step) if args.target_step is not None else _read_steps_from_config(config_path=config_path)
    final_checkpoint = checkpoint_dir / f"checkpoint_step_{target_step:08d}.pt"
    state_path = output_root / "wait_state.json"

    write_json(output_root / "env_snapshot.json", environment_snapshot(paths=[output_root]))
    write_json(output_root / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))
    write_json(output_root / "env_fingerprint.json", environment_fingerprint())
    write_run_md(
        output_root / "RUN.md",
        {
            "run_dir": str(run_dir),
            "config_path": str(config_path),
            "target_step": int(target_step),
            "final_checkpoint": str(final_checkpoint),
            "args": vars(args),
            "paths": {
                "output_root": str(output_root),
                "wait_state_json": str(state_path),
                "env_snapshot_json": str(output_root / "env_snapshot.json"),
                "codebase_fingerprint_json": str(output_root / "codebase_fingerprint.json"),
                "env_fingerprint_json": str(output_root / "env_fingerprint.json"),
            },
            "commands": {"invocation": {"argv": [sys.executable, *sys.argv]}},
        },
    )

    _log(f"waiting_for_final_checkpoint path={final_checkpoint}")
    while not final_checkpoint.exists():
        latest_step, latest_path = _find_latest_step_checkpoint(checkpoint_dir=checkpoint_dir)
        state = {
            "status": "waiting",
            "timestamp": _timestamp(),
            "target_step": int(target_step),
            "final_checkpoint_exists": final_checkpoint.exists(),
            "latest_step": int(latest_step) if latest_step is not None else None,
            "latest_checkpoint": str(latest_path) if latest_path is not None else None,
            "free_gb_drive_4": _free_gb(Path(args.drive_mount)),
        }
        write_json(state_path, state)
        time.sleep(float(args.poll_seconds))

    _log(f"final_checkpoint_ready path={final_checkpoint}")
    write_json(
        state_path,
        {
            "status": "running_bundle",
            "timestamp": _timestamp(),
            "target_step": int(target_step),
            "final_checkpoint": str(final_checkpoint),
            "free_gb_drive_4": _free_gb(Path(args.drive_mount)),
        },
    )

    alpha_key = _format_alpha(float(args.claim_alpha))
    alpha_sweep_root = output_root / "alpha_sweep"
    last_k_root = output_root / "last_k_eval"
    claim_sample_root = output_root / f"claim_samples_alpha_{alpha_key}"
    claim_eval_root = output_root / "claim_eval"
    claim_eval_root.mkdir(parents=True, exist_ok=True)

    records: list[CommandRecord] = []
    records.append(
        _run(
            name="last_k_eval",
            argv=[
                sys.executable,
                "scripts/eval_last_k_checkpoints.py",
                "--output-root",
                str(last_k_root),
                "--device",
                str(args.device),
                "--mode",
                "latent",
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--k",
                str(int(args.last_k)),
                "--config",
                str(config_path),
                "--n-samples",
                str(int(args.last_k_samples)),
                "--batch-size",
                str(int(args.last_k_batch_size)),
                "--decode-mode",
                "sd_vae",
                "--decode-image-size",
                "256",
                "--sd-vae-model-id",
                str(args.sd_vae_model_id),
                "--reference-imagefolder-root",
                str((repo_root / args.reference_imagefolder_root).resolve()),
                "--inception-weights",
                "pretrained",
                "--load-reference-stats",
                str((repo_root / args.load_reference_stats).resolve()),
                "--overwrite",
                "--append-experiment-log",
                "--experiment-run-id-prefix",
                "EXP-G3-LASTK-BUNDLE",
                "--experiment-notes",
                f"postrun claim bundle for {run_dir.name}",
            ],
            cwd=repo_root,
        )
    )
    records.append(
        _run(
            name="alpha_sweep",
            argv=[
                sys.executable,
                "scripts/eval_alpha_sweep.py",
                "--output-root",
                str(alpha_sweep_root),
                "--device",
                str(args.device),
                "--mode",
                "latent",
                "--checkpoint-path",
                str(final_checkpoint),
                "--config",
                str(config_path),
                "--alphas",
                *[str(value) for value in args.alpha_sweep],
                "--n-samples",
                str(int(args.alpha_sweep_samples)),
                "--batch-size",
                str(int(args.alpha_sweep_batch_size)),
                "--decode-mode",
                "sd_vae",
                "--decode-image-size",
                "256",
                "--sd-vae-model-id",
                str(args.sd_vae_model_id),
                "--reference-imagefolder-root",
                str((repo_root / args.reference_imagefolder_root).resolve()),
                "--inception-weights",
                "pretrained",
                "--load-reference-stats",
                str((repo_root / args.load_reference_stats).resolve()),
                "--overwrite",
                "--append-experiment-log",
                "--experiment-run-id-prefix",
                "EXP-G3-ALPHA-BUNDLE",
                "--experiment-notes",
                f"postrun claim bundle for {run_dir.name}",
            ],
            cwd=repo_root,
        )
    )

    alpha_nn_generated_root = alpha_sweep_root / f"alpha_{alpha_key}" / "samples" / "images"
    records.append(
        _run(
            name="alpha_nn_audit",
            argv=[
                sys.executable,
                "scripts/audit_nearest_neighbors.py",
                "--generated-root",
                str(alpha_nn_generated_root),
                "--reference-root",
                str((repo_root / args.reference_imagefolder_root).resolve()),
                "--device",
                str(args.device),
                "--max-generated",
                str(int(args.nn_max_generated)),
                "--max-reference",
                str(int(args.nn_max_reference)),
                "--output-path",
                str(alpha_sweep_root / f"alpha_{alpha_key}" / "nn_audit.json"),
            ],
            cwd=repo_root,
        )
    )

    records.append(
        _run(
            name="claim_sample_set",
            argv=[
                sys.executable,
                "scripts/sample_latent.py",
                "--device",
                str(args.device),
                "--checkpoint-path",
                str(final_checkpoint),
                "--config",
                str(config_path),
                "--output-root",
                str(claim_sample_root),
                "--n-samples",
                str(int(args.claim_sample_count)),
                "--batch-size",
                str(int(args.claim_sample_batch_size)),
                "--alpha",
                str(float(args.claim_alpha)),
                "--write-imagefolder",
                "--decode-mode",
                "sd_vae",
                "--decode-image-size",
                "256",
                "--sd-vae-model-id",
                str(args.sd_vae_model_id),
            ],
            cwd=repo_root,
        )
    )

    records.append(
        _run(
            name="claim_eval",
            argv=[
                sys.executable,
                "scripts/eval_fid_is.py",
                "--device",
                str(args.device),
                "--batch-size",
                str(int(args.claim_eval_batch_size)),
                "--inception-weights",
                "pretrained",
                "--reference-source",
                "imagefolder",
                "--reference-imagefolder-root",
                str((repo_root / args.reference_imagefolder_root).resolve()),
                "--generated-source",
                "imagefolder",
                "--generated-imagefolder-root",
                str(claim_sample_root / "images"),
                "--load-reference-stats",
                str((repo_root / args.load_reference_stats).resolve()),
                "--output-path",
                str(claim_eval_root / "eval_pretrained.json"),
            ],
            cwd=repo_root,
        )
    )

    records.append(
        _run(
            name="claim_nn_audit",
            argv=[
                sys.executable,
                "scripts/audit_nearest_neighbors.py",
                "--generated-root",
                str(claim_sample_root / "images"),
                "--reference-root",
                str((repo_root / args.reference_imagefolder_root).resolve()),
                "--device",
                str(args.device),
                "--max-generated",
                str(int(args.nn_max_generated)),
                "--max-reference",
                str(int(args.nn_max_reference)),
                "--output-path",
                str(claim_eval_root / "nn_audit.json"),
            ],
            cwd=repo_root,
        )
    )

    summary = {
        "status": "done",
        "timestamp": _timestamp(),
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "target_step": int(target_step),
        "final_checkpoint": str(final_checkpoint),
        "paths": {
            "output_root": str(output_root),
            "last_k_root": str(last_k_root),
            "alpha_sweep_root": str(alpha_sweep_root),
            "claim_sample_root": str(claim_sample_root),
            "claim_eval_root": str(claim_eval_root),
            "wait_state_json": str(state_path),
            "bundle_summary_json": str(output_root / "bundle_summary.json"),
        },
        "commands": [record.__dict__ for record in records],
        "env_fingerprint": environment_fingerprint(),
    }
    write_json(output_root / "bundle_summary.json", summary)
    write_json(state_path, {"status": "done", "timestamp": _timestamp(), "summary_path": str(output_root / "bundle_summary.json")})
    print(json.dumps(summary, indent=2))


def _read_steps_from_config(*, config_path: Path) -> int:
    text = config_path.read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s*steps\s*:\s*(\d+)\s*$", text)
    if match is None:
        raise ValueError(f"Could not parse steps from config: {config_path}")
    return int(match.group(1))


def _find_latest_step_checkpoint(*, checkpoint_dir: Path) -> tuple[int | None, Path | None]:
    best_step: int | None = None
    best_path: Path | None = None
    for path in checkpoint_dir.glob("checkpoint_step_*.pt"):
        match = re.fullmatch(r"checkpoint_step_(\d{8})\.pt", path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if best_step is None or step > best_step:
            best_step = step
            best_path = path
    return best_step, best_path


def _run(*, name: str, argv: list[str], cwd: Path) -> CommandRecord:
    _log(f"run[{name}] {' '.join(argv)}")
    started = time.time()
    result = subprocess.run(argv, cwd=cwd, check=False)
    duration_seconds = float(time.time() - started)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed name={name} rc={result.returncode}")
    _log(f"ok[{name}] duration={duration_seconds:.1f}s")
    return CommandRecord(
        name=name,
        argv=list(map(str, argv)),
        returncode=int(result.returncode),
        duration_seconds=duration_seconds,
    )


def _format_alpha(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return float(usage.free) / (1024.0**3)


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for final latent checkpoint then run full claim/evidence bundle.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--target-step", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=180.0)
    parser.add_argument("--drive-mount", type=str, default="/mnt/drive_4")
    add_device_argument(parser, default="cuda:0")
    parser.add_argument("--sd-vae-model-id", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--reference-imagefolder-root", type=str, default="outputs/datasets/imagenet1k_raw/val")
    parser.add_argument("--load-reference-stats", type=str, default="outputs/datasets/imagenet1k_val_reference_stats_pretrained.pt")
    parser.add_argument("--last-k", type=int, default=2)
    parser.add_argument("--last-k-samples", type=int, default=2000)
    parser.add_argument("--last-k-batch-size", type=int, default=32)
    parser.add_argument("--alpha-sweep", nargs="+", type=float, default=[1.0, 1.5, 2.0, 2.5, 3.0])
    parser.add_argument("--alpha-sweep-samples", type=int, default=2000)
    parser.add_argument("--alpha-sweep-batch-size", type=int, default=32)
    parser.add_argument("--claim-alpha", type=float, default=1.5)
    parser.add_argument("--claim-sample-count", type=int, default=50000)
    parser.add_argument("--claim-sample-batch-size", type=int, default=32)
    parser.add_argument("--claim-eval-batch-size", type=int, default=64)
    parser.add_argument("--nn-max-generated", type=int, default=512)
    parser.add_argument("--nn-max-reference", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    main()
