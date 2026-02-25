from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from drifting_models.utils import add_device_argument, codebase_fingerprint, environment_fingerprint, environment_snapshot, write_json
from drifting_models.utils.run_md import write_run_md


@dataclass(frozen=True)
class MAEWidthSpec:
    name: str
    config_path: str
    output_dir: str
    expected_base_channels: int


_DEFAULT_SPECS: tuple[MAEWidthSpec, ...] = (
    MAEWidthSpec(
        name="w256",
        config_path="configs/mae/imagenet1k_sdvae_latents_shards_table8_w256_bootstrap.yaml",
        output_dir="outputs/imagenet/mae_variant_a_w256",
        expected_base_channels=256,
    ),
    MAEWidthSpec(
        name="w640",
        config_path="configs/mae/imagenet1k_sdvae_latents_shards_table8_w640_bootstrap.yaml",
        output_dir="outputs/imagenet/mae_variant_a_w640",
        expected_base_channels=640,
    ),
)


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing tensor-shards manifest: {manifest_path}")

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = (repo_root / args.output_root / f"mae_width_parity_exports_{run_stamp}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "status.json"

    write_json(run_root / "env_snapshot.json", environment_snapshot(paths=[run_root]))
    write_json(run_root / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))
    write_json(run_root / "env_fingerprint.json", environment_fingerprint())

    summary: dict[str, object] = {
        "status": "running",
        "timestamp_start": _timestamp(),
        "repo_root": str(repo_root),
        "args": vars(args),
        "paths": {
            "run_root": str(run_root),
            "status_json": str(status_path),
            "manifest_path": str(manifest_path),
            "env_snapshot_json": str(run_root / "env_snapshot.json"),
            "codebase_fingerprint_json": str(run_root / "codebase_fingerprint.json"),
            "env_fingerprint_json": str(run_root / "env_fingerprint.json"),
        },
        "variants": [],
    }
    write_json(status_path, summary)
    write_run_md(
        run_root / "RUN.md",
        {
            "output_root": str(run_root),
            "args": vars(args),
            "paths": summary["paths"],
            "commands": {"invocation": {"argv": [sys.executable, *sys.argv], "returncode": 0}},
        },
    )

    if args.wait_state_path is not None:
        wait_state_path = (repo_root / args.wait_state_path).resolve()
        wait_record = _wait_for_state_done(
            wait_state_path=wait_state_path,
            poll_seconds=float(args.poll_seconds),
            timeout_hours=float(args.wait_timeout_hours),
        )
        summary["wait_state_result"] = wait_record
        write_json(status_path, summary)

    if args.wait_for_gpu_idle:
        idle_record = _wait_for_gpu_idle(
            poll_seconds=float(args.poll_seconds),
            timeout_hours=float(args.wait_timeout_hours),
        )
        summary["gpu_idle_wait_result"] = idle_record
        write_json(status_path, summary)

    for spec in _DEFAULT_SPECS:
        variant_record: dict[str, object] = {
            "name": spec.name,
            "config_path": spec.config_path,
            "output_dir": spec.output_dir,
            "expected_base_channels": int(spec.expected_base_channels),
            "timestamp_start": _timestamp(),
        }
        summary["variants"].append(variant_record)
        write_json(status_path, summary)

        free_gb = _free_gb(Path(args.drive_mount))
        variant_record["free_gb_before"] = float(free_gb)
        if free_gb < float(args.min_free_gb):
            raise RuntimeError(
                f"Refusing to run {spec.name}: free space {free_gb:.1f}GB < min_free_gb={args.min_free_gb}"
            )

        output_dir = (repo_root / spec.output_dir).resolve()
        export_path = output_dir / "mae_encoder.pt"
        variant_record["export_path"] = str(export_path)

        if export_path.exists() and args.skip_existing:
            variant_record["status"] = "skipped_existing"
            variant_record["bundle_audit"] = _audit_mae_bundle(output_dir=output_dir, expected_base_channels=spec.expected_base_channels)
            variant_record["timestamp_end"] = _timestamp()
            write_json(status_path, summary)
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        train_log_path = run_root / f"{spec.name}_train.log"
        train_argv = [
            sys.executable,
            "scripts/train_mae.py",
            "--config",
            spec.config_path,
            "--device",
            args.device,
            "--output-dir",
            spec.output_dir,
            "--checkpoint-path",
            str(Path(spec.output_dir) / "checkpoint.pt"),
            "--export-encoder-path",
            str(Path(spec.output_dir) / "mae_encoder.pt"),
            "--real-tensor-shards-manifest-path",
            str(manifest_path),
            "--save-every",
            str(int(args.save_every)),
        ]
        variant_record["train_argv"] = [str(value) for value in train_argv]
        variant_record["train_log_path"] = str(train_log_path)
        variant_record["status"] = "running"
        write_json(status_path, summary)

        with train_log_path.open("w", encoding="utf-8") as log_handle:
            result = subprocess.run(train_argv, cwd=repo_root, stdout=log_handle, stderr=subprocess.STDOUT, text=True, check=False)

        variant_record["returncode"] = int(result.returncode)
        if result.returncode != 0:
            variant_record["status"] = "failed"
            variant_record["log_tail"] = _tail(train_log_path, lines=80)
            summary["status"] = "failed"
            summary["timestamp_end"] = _timestamp()
            write_json(status_path, summary)
            raise RuntimeError(f"MAE width export run failed for {spec.name}; see {train_log_path}")

        variant_record["bundle_audit"] = _audit_mae_bundle(output_dir=output_dir, expected_base_channels=spec.expected_base_channels)
        variant_record["status"] = "done"
        variant_record["timestamp_end"] = _timestamp()
        variant_record["free_gb_after"] = float(_free_gb(Path(args.drive_mount)))
        write_json(status_path, summary)

    summary["status"] = "done"
    summary["timestamp_end"] = _timestamp()
    write_json(run_root / "summary.json", summary)
    write_json(status_path, summary)
    print(json.dumps(summary, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run width-parity MAE exports (w256 + w640) with provenance and optional upstream wait gates."
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="outputs/datasets/imagenet1k_train_sdvae_latents_shards/manifest.json",
    )
    add_device_argument(parser, default="cuda:0")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--wait-for-gpu-idle", action="store_true", dest="wait_for_gpu_idle")
    parser.add_argument("--wait-state-path", type=str, default=None)
    parser.add_argument("--poll-seconds", type=float, default=180.0)
    parser.add_argument("--wait-timeout-hours", type=float, default=336.0)
    parser.add_argument("--drive-mount", type=str, default="/mnt/drive_4")
    parser.add_argument("--min-free-gb", type=float, default=220.0)
    parser.add_argument("--output-root", type=str, default="outputs/ops")
    return parser.parse_args()


def _wait_for_state_done(*, wait_state_path: Path, poll_seconds: float, timeout_hours: float) -> dict[str, object]:
    started = time.time()
    timeout_s = max(1.0, float(timeout_hours) * 3600.0)
    last_state: dict[str, object] | None = None
    while True:
        elapsed_s = time.time() - started
        if elapsed_s > timeout_s:
            raise TimeoutError(f"Timed out waiting for done state at {wait_state_path}")
        if wait_state_path.exists():
            raw = wait_state_path.read_text(encoding="utf-8")
            state = json.loads(raw)
            if isinstance(state, dict):
                last_state = state
                if str(state.get("status")) == "done":
                    return {
                        "wait_state_path": str(wait_state_path),
                        "status": "done",
                        "elapsed_seconds": float(elapsed_s),
                        "last_state": last_state,
                    }
        time.sleep(max(1.0, float(poll_seconds)))


def _wait_for_gpu_idle(*, poll_seconds: float, timeout_hours: float) -> dict[str, object]:
    started = time.time()
    timeout_s = max(1.0, float(timeout_hours) * 3600.0)
    last_active: list[str] = []
    while True:
        elapsed_s = time.time() - started
        if elapsed_s > timeout_s:
            raise TimeoutError("Timed out waiting for GPU-bound run queue to go idle")
        active = _active_gpu_job_lines()
        if not active:
            return {
                "status": "idle",
                "elapsed_seconds": float(elapsed_s),
                "last_active": last_active,
            }
        last_active = active
        time.sleep(max(1.0, float(poll_seconds)))


def _active_gpu_job_lines() -> list[str]:
    result = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []
    keywords = (
        "scripts/train_latent.py",
        "scripts/train_mae.py",
        "scripts/sample_latent.py",
        "scripts/eval_fid_is.py",
        "scripts/eval_alpha_sweep.py",
        "scripts/eval_last_k_checkpoints.py",
        "scripts/audit_nearest_neighbors.py",
    )
    lines = []
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "rg " in line:
            continue
        if any(keyword in line for keyword in keywords):
            lines.append(line)
    return lines


def _audit_mae_bundle(*, output_dir: Path, expected_base_channels: int) -> dict[str, object]:
    required = (
        output_dir / "mae_summary.json",
        output_dir / "mae_encoder.pt",
        output_dir / "RUN.md",
        output_dir / "env_snapshot.json",
        output_dir / "codebase_fingerprint.json",
        output_dir / "env_fingerprint.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing MAE bundle files for {output_dir}: {missing}")

    summary_payload = json.loads((output_dir / "mae_summary.json").read_text(encoding="utf-8"))
    if not isinstance(summary_payload, dict):
        raise ValueError(f"Invalid JSON payload: {output_dir / 'mae_summary.json'}")
    summary_config = summary_payload.get("config")
    if not isinstance(summary_config, dict):
        raise ValueError(f"Missing `config` object in {output_dir / 'mae_summary.json'}")

    export_payload = torch.load(output_dir / "mae_encoder.pt", map_location="cpu")
    if not isinstance(export_payload, dict):
        raise ValueError(f"Invalid torch payload at {output_dir / 'mae_encoder.pt'}")
    model_config = export_payload.get("mae_model_config")
    if not isinstance(model_config, dict):
        raise ValueError(f"Missing `mae_model_config` in {output_dir / 'mae_encoder.pt'}")

    summary_base_channels = int(summary_config.get("base_channels", -1))
    summary_encoder_arch = str(summary_config.get("encoder_arch"))
    model_base_channels = int(model_config.get("base_channels", -1))
    model_encoder_arch = str(model_config.get("encoder_arch"))

    if summary_base_channels != int(expected_base_channels):
        raise ValueError(
            f"Summary base_channels mismatch for {output_dir}: {summary_base_channels} != {expected_base_channels}"
        )
    if model_base_channels != int(expected_base_channels):
        raise ValueError(
            f"Export base_channels mismatch for {output_dir}: {model_base_channels} != {expected_base_channels}"
        )
    if summary_encoder_arch != "paper_resnet34_unet":
        raise ValueError(f"Summary encoder_arch mismatch for {output_dir}: {summary_encoder_arch}")
    if model_encoder_arch != "paper_resnet34_unet":
        raise ValueError(f"Export encoder_arch mismatch for {output_dir}: {model_encoder_arch}")

    return {
        "output_dir": str(output_dir),
        "summary_base_channels": int(summary_base_channels),
        "export_base_channels": int(model_base_channels),
        "summary_encoder_arch": summary_encoder_arch,
        "export_encoder_arch": model_encoder_arch,
        "summary_steps": int(summary_config.get("steps", -1)),
    }


def _free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return float(usage.free) / (1024.0**3)


def _tail(path: Path, *, lines: int) -> str:
    text = path.read_text(encoding="utf-8")
    chunks = text.splitlines()
    return "\n".join(chunks[-max(1, int(lines)) :])


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


if __name__ == "__main__":
    main()
