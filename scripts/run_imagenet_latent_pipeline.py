from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from drifting_models.utils import add_device_argument, codebase_fingerprint, environment_fingerprint, environment_snapshot, write_json
from drifting_models.utils.run_md import write_run_md


@dataclass(frozen=True)
class Cmd:
    argv: list[str]
    cwd: Path


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = repo_root / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_path = logs_dir / "imagenet_latent_pipeline_state.json"
    write_json(logs_dir / "env_snapshot.json", environment_snapshot(paths=[logs_dir]))
    write_json(logs_dir / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))
    write_json(logs_dir / "env_fingerprint.json", environment_fingerprint())
    write_run_md(
        logs_dir / "RUN.md",
        {
            "output_root": str(logs_dir),
            "args": vars(args),
            "paths": {
                "state_path": str(state_path),
                "env_snapshot_json": str(logs_dir / "env_snapshot.json"),
                "codebase_fingerprint_json": str(logs_dir / "codebase_fingerprint.json"),
                "env_fingerprint_json": str(logs_dir / "env_fingerprint.json"),
            },
            "commands": {"invocation": {"argv": [sys.executable, *sys.argv]}},
        },
    )

    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        print(f"[{ts}] {msg}", flush=True)

    def write_state(payload: dict[str, object]) -> None:
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(state_path)

    def drive_free_gb(path: Path) -> float:
        usage = shutil.disk_usage(path)
        return float(usage.free) / (1024.0**3)

    def guard_disk(stage: str) -> None:
        free = drive_free_gb(Path(args.drive_mount))
        if free < float(args.min_free_gb):
            raise RuntimeError(f"Refusing to proceed at stage={stage}: free space {free:.1f}GB < min_free_gb={args.min_free_gb}")
        log(f"disk_free_gb={free:.1f} stage={stage}")

    def train_extract_status(train_root: Path) -> dict[str, int]:
        if not train_root.exists():
            return {"wnid_dirs": 0, "synset_tars": 0}
        wnid_dirs = 0
        synset_tars = 0
        for p in train_root.iterdir():
            if p.is_dir() and p.name.startswith("n"):
                wnid_dirs += 1
            elif p.is_file() and p.suffix == ".tar":
                synset_tars += 1
        return {"wnid_dirs": int(wnid_dirs), "synset_tars": int(synset_tars)}

    def run(cmd: Cmd, *, name: str) -> None:
        log(f"run[{name}]: {' '.join(map(str, cmd.argv))}")
        result = subprocess.run(cmd.argv, cwd=cmd.cwd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed ({name}) rc={result.returncode}")

    # Paths
    output_root = repo_root / "outputs" / "datasets" / "imagenet1k_raw"
    train_root = output_root / "train"
    train_manifest = repo_root / "outputs" / "datasets" / "imagenet1k_train_sdvae_latents_shards" / "manifest.json"
    mae_encoder = repo_root / "outputs" / "imagenet" / "mae_variant_a_w64" / "mae_encoder.pt"
    latent_ckpt = repo_root / "outputs" / "imagenet" / "latent_smoke_mae" / "checkpoint.pt"
    latent_summary = repo_root / "outputs" / "imagenet" / "latent_smoke_mae" / "latent_summary.json"
    latent_queue_report = repo_root / "outputs" / "imagenet" / "latent_smoke_mae" / "queue_report.md"

    # 0) Disk guard
    guard_disk("start")

    # 1) Ensure train extraction complete (resume-friendly)
    while True:
        status = train_extract_status(train_root)
        write_state({"phase": "train_extract", "status": status, "disk_free_gb": drive_free_gb(Path(args.drive_mount))})
        log(f"train_extract_status synset_tars={status['synset_tars']} wnid_dirs={status['wnid_dirs']}")
        if status["synset_tars"] == 0 and status["wnid_dirs"] >= 1000:
            break
        # If we still have synset tarballs, expand them via the existing script in resume mode.
        run(
            Cmd(
                argv=[
                    sys.executable,
                    "scripts/prepare_imagenet1k_from_archives.py",
                    "--output-root",
                    str(output_root),
                    "--train-archive-path",
                    str(repo_root / args.train_archive_path),
                    "--val-archive-path",
                    str(repo_root / args.val_archive_path),
                    "--devkit-archive-path",
                    str(repo_root / args.devkit_archive_path),
                    "--split",
                    "train",
                    "--allow-existing",
                    "--skip-md5",
                ],
                cwd=repo_root,
            ),
            name="prepare_imagenet_train_resume",
        )
        time.sleep(float(args.poll_seconds))

    # 2) Export train latents (if missing)
    if not train_manifest.exists():
        guard_disk("export_train_latents")
        out_dir = repo_root / "outputs" / "datasets" / "imagenet1k_train_sdvae_latents_shards"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_state({"phase": "export_train_latents", "disk_free_gb": drive_free_gb(Path(args.drive_mount))})
        run(
            Cmd(
                argv=[
                    sys.executable,
                    "scripts/export_sd_vae_latents_tensor_file.py",
                    "--imagefolder-root",
                    str(train_root),
                    "--output-shards-dir",
                    str(out_dir),
                    "--shard-size",
                    str(int(args.latent_shard_size)),
                    "--batch-size",
                    str(int(args.latent_batch_size)),
                    "--num-workers",
                    str(int(args.latent_num_workers)),
                    "--device",
                    str(args.device),
                    "--image-size",
                    "256",
                    "--save-dtype",
                    "fp16",
                    "--latent-sampling",
                    "mean",
                    "--summary-json-path",
                    str(out_dir / "export_summary.json"),
                ],
                cwd=repo_root,
            ),
            name="export_train_latents",
        )
        guard_disk("verify_train_latents_manifest")
        run(
            Cmd(
                argv=[
                    sys.executable,
                    "scripts/verify_tensor_shards_manifest.py",
                    "--manifest-path",
                    str(out_dir / "manifest.json"),
                    "--mode",
                    "quick",
                    "--max-shards",
                    "2",
                    "--check-sha256",
                    "--output-path",
                    str(out_dir / "verify_quick.json"),
                ],
                cwd=repo_root,
            ),
            name="verify_train_latents",
        )

    # 3) MAE smoke (if missing)
    if not mae_encoder.exists():
        guard_disk("mae_smoke")
        write_state({"phase": "mae_smoke", "disk_free_gb": drive_free_gb(Path(args.drive_mount))})
        run(
            Cmd(
                argv=[
                    sys.executable,
                    "scripts/train_mae.py",
                    "--config",
                    "configs/mae/imagenet1k_sdvae_latents_shards_smoke.yaml",
                    "--output-dir",
                    "outputs/imagenet/mae_variant_a_w64",
                    "--real-tensor-shards-manifest-path",
                    str(train_manifest),
                ],
                cwd=repo_root,
            ),
            name="mae_smoke",
        )

    # 4) Latent drifting smoke (if missing)
    if not latent_ckpt.exists():
        guard_disk("latent_smoke_train")
        write_state({"phase": "latent_smoke_train", "disk_free_gb": drive_free_gb(Path(args.drive_mount))})
        (repo_root / "outputs" / "imagenet" / "latent_smoke_mae").mkdir(parents=True, exist_ok=True)
        run(
            Cmd(
                argv=[
                    sys.executable,
                    "scripts/train_latent.py",
                    "--config",
                    "configs/latent/imagenet1k_sdvae_latents_queue_smoke_mae.yaml",
                    "--output-dir",
                    "outputs/imagenet/latent_smoke_mae",
                    "--checkpoint-path",
                    str(latent_ckpt),
                ],
                cwd=repo_root,
            ),
            name="latent_smoke_train",
        )
        if latent_summary.exists():
            run(
                Cmd(
                    argv=[
                        sys.executable,
                        "scripts/queue_report.py",
                        "--summary-path",
                        str(latent_summary),
                        "--output-path",
                        str(latent_queue_report),
                    ],
                    cwd=repo_root,
                ),
                name="latent_smoke_queue_report",
            )
        else:
            log(f"warn: missing latent summary; skipping queue report: {latent_summary}")

    # 5) Sample + eval (idempotent by run_id)
    guard_disk("sample_eval")
    run_id = time.strftime("%Y-%m-%d_%H%M%S")
    samples_root = repo_root / "outputs" / "imagenet" / f"latent_smoke_mae_samples_{run_id}"
    eval_root = repo_root / "outputs" / "imagenet" / f"latent_smoke_mae_eval_{run_id}"
    samples_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    write_state({"phase": "sample_eval", "run_id": run_id, "disk_free_gb": drive_free_gb(Path(args.drive_mount))})

    sample_argv = [
        sys.executable,
        "scripts/sample_latent.py",
        "--checkpoint-path",
        str(latent_ckpt),
        "--output-root",
        str(samples_root),
        "--n-samples",
        str(int(args.eval_sample_count)),
        "--batch-size",
        str(int(args.eval_sample_batch_size)),
        "--alpha",
        "1.0",
        "--write-imagefolder",
        "--decode-mode",
        "sd_vae",
        "--sd-vae-model-id",
        str(args.sd_vae_model_id),
        "--decode-image-size",
        "256",
        "--postprocess-mode",
        "clamp_0_1",
        "--image-format",
        "jpg",
    ]
    if args.sd_vae_subfolder is not None:
        sample_argv.extend(["--sd-vae-subfolder", str(args.sd_vae_subfolder)])
    if args.sd_vae_revision is not None:
        sample_argv.extend(["--sd-vae-revision", str(args.sd_vae_revision)])
    run(Cmd(argv=sample_argv, cwd=repo_root), name="sample_sdvae_decode")
    run(
        Cmd(
            argv=[
                sys.executable,
                "scripts/eval_fid_is.py",
                "--reference-source",
                "imagefolder",
                "--reference-imagefolder-root",
                str(repo_root / "outputs" / "datasets" / "imagenet1k_raw" / "val"),
                "--generated-source",
                "imagefolder",
                "--generated-imagefolder-root",
                str(samples_root / "images"),
                "--load-reference-stats",
                str(repo_root / "outputs" / "datasets" / "imagenet1k_val_reference_stats_pretrained.pt"),
                "--inception-weights",
                "pretrained",
                "--output-path",
                str(eval_root / "eval_pretrained.json"),
            ],
            cwd=repo_root,
        ),
        name="eval_fid_is",
    )
    log(f"done run_id={run_id} samples_root={samples_root} eval_root={eval_root}")
    write_state({"phase": "done", "run_id": run_id, "samples_root": str(samples_root), "eval_root": str(eval_root)})


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Idempotent ImageNet latent protocol pipeline runner (resume-friendly).")
    p.add_argument("--drive-mount", type=str, default="/mnt/drive_4")
    p.add_argument("--min-free-gb", type=float, default=120.0)
    p.add_argument("--poll-seconds", type=float, default=30.0)
    add_device_argument(p, default="auto")

    p.add_argument("--train-archive-path", type=str, default="data/ILSVRC2012_img_train.tar.1")
    p.add_argument("--val-archive-path", type=str, default="data/ILSVRC2012_img_val.tar")
    p.add_argument("--devkit-archive-path", type=str, default="data/ILSVRC2012_devkit_t12.tar.gz")

    p.add_argument("--latent-shard-size", type=int, default=10000)
    p.add_argument("--latent-batch-size", type=int, default=16)
    p.add_argument("--latent-num-workers", type=int, default=8)

    p.add_argument("--eval-sample-count", type=int, default=5000)
    p.add_argument("--eval-sample-batch-size", type=int, default=64)
    # SD-VAE decode parameters for sample_latent.py
    p.add_argument("--sd-vae-model-id", type=str, default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--sd-vae-subfolder", type=str, default=None)
    p.add_argument("--sd-vae-revision", type=str, default="31f26fdeee1355a5c34592e401dd41e45d25a493")
    return p.parse_args()


if __name__ == "__main__":
    main()
