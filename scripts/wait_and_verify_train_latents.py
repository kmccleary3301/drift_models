from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    manifest_path = repo_root / args.manifest_relpath
    summary_path = repo_root / args.export_summary_relpath
    output_path = repo_root / args.output_relpath

    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        print(f"[{ts}] {msg}", flush=True)

    def free_gb() -> float:
        return float(shutil.disk_usage(args.drive_mount).free) / (1024.0**3)

    log(f"waiting_for manifest={manifest_path} summary={summary_path}")
    while True:
        if manifest_path.exists() and summary_path.exists():
            break
        log(f"still_waiting free_gb={free_gb():.1f}")
        time.sleep(float(args.poll_seconds))

    log("files_present; running verifier")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    verify_cmd = [
        sys.executable,
        "scripts/verify_tensor_shards_manifest.py",
        "--manifest-path",
        str(manifest_path),
        "--mode",
        "quick",
        "--max-shards",
        str(int(args.max_shards)),
        "--check-sha256",
        "--output-path",
        str(output_path),
    ]
    result = subprocess.run(verify_cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        log("verify_failed")
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(int(result.returncode))
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    log(f"verify_ok total_items_manifest={payload.get('total_items_manifest')} label_max={payload.get('label_max')}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wait for ImageNet train latents manifest, then run quick verification.")
    p.add_argument("--drive-mount", type=str, default="/mnt/drive_4")
    p.add_argument("--poll-seconds", type=float, default=600.0)
    p.add_argument(
        "--manifest-relpath",
        type=str,
        default="outputs/datasets/imagenet1k_train_sdvae_latents_shards/manifest.json",
    )
    p.add_argument(
        "--export-summary-relpath",
        type=str,
        default="outputs/datasets/imagenet1k_train_sdvae_latents_shards/export_summary.json",
    )
    p.add_argument(
        "--output-relpath",
        type=str,
        default="outputs/datasets/imagenet1k_train_sdvae_latents_shards/verify_quick.json",
    )
    p.add_argument("--max-shards", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    main()
