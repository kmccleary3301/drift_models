from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    summary_path = (repo_root / args.summary_relpath).resolve()
    out_path = (repo_root / args.output_relpath).resolve()

    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        print(f"[{ts}] {msg}", flush=True)

    log(f"waiting_for summary={summary_path}")
    while not summary_path.exists():
        time.sleep(float(args.poll_seconds))
    log("summary_present; generating queue report")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/queue_report.py",
        "--summary-path",
        str(summary_path),
        "--output-path",
        str(out_path),
        "--max-classes-listed",
        str(int(args.max_classes_listed)),
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise SystemExit(int(result.returncode))
    log(f"done output={out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wait for a training summary then write queue_report.md next to it.")
    p.add_argument("--poll-seconds", type=float, default=60.0)
    p.add_argument("--summary-relpath", type=str, required=True)
    p.add_argument("--output-relpath", type=str, required=True)
    p.add_argument("--max-classes-listed", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    main()

