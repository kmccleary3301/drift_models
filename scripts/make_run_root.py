from __future__ import annotations

import argparse
import json
from pathlib import Path

from drifting_models.utils import build_experimental_run_root, build_stable_run_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Create canonical run root paths for stable or experimental lanes.")
    parser.add_argument("--lane", choices=("stable", "experimental"), required=True)
    parser.add_argument("--base-dir", type=str, default="outputs/imagenet")
    parser.add_argument("--name", type=str, default=None, help="Required for --lane experimental")
    parser.add_argument("--timestamp", type=str, default=None, help="Optional explicit timestamp YYYYMMDD_HHMMSS")
    parser.add_argument("--mkdir", action="store_true", help="Create the output directory on disk")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    if args.lane == "stable":
        run_root = build_stable_run_root(base_dir=base_dir, timestamp=args.timestamp)
    else:
        if args.name is None or not args.name.strip():
            raise ValueError("--name is required for --lane experimental")
        run_root = build_experimental_run_root(base_dir=base_dir, name=args.name, timestamp=args.timestamp)

    if args.mkdir:
        run_root.mkdir(parents=True, exist_ok=False)

    print(json.dumps({"lane": args.lane, "run_root": str(run_root)}, indent=2))


if __name__ == "__main__":
    main()
