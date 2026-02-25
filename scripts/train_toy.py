from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from drifting_models.train import (
    AblationConfig,
    default_ablation_suite,
    parse_simple_yaml_config,
    run_toy_suite,
)
from drifting_models.utils import (
    add_device_argument,
    codebase_fingerprint,
    environment_fingerprint,
    environment_snapshot,
    resolve_device,
    write_json,
)
from drifting_models.utils.run_md import write_run_md


def main() -> None:
    args = _parse_args()
    config = parse_simple_yaml_config(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    write_json(output_dir / "env_snapshot.json", environment_snapshot(paths=[output_dir]))
    write_json(output_dir / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))

    if args.ablation == "all":
        ablations = default_ablation_suite()
    else:
        ablations = [_named_ablation(args.ablation)]

    device = resolve_device(args.device)

    summary = run_toy_suite(
        config=config,
        ablations=ablations,
        output_dir=output_dir,
        device=device,
    )
    ranking = summary["ranking_by_mean_distance"]
    run_summary = {
        "output_root": str(output_dir),
        "paths": {
            "toy_results_json": str(output_dir / "toy_results.json"),
            "toy_ablation_table_md": str(output_dir / "toy_ablation_table.md"),
            "env_snapshot_json": str(output_dir / "env_snapshot.json"),
            "codebase_fingerprint_json": str(output_dir / "codebase_fingerprint.json"),
        },
        "args": {
            "config": str(args.config),
            "output_dir": str(args.output_dir),
            "ablation": args.ablation,
            "device": args.device,
        },
        "commands": {"train_toy": {"argv": list(sys.argv), "returncode": 0}},
        "env_fingerprint": environment_fingerprint(),
        "ranking_by_mean_distance": ranking,
    }
    write_json(output_dir / "run_summary.json", run_summary)
    write_run_md(output_dir / "RUN.md", run_summary)
    print(json.dumps({"output_dir": str(output_dir), "ranking_by_mean_distance": ranking}, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Drifting Models toy 2D experiments")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/toy/base.yaml"),
        help="Path to simple key:value yaml config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/toy"),
        help="Output directory for JSON artifacts",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="all",
        help="Ablation name or 'all'",
    )
    add_device_argument(parser, default="auto")
    return parser.parse_args()


def _named_ablation(name: str) -> AblationConfig:
    table = {entry.name: entry for entry in default_ablation_suite()}
    if name not in table:
        available = ", ".join(sorted(table))
        raise ValueError(f"Unknown ablation '{name}'. Available: {available}, all")
    return table[name]


if __name__ == "__main__":
    main()
