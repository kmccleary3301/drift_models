from __future__ import annotations

import argparse
import json
from pathlib import Path

from drifting_models.utils.experiment_log import (
    append_experiment_log,
    build_entry_from_summary,
    default_run_id,
)


def main() -> None:
    args = _parse_args()
    summary_path = Path(args.summary_json)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("summary_json must contain a JSON object")

    run_id = args.run_id or default_run_id(args.run_id_prefix)
    entry = build_entry_from_summary(
        kind=args.kind,
        run_id=run_id,
        date=args.date,
        summary=summary,
        notes=args.notes,
    )
    append_experiment_log(log_path=Path(args.experiment_log), entry_md=entry)
    print(entry)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a run summary to docs/experiment_log.md")
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--experiment-log", type=str, default="docs/experiment_log.md")
    parser.add_argument("--kind", choices=("end_to_end_pixel", "end_to_end_latent", "alpha_sweep", "last_k", "custom"), required=True)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--run-id-prefix", type=str, default="EXP-AUTO")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--notes", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()

