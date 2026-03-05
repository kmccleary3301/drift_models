from __future__ import annotations

import argparse
import json
from pathlib import Path

from drifting_models.utils import is_experimental_run_name, is_stable_run_name, write_json


_DEFAULT_EVAL_GLOBS: tuple[str, ...] = (
    "eval/eval_summary.json",
    "eval/eval_pretrained.json",
    "eval_pretrained.json",
    "claim_bundle/claim_eval/eval_pretrained.json",
    "claim_bundle/claim_eval/eval_pretrained_*.json",
    "claim_bundle/alpha_sweep/alpha_sweep_summary.json",
    "claim_bundle/last_k_eval/last_k_summary.json",
    "alpha_sweep/alpha_sweep_summary.json",
    "last_k_eval/last_k_summary.json",
)

_CORE_FILES: tuple[str, ...] = (
    "RUN.md",
    "env_snapshot.json",
    "codebase_fingerprint.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate canonical run artifact bundle files.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--lane", choices=("stable", "experimental", "any"), default="any")
    parser.add_argument("--eval-glob", action="append", default=None)
    parser.add_argument("--allow-missing-eval-summaries", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    eval_globs = tuple(args.eval_glob) if args.eval_glob else _DEFAULT_EVAL_GLOBS
    report = validate_run_artifacts(
        run_root=Path(args.run_root),
        lane=str(args.lane),
        eval_globs=eval_globs,
        require_eval_summaries=not bool(args.allow_missing_eval_summaries),
    )

    if args.output_json is not None:
        write_json(Path(args.output_json), report)

    print(json.dumps(report, indent=2))
    if not bool(report["pass"]):
        raise SystemExit(1)


def validate_run_artifacts(
    *,
    run_root: Path,
    lane: str,
    eval_globs: tuple[str, ...],
    require_eval_summaries: bool,
) -> dict[str, object]:
    resolved_root = run_root.resolve()
    failures: list[str] = []

    if not resolved_root.exists():
        failures.append(f"Missing run root: {resolved_root}")
    if not resolved_root.is_dir():
        failures.append(f"Run root is not a directory: {resolved_root}")

    lane_check = _check_lane_name(lane=lane, run_name=resolved_root.name)
    if lane_check is not None:
        failures.append(lane_check)

    missing_core: list[str] = []
    present_core: list[str] = []
    for relative in _CORE_FILES:
        target = resolved_root / relative
        if target.exists():
            present_core.append(relative)
        else:
            missing_core.append(relative)
    if missing_core:
        failures.append(f"Missing core files: {', '.join(missing_core)}")

    eval_matches: list[str] = []
    for pattern in eval_globs:
        for path in resolved_root.glob(pattern):
            if path.is_file():
                eval_matches.append(str(path.relative_to(resolved_root)))
    eval_matches = sorted(set(eval_matches))
    if require_eval_summaries and not eval_matches:
        failures.append("Missing eval summaries (no files matched eval globs)")

    return {
        "pass": len(failures) == 0,
        "run_root": str(resolved_root),
        "lane": lane,
        "core_files": {"required": list(_CORE_FILES), "present": present_core, "missing": missing_core},
        "eval_globs": list(eval_globs),
        "eval_matches": eval_matches,
        "failures": failures,
    }


def _check_lane_name(*, lane: str, run_name: str) -> str | None:
    if lane == "any":
        return None
    if lane == "stable" and not is_stable_run_name(run_name):
        return f"Run root name must match stable_<timestamp>, got: {run_name}"
    if lane == "experimental" and not is_experimental_run_name(run_name):
        return f"Run root name must match exp_<name>_<timestamp>, got: {run_name}"
    return None


if __name__ == "__main__":
    main()
