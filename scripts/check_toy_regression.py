from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_path)
    output_path = Path(args.output_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    deltas = summary.get("deltas_vs_baseline", {})
    finite_checks = summary.get("finite_checks", {})
    all_finite = bool(finite_checks.get("all_finite", False))

    baseline_delta = deltas.get("baseline", {})
    attraction_delta = deltas.get("attraction_only", {})
    attraction_mean_delta = float(attraction_delta.get("mean_distance_to_target", 0.0))
    baseline_mean_delta = float(baseline_delta.get("mean_distance_to_target", 0.0))
    attraction_mode_delta = float(attraction_delta.get("mode_balance_error", 0.0))

    checks = [
        {
            "name": "all_finite_metrics",
            "passed": all_finite,
            "value": all_finite,
        },
        {
            "name": "baseline_zero_delta",
            "passed": abs(baseline_mean_delta) <= args.baseline_zero_tolerance,
            "value": baseline_mean_delta,
            "threshold": args.baseline_zero_tolerance,
        },
        {
            "name": "attraction_only_worse_than_baseline",
            "passed": attraction_mean_delta >= args.min_attraction_only_delta,
            "value": attraction_mean_delta,
            "threshold": args.min_attraction_only_delta,
        },
        {
            "name": "attraction_only_mode_balance_stable",
            "passed": abs(attraction_mode_delta) <= args.max_mode_balance_delta,
            "value": attraction_mode_delta,
            "threshold": args.max_mode_balance_delta,
        },
    ]

    passed = all(check["passed"] for check in checks)
    report = {
        "summary_path": str(summary_path),
        "passed": passed,
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "passed": passed}, indent=2))
    if not passed:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regression gate for toy anti-symmetry ablation summary.")
    parser.add_argument("--summary-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--min-attraction-only-delta", type=float, default=1.0)
    parser.add_argument("--baseline-zero-tolerance", type=float, default=1e-9)
    parser.add_argument("--max-mode-balance-delta", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
