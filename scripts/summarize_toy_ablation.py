from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))

    results = payload.get("results", [])
    by_name = {str(entry.get("ablation")): entry for entry in results}
    if args.baseline not in by_name:
        raise ValueError(f"Baseline ablation '{args.baseline}' not found in {input_path}")

    baseline_metrics = by_name[args.baseline].get("final_metrics", {})
    if not isinstance(baseline_metrics, dict):
        raise ValueError("Baseline final_metrics missing or invalid")

    finite_report = build_finite_report(results)
    deltas = build_delta_table(by_name=by_name, baseline=args.baseline, baseline_metrics=baseline_metrics)

    summary = {
        "source_results_path": str(input_path),
        "baseline": args.baseline,
        "device": payload.get("device"),
        "seed": payload.get("config", {}).get("seed"),
        "finite_checks": finite_report,
        "deltas_vs_baseline": deltas,
        "ranking_by_mean_distance": payload.get("ranking_by_mean_distance", []),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "all_finite": finite_report["all_finite"], "ablation_count": len(by_name)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize toy ablation results with finite checks and deltas.")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--baseline", type=str, default="baseline")
    return parser.parse_args()


def build_finite_report(results: list[object]) -> dict[str, object]:
    non_finite_entries: list[dict[str, object]] = []
    for entry in results:
        if not isinstance(entry, dict):
            continue
        ablation = str(entry.get("ablation"))
        final_metrics = entry.get("final_metrics", {})
        if isinstance(final_metrics, dict):
            for key, value in final_metrics.items():
                if not is_finite_number(value):
                    non_finite_entries.append({"ablation": ablation, "location": "final_metrics", "key": str(key), "value": value})
        history = entry.get("history", [])
        if isinstance(history, list):
            for index, row in enumerate(history):
                if not isinstance(row, dict):
                    continue
                for key, value in row.items():
                    if not is_finite_number(value):
                        non_finite_entries.append(
                            {
                                "ablation": ablation,
                                "location": "history",
                                "index": index,
                                "key": str(key),
                                "value": value,
                            }
                        )
    return {"all_finite": len(non_finite_entries) == 0, "non_finite_entries": non_finite_entries}


def build_delta_table(
    *,
    by_name: dict[str, dict[str, object]],
    baseline: str,
    baseline_metrics: dict[str, object],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for ablation_name, entry in sorted(by_name.items()):
        metrics = entry.get("final_metrics", {})
        if not isinstance(metrics, dict):
            continue
        per_metric_delta: dict[str, float] = {}
        for key, baseline_value in baseline_metrics.items():
            if key not in metrics:
                continue
            if not (is_finite_number(metrics[key]) and is_finite_number(baseline_value)):
                continue
            per_metric_delta[str(key)] = float(metrics[key]) - float(baseline_value)
        output[ablation_name] = per_metric_delta
    return output


def is_finite_number(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


if __name__ == "__main__":
    main()
