from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_series(logs: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for entry in logs:
        value = entry.get(key)
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "min": float(min(values)),
        "mean": float(sum(values) / len(values)),
        "max": float(max(values)),
        "n": float(len(values)),
    }


def _render_md(payload: dict[str, Any], source_path: Path) -> str:
    lines: list[str] = []
    lines.append("# Queue Audit")
    lines.append("")
    lines.append(f"- Source summary: `{source_path}`")
    lines.append("")

    warmup = payload.get("queue_warmup_report")
    if isinstance(warmup, dict):
        lines.append("## Warmup")
        lines.append("")
        rendered = dict(warmup)
        counts = rendered.get("counts")
        if isinstance(counts, list) and len(counts) > 100:
            numeric = [float(v) for v in counts if isinstance(v, (int, float))]
            rendered.pop("counts", None)
            rendered["counts_summary"] = _summary_stats(numeric)
            rendered["counts_head"] = numeric[:20]
        lines.append("```json")
        lines.append(json.dumps(rendered, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")

    totals = payload.get("queue_underflow_totals")
    if isinstance(totals, dict):
        lines.append("## Underflow Totals")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(totals, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")

    logs = payload.get("logs", [])
    if not isinstance(logs, list):
        logs = []
    logs = [entry for entry in logs if isinstance(entry, dict)]

    keys = [
        "queue_global_count",
        "queue_covered_classes",
        "queue_underflow_backfilled",
        "queue_underflow_missing_labels",
    ]
    series_stats: dict[str, dict[str, float]] = {}
    for key in keys:
        series = _maybe_series(logs, key)
        stats = _summary_stats(series)
        if stats:
            series_stats[key] = stats

    if series_stats:
        lines.append("## Logged Time Series (summary)")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(series_stats, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")

    if logs:
        lines.append("## Last Logged Step Snapshot")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(logs[-1], indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Render a simple queue audit markdown from a training summary JSON.")
    p.add_argument("--summary-path", type=str, required=True, help="Path to latent_summary.json or pixel_summary.json")
    p.add_argument("--output-md", type=str, required=True)
    args = p.parse_args()

    summary_path = Path(args.summary_path)
    payload = _read_json(summary_path)
    md = _render_md(payload, summary_path)

    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
