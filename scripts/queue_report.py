from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QueueReport:
    ok: bool
    summary_path: str
    kind: str
    output_path: str
    warnings: list[str]


def main() -> None:
    args = _parse_args()
    summary_path = Path(args.summary_path).expanduser().resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    kind = _infer_kind(payload)

    warnings: list[str] = []
    queue_warmup = payload.get("queue_warmup_report")
    if not isinstance(queue_warmup, dict):
        raise ValueError("summary has no queue_warmup_report; nothing to report")

    counts = queue_warmup.get("counts") if isinstance(queue_warmup.get("counts"), list) else None
    report_level = payload.get("queue_report_level")
    real_sanity = payload.get("real_provider_sanity_report") if isinstance(payload.get("real_provider_sanity_report"), dict) else None

    md = []
    md.append(f"# Queue Report ({kind})")
    md.append("")
    md.append(f"- Summary: `{summary_path}`")
    if isinstance(report_level, str):
        md.append(f"- `queue_report_level`: `{report_level}`")
    md.append(f"- Warmup mode: `{queue_warmup.get('warmup_mode')}`")
    md.append(f"- Samples pushed: `{queue_warmup.get('samples_pushed')}`")
    md.append(f"- Global count: `{queue_warmup.get('global_count')}`")
    md.append(f"- Covered classes: `{queue_warmup.get('covered_classes')}`")
    md.append(f"- Min class count (nonzero): `{queue_warmup.get('min_class_count_nonzero')}`")
    md.append(f"- Max class count: `{queue_warmup.get('max_class_count')}`")
    md.append(f"- Mean class count: `{queue_warmup.get('mean_class_count')}`")
    if isinstance(queue_warmup.get("count_quantiles_nonzero"), dict):
        q = queue_warmup["count_quantiles_nonzero"]
        md.append(f"- Nonzero count quantiles: p10={q.get('p10')} p50={q.get('p50')} p90={q.get('p90')}")

    if real_sanity is not None:
        md.append("")
        md.append("## Real Provider Sanity")
        md.append(f"- Sample batches: `{real_sanity.get('sample_batches')}`")
        md.append(f"- Samples: `{real_sanity.get('samples')}`")
        md.append(f"- Labels: min={real_sanity.get('label_min')} max={real_sanity.get('label_max')}")
        md.append(f"- Covered classes: `{real_sanity.get('covered_classes')}`")
        md.append(f"- Min class count (nonzero): `{real_sanity.get('min_class_count_nonzero')}`")
        md.append(f"- Max class count: `{real_sanity.get('max_class_count')}`")
        md.append(f"- Mean class count: `{real_sanity.get('mean_class_count')}`")

    if counts is None:
        warnings.append("queue_warmup_report.counts missing (run with --queue-report-level full).")
    else:
        md.append("")
        md.append("## Coverage Details")
        counts_int = [int(x) for x in counts]
        missing = [i for i, c in enumerate(counts_int) if c == 0]
        if missing:
            md.append(f"- Missing classes: `{len(missing)}`")
            md.append(f"- First missing (up to {args.max_classes_listed}): `{missing[: int(args.max_classes_listed)]}`")
        else:
            md.append("- Missing classes: `0`")
        md.append("")
        md.append("## Histogram (Class Counts)")
        md.extend(_ascii_histogram(counts_int, width=int(args.hist_width)))
        md.append("")
        md.append("## Lowest Nonzero Classes")
        lows = sorted([(c, i) for i, c in enumerate(counts_int) if c > 0])[: int(args.max_classes_listed)]
        md.append("```")
        for c, i in lows:
            md.append(f"class={i:4d} count={c}")
        md.append("```")

    # Per-step queue stats (if present)
    logs = payload.get("logs")
    if isinstance(logs, list) and logs and all(isinstance(x, dict) for x in logs):
        series = [(x.get("step"), x.get("queue_global_count"), x.get("queue_covered_classes")) for x in logs]
        series = [(a, b, c) for (a, b, c) in series if isinstance(a, (int, float)) and isinstance(b, (int, float)) and isinstance(c, (int, float))]
        if series:
            md.append("")
            md.append("## Per-Step Queue Snapshot (Logged Steps)")
            md.append("```")
            md.append("step queue_global_count queue_covered_classes")
            for step, gcount, covered in series[- int(args.max_steps_listed) :]:
                md.append(f"{int(step):>5d} {int(gcount):>17d} {int(covered):>19d}")
            md.append("```")

    out_path = Path(args.output_path).expanduser().resolve() if args.output_path else summary_path.parent / "queue_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    report = QueueReport(
        ok=True,
        summary_path=str(summary_path),
        kind=kind,
        output_path=str(out_path),
        warnings=warnings,
    )
    print(json.dumps(report.__dict__, indent=2))


def _infer_kind(payload: dict[str, Any]) -> str:
    if "train_config" in payload:
        if "latent_feature_decode_mode" in str(payload.get("train_config", {})):
            return "latent"
        return "train"
    if "model_config" in payload:
        return "latent_or_pixel"
    return "unknown"


def _ascii_histogram(counts: list[int], *, width: int) -> list[str]:
    # Buckets chosen to be informative for both tiny and large queue caps.
    buckets = [
        (0, 0),
        (1, 1),
        (2, 3),
        (4, 7),
        (8, 15),
        (16, 31),
        (32, 63),
        (64, 127),
        (128, 255),
        (256, 511),
        (512, 1023),
        (1024, math.inf),
    ]
    bucket_counts = []
    for lo, hi in buckets:
        n = 0
        for c in counts:
            if c < lo:
                continue
            if hi is math.inf:
                if c >= lo:
                    n += 1
            elif lo <= c <= hi:
                n += 1
        bucket_counts.append(((lo, hi), n))
    max_n = max((n for (_, n) in bucket_counts), default=1)
    lines = []
    for (lo, hi), n in bucket_counts:
        bar = "#" * int(round((n / max_n) * max(1, width)))
        label = f"{lo:>4d}+ " if hi is math.inf else f"{lo:>4d}-{int(hi):<4d}"
        lines.append(f"{label} | {bar} ({n})")
    return lines


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a compact markdown report from a queue-enabled training summary JSON.")
    p.add_argument("--summary-path", type=str, required=True)
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--max-classes-listed", type=int, default=20)
    p.add_argument("--max-steps-listed", type=int, default=30)
    p.add_argument("--hist-width", type=int, default=40)
    return p.parse_args()


if __name__ == "__main__":
    main()

