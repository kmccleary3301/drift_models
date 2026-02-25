from __future__ import annotations

import datetime as dt
import re
from pathlib import Path


def bar(pct: float, width: int = 24) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(round(width * pct / 100.0))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {pct:5.1f}%"


def parse_stage_table(md: str) -> dict[int, float]:
    # Extract rows like: | 5 | 100% | ... |
    stages: dict[int, float] = {}
    in_table = False
    for line in md.splitlines():
        if line.strip().startswith("| Stage") and "Completion" in line:
            in_table = True
            continue
        if in_table:
            if not line.strip().startswith("|"):
                break
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cols) < 2:
                continue
            if not cols[0].isdigit():
                continue
            stage = int(cols[0])
            m = re.match(r"^(\d+(?:\.\d+)?)%$", cols[1])
            if not m:
                continue
            stages[stage] = float(m.group(1))
    return stages


def parse_weights(md: str) -> dict[int, float]:
    # Extract rows like: | 5 | Latent... | 20 |
    weights: dict[int, float] = {}
    in_table = False
    for line in md.splitlines():
        if line.strip().startswith("| Stage") and "Weight" in line:
            in_table = True
            continue
        if in_table:
            if not line.strip().startswith("|"):
                break
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cols) < 3:
                continue
            if not cols[0].isdigit():
                continue
            stage = int(cols[0])
            try:
                w = float(cols[2])
            except ValueError:
                continue
            weights[stage] = w
    return weights


def weighted_overall(weights: dict[int, float], stages: dict[int, float]) -> float:
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    acc = 0.0
    for s, w in weights.items():
        acc += w * stages.get(s, 0.0)
    return acc / total


def main() -> None:
    repro_path = Path("docs/reproduction_report.md")
    md = repro_path.read_text(encoding="utf-8")
    weights = parse_weights(md)
    stages = parse_stage_table(md)

    today = dt.date.today().isoformat()
    overall = weighted_overall(weights, stages)

    lines: list[str] = []
    lines.append("# Plan Progress Bars")
    lines.append("")
    lines.append(f"- Updated: {today}")
    lines.append("- Source of truth: `docs/reproduction_report.md` stage table")
    lines.append("")
    lines.append(f"## Overall")
    lines.append(f"{bar(overall)}")
    lines.append("")
    lines.append("## By Stage")
    for stage in sorted(weights.keys()):
        w = weights[stage]
        pct = stages.get(stage, 0.0)
        lines.append(f"- Stage {stage} (weight {w:g}): {bar(pct)}")

    Path("docs/PROGRESS_BARS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
