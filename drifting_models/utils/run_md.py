from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_run_md(run_summary: dict[str, Any]) -> str:
    output_root = run_summary.get("output_root", "")
    paths = run_summary.get("paths", {})
    args = run_summary.get("args", {})
    commands = run_summary.get("commands", {})

    lines: list[str] = []
    lines.append("# Run Summary")
    lines.append("")
    lines.append(f"- Output root: `{output_root}`")
    lines.append("")

    if isinstance(paths, dict) and paths:
        lines.append("## Paths")
        lines.append("")
        for key in sorted(paths.keys()):
            value = paths.get(key)
            if value is None:
                continue
            lines.append(f"- {key}: `{value}`")
        lines.append("")

    if isinstance(args, dict) and args:
        lines.append("## Args")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(args, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")

    if isinstance(commands, dict) and commands:
        lines.append("## Commands")
        lines.append("")
        for name in sorted(commands.keys()):
            entry = commands.get(name)
            if not isinstance(entry, dict):
                continue
            argv = entry.get("argv", [])
            returncode = entry.get("returncode", "")
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"- returncode: `{returncode}`")
            if isinstance(argv, list) and argv:
                lines.append("- argv:")
                lines.append("")
                lines.append("```bash")
                lines.append(" ".join(map(str, argv)))
                lines.append("```")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_run_md(path: Path, run_summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_run_md(run_summary), encoding="utf-8")

