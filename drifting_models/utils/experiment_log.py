from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any


def default_run_id(prefix: str) -> str:
    now = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{now}"


def append_experiment_log(
    *,
    log_path: Path,
    entry_md: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        existing = log_path.read_text(encoding="utf-8").rstrip() + "\n\n"
    else:
        existing = ""
    log_path.write_text(existing + entry_md.rstrip() + "\n", encoding="utf-8")


def build_entry_from_summary(
    *,
    kind: str,
    run_id: str,
    date: str | None,
    summary: dict[str, Any],
    notes: str | None,
) -> str:
    resolved_date = date or _dt.date.today().isoformat()
    args = summary.get("args", {}) if isinstance(summary.get("args", {}), dict) else {}
    paths = summary.get("paths", {}) if isinstance(summary.get("paths", {}), dict) else {}

    lines: list[str] = []
    lines.append(f"- **Run ID**: {run_id}")
    lines.append(f"- **Date**: {resolved_date}")
    lines.append("- **Code hash**: N/A (not a git repository)")

    if kind in {"end_to_end_pixel", "end_to_end_latent"}:
        train_config = args.get("train_config")
        lines.append(f"- **Config path + hash**: {train_config if train_config else 'scripted runner args'}")
        lines.append(f"- **Hardware**: {args.get('device', 'unknown')}")
        lines.append(f"- **Seed(s)**: {args.get('seed', 'unknown')}")
        lines.append(f"- **Objective**: Validate {kind.replace('_', ' ')} runner (train -> sample -> eval)")
        lines.append(
            "- **Key toggles**: "
            + json.dumps(
                {
                    "train_steps": args.get("train_steps"),
                    "sample_count": args.get("sample_count"),
                    "inception_weights": args.get("inception_weights"),
                },
                sort_keys=True,
            )
        )
        artifacts = [
            paths.get("checkpoint_path"),
            paths.get("eval_summary_path"),
            str(Path(summary.get("output_root", "")) / "run_summary.json"),
        ]
        lines.append("- **Artifacts**:")
        for item in artifacts:
            if item:
                lines.append(f"  - `{item}`")
        lines.append("- **Outcome**: success")
        if notes:
            lines.append(f"- **Notes / next action**: {notes}")
        else:
            lines.append("- **Notes / next action**: extend horizons and switch to pretrained Inception for comparable metrics")
        return "\n".join(lines)

    if kind == "alpha_sweep":
        lines.append(f"- **Config path + hash**: {args.get('config', 'scripted alpha sweep args')}")
        lines.append(f"- **Hardware**: {args.get('device', 'unknown')}")
        lines.append(f"- **Seed(s)**: {args.get('seed', 'unknown')}")
        lines.append("- **Objective**: Sweep alpha values and record FID/IS per alpha")
        lines.append(
            "- **Key toggles**: "
            + json.dumps(
                {
                    "mode": args.get("mode"),
                    "alphas": args.get("alphas"),
                    "n_samples": args.get("n_samples"),
                    "inception_weights": args.get("inception_weights"),
                    "reference_cache": bool(args.get("reference_cache", False)),
                },
                sort_keys=True,
            )
        )
        lines.append("- **Artifacts**:")
        out = summary.get("output_root", "")
        if out:
            lines.append(f"  - `{out}/alpha_sweep_summary.json`")
            lines.append(f"  - `{out}/alpha_sweep_summary.md`")
        lines.append("- **Outcome**: success")
        lines.append(f"- **Notes / next action**: {notes}" if notes else "- **Notes / next action**: use pretrained Inception for comparable sweeps")
        return "\n".join(lines)

    if kind == "last_k":
        lines.append(f"- **Config path + hash**: {args.get('config', 'scripted last-K args')}")
        lines.append(f"- **Hardware**: {args.get('device', 'unknown')}")
        lines.append(f"- **Seed(s)**: {args.get('seed', 'unknown')}")
        lines.append("- **Objective**: Evaluate last-K step checkpoints with consistent sampling/eval settings")
        lines.append(
            "- **Key toggles**: "
            + json.dumps(
                {
                    "mode": args.get("mode"),
                    "k": args.get("k"),
                    "n_samples": args.get("n_samples"),
                    "inception_weights": args.get("inception_weights"),
                    "reference_cache": bool(args.get("reference_cache", False)),
                },
                sort_keys=True,
            )
        )
        lines.append("- **Artifacts**:")
        out = summary.get("output_root", "")
        if out:
            lines.append(f"  - `{out}/last_k_summary.json`")
            lines.append(f"  - `{out}/last_k_summary.md`")
        lines.append("- **Outcome**: success")
        lines.append(f"- **Notes / next action**: {notes}" if notes else "- **Notes / next action**: increase K and sample counts for trend stability")
        return "\n".join(lines)

    lines.append(f"- **Config path + hash**: {args.get('config', 'N/A')}")
    lines.append(f"- **Hardware**: {args.get('device', 'unknown')}")
    lines.append(f"- **Seed(s)**: {args.get('seed', 'unknown')}")
    lines.append(f"- **Objective**: {kind}")
    lines.append("- **Outcome**: success")
    if notes:
        lines.append(f"- **Notes / next action**: {notes}")
    return "\n".join(lines)

