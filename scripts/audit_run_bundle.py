from __future__ import annotations

import argparse
import json
from pathlib import Path


def _required_paths(run_kind: str) -> list[str]:
    common = [
        "env_snapshot.json",
        "codebase_fingerprint.json",
        "env_fingerprint.json",
        "RUN.md",
    ]
    if run_kind == "latent":
        return common + ["latent_summary.json"]
    if run_kind == "pixel":
        return common + ["pixel_summary.json"]
    if run_kind == "mae":
        return common + ["mae_summary.json", "mae_encoder.pt"]
    raise ValueError(f"unsupported run kind: {run_kind}")


def _infer_run_kind(run_dir: Path) -> str:
    if (run_dir / "latent_summary.json").exists():
        return "latent"
    if (run_dir / "pixel_summary.json").exists():
        return "pixel"
    if (run_dir / "mae_summary.json").exists():
        return "mae"
    return "latent"


def _audit_run(run_dir: Path, run_kind: str) -> dict[str, object]:
    required = _required_paths(run_kind)
    present = {path: (run_dir / path).exists() for path in required}
    optional = {
        "checkpoint.pt": (run_dir / "checkpoint.pt").exists(),
        "checkpoints_dir": (run_dir / "checkpoints").exists(),
    }
    return {
        "run_dir": str(run_dir),
        "run_kind": run_kind,
        "required": present,
        "optional": optional,
        "pass": all(present.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit required run artifact bundles for latent/pixel/mae runs.")
    parser.add_argument("--run-dir", action="append", required=True, help="Run directory path. May be passed multiple times.")
    parser.add_argument("--run-kind", choices=("auto", "latent", "pixel", "mae"), default="auto")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-md", type=str, default=None)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for raw in args.run_dir:
        run_dir = Path(raw)
        kind = _infer_run_kind(run_dir) if args.run_kind == "auto" else str(args.run_kind)
        rows.append(_audit_run(run_dir=run_dir, run_kind=kind))

    payload = {
        "runs": rows,
        "pass_count": sum(1 for row in rows if bool(row["pass"])),
        "total": len(rows),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.output_md is not None:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Run Bundle Audit",
            "",
            "| run_dir | kind | pass | missing_required |",
            "| --- | --- | ---: | --- |",
        ]
        for row in rows:
            required = row["required"]
            missing = [key for key, exists in required.items() if not exists]
            lines.append(
                f"| `{row['run_dir']}` | `{row['run_kind']}` | `{str(bool(row['pass'])).lower()}` | "
                f"`{', '.join(missing) if missing else '-'}` |"
            )
        lines.append("")
        lines.append(f"- Pass: {payload['pass_count']}/{payload['total']}")
        output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
