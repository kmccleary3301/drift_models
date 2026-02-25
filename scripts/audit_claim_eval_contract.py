from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_eval(payload: dict[str, object]) -> dict[str, str]:
    mismatches: dict[str, str] = {}
    inception_weights = str(payload.get("inception_weights"))
    if inception_weights != "pretrained":
        mismatches["inception_weights"] = f"expected pretrained got {inception_weights}"

    metrics_validity = str(payload.get("metrics_validity"))
    if metrics_validity not in {"standard", "comparable"}:
        mismatches["metrics_validity"] = f"expected standard/comparable got {metrics_validity}"

    inception_provenance = payload.get("inception_provenance")
    if isinstance(inception_provenance, dict):
        weights_mode = str(inception_provenance.get("weights_mode"))
        if weights_mode != "pretrained":
            mismatches["inception_provenance.weights_mode"] = f"expected pretrained got {weights_mode}"
    else:
        mismatches["inception_provenance"] = "missing"

    has_loaded_ref_path = payload.get("load_reference_stats") is not None
    ref_provenance = payload.get("reference_stats_provenance")
    has_loaded_ref_sha = isinstance(ref_provenance, dict) and ref_provenance.get("loaded_path_sha256") is not None
    if not (has_loaded_ref_path or has_loaded_ref_sha):
        mismatches["reference_stats"] = "expected cached reference stats path/provenance"

    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit claim-facing eval JSON files for pretrained-Inception comparability contract.")
    parser.add_argument("--eval-json", action="append", required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-md", type=str, default=None)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for raw in args.eval_json:
        path = Path(raw)
        payload = _load_json(path)
        mismatches = _check_eval(payload)
        rows.append(
            {
                "path": str(path),
                "pass": len(mismatches) == 0,
                "mismatches": mismatches,
            }
        )

    result = {
        "rows": rows,
        "pass_count": sum(1 for row in rows if bool(row["pass"])),
        "total": len(rows),
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.output_md is not None:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Claim Eval Contract Audit",
            "",
            "| eval_json | pass | mismatch_count |",
            "| --- | ---: | ---: |",
        ]
        for row in rows:
            lines.append(f"| `{row['path']}` | `{str(bool(row['pass'])).lower()}` | {len(row['mismatches'])} |")
        lines.append("")
        for row in rows:
            lines.append(f"## {row['path']}")
            mismatches = row["mismatches"]
            if not mismatches:
                lines.append("- PASS")
            else:
                for key, value in mismatches.items():
                    lines.append(f"- `{key}`: {value}")
            lines.append("")
        lines.append(f"- Pass: {result['pass_count']}/{result['total']}")
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
