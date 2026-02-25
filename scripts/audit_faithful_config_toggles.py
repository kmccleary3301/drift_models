from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FIELDS: dict[str, str] = {
    "use-feature-loss": "true",
    "feature-encoder": "mae",
    "mae-encoder-arch": "paper_resnet34_unet",
    "feature-temperature-aggregation": "sum_drifts_then_mse",
    "feature-loss-term-reduction": "sum",
    "include-input-x2-mean": "true",
    "include-patch4-stats": "true",
    "feature-include-raw-drift-loss": "true",
    "feature-raw-drift-loss-weight": "1.0",
    "use-queue": "true",
    "queue-strict-without-replacement": "true",
}

_TABLE8_TEMPLATE_OVERRIDES: dict[str, dict[str, str]] = {
    "imagenet1k_sdvae_latents_table8_ablation_default_template.yaml": {
        "feature-base-channels": "256",
        "mae-encoder-path-contains": "w256",
    },
    "imagenet1k_sdvae_latents_table8_b2_template.yaml": {
        "feature-base-channels": "640",
        "mae-encoder-path-contains": "w640",
    },
    "imagenet1k_sdvae_latents_table8_l2_template.yaml": {
        "feature-base-channels": "640",
        "mae-encoder-path-contains": "w640",
    },
}


def _load_simple_kv(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        entries[key.strip()] = value.strip()
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit paper-faithful config toggles on YAML-like key:value configs.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-md", type=str, default=None)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for raw in args.config:
        path = Path(raw)
        cfg = _load_simple_kv(path)
        mismatches: dict[str, dict[str, str | None]] = {}
        for key, expected in REQUIRED_FIELDS.items():
            actual = cfg.get(key)
            if actual != expected:
                mismatches[key] = {"expected": expected, "actual": actual}
        overrides = _TABLE8_TEMPLATE_OVERRIDES.get(path.name, {})
        for key, expected in overrides.items():
            if key == "mae-encoder-path-contains":
                actual_path = cfg.get("mae-encoder-path")
                if actual_path is None or expected not in actual_path:
                    mismatches[key] = {"expected": expected, "actual": actual_path}
                continue
            actual = cfg.get(key)
            if actual != expected:
                mismatches[key] = {"expected": expected, "actual": actual}
        rows.append(
            {
                "config": str(path),
                "pass": len(mismatches) == 0,
                "mismatches": mismatches,
            }
        )

    payload = {
        "required_fields": REQUIRED_FIELDS,
        "rows": rows,
        "pass_count": sum(1 for row in rows if bool(row["pass"])),
        "total": len(rows),
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.output_md is not None:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Faithful Config Toggle Audit",
            "",
            "| config | pass | mismatch_count |",
            "| --- | ---: | ---: |",
        ]
        for row in rows:
            mismatch_count = len(row["mismatches"])
            lines.append(f"| `{row['config']}` | `{str(bool(row['pass'])).lower()}` | {mismatch_count} |")
        lines.append("")
        for row in rows:
            lines.append(f"## {row['config']}")
            mismatches = row["mismatches"]
            if not mismatches:
                lines.append("- PASS")
                lines.append("")
                continue
            for key, diff in mismatches.items():
                lines.append(f"- `{key}` expected `{diff['expected']}` got `{diff['actual']}`")
            lines.append("")
        lines.append(f"- Pass: {payload['pass_count']}/{payload['total']}")
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
