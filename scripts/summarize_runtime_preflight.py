from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class PreflightRow:
    source_file: str
    status: str
    device_arg: str
    resolved_device: str
    auto_device: str
    torch_version: str
    pass_count: int
    fail_count: int
    skip_count: int
    failed_checks: tuple[str, ...]


def main() -> None:
    args = _parse_args()
    rows = _load_rows(patterns=list(args.input_glob))
    if not rows and not args.allow_empty:
        raise ValueError("No runtime preflight JSON files matched --input-glob.")

    summary = _build_summary(rows=rows)
    markdown = _to_markdown(summary=summary, title=str(args.title))

    if args.output_md is not None:
        output_md = Path(args.output_md).expanduser().resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")
    if args.output_json is not None:
        output_json = Path(args.output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(markdown)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate runtime_preflight JSON reports into a compact markdown table.")
    parser.add_argument(
        "--input-glob",
        type=str,
        nargs="+",
        default=["outputs/runtime_preflight_ci/*.json"],
        help="One or more glob patterns for preflight JSON files.",
    )
    parser.add_argument("--output-md", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--title", type=str, default="Runtime Preflight Summary")
    return parser.parse_args()


def _load_rows(*, patterns: list[str]) -> list[PreflightRow]:
    collected: set[Path] = set()
    for pattern in patterns:
        for raw_path in glob.glob(pattern, recursive=True):
            path = Path(raw_path).expanduser().resolve()
            if path.is_file():
                collected.add(path)
    rows: list[PreflightRow] = []
    for path in sorted(collected):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(_row_from_payload(path=path, payload=payload))
    return rows


def _row_from_payload(*, path: Path, payload: dict[str, object]) -> PreflightRow:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        checks = []
    failed_checks = tuple(
        str(item.get("name"))
        for item in checks
        if isinstance(item, dict) and str(item.get("status")) == "fail"
    )
    return PreflightRow(
        source_file=path.name,
        status=str(payload.get("status", "unknown")),
        device_arg=str(_nested_get(payload, "args", "device", default="unknown")),
        resolved_device=str(_resolved_device_from_checks(checks=checks)),
        auto_device=str(_auto_device_from_checks(checks=checks)),
        torch_version=str(_nested_get(payload, "torch", "version", default="unknown")),
        pass_count=int(payload.get("pass_count", 0)),
        fail_count=int(payload.get("fail_count", 0)),
        skip_count=int(payload.get("skip_count", 0)),
        failed_checks=failed_checks,
    )


def _resolved_device_from_checks(*, checks: list[object]) -> str:
    for item in checks:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name"))
        if not name.startswith("device.resolve("):
            continue
        details = item.get("details")
        if isinstance(details, dict):
            resolved = details.get("resolved_device")
            if resolved is not None:
                return str(resolved)
    return "unknown"


def _auto_device_from_checks(*, checks: list[object]) -> str:
    for item in checks:
        if not isinstance(item, dict):
            continue
        if str(item.get("name")) != "capabilities.detect":
            continue
        details = item.get("details")
        if isinstance(details, dict):
            auto_device = details.get("auto_device")
            if auto_device is not None:
                return str(auto_device)
    return "unknown"


def _nested_get(payload: dict[str, object], key: str, nested_key: str, *, default: object) -> object:
    inner = payload.get(key)
    if not isinstance(inner, dict):
        return default
    return inner.get(nested_key, default)


def _build_summary(*, rows: list[PreflightRow]) -> dict[str, object]:
    fail_reports = [row for row in rows if row.status != "pass"]
    failed_check_counter: Counter[str] = Counter()
    backend_failed_check_counter: dict[str, Counter[str]] = {}
    failed_reports_by_backend: Counter[str] = Counter()
    for row in rows:
        backend = _backend_from_device_string(row.resolved_device)
        if row.status != "pass":
            failed_reports_by_backend[backend] += 1
        for check_name in row.failed_checks:
            failed_check_counter[check_name] += 1
            backend_counter = backend_failed_check_counter.get(backend)
            if backend_counter is None:
                backend_counter = Counter()
                backend_failed_check_counter[backend] = backend_counter
            backend_counter[check_name] += 1
    top_failed_checks = _sorted_failed_check_items(counter=failed_check_counter)
    failed_checks_by_backend = {
        backend: _sorted_failed_check_items(counter=counter)
        for backend, counter in sorted(backend_failed_check_counter.items(), key=lambda item: item[0])
    }
    summary = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "report_count": int(len(rows)),
        "pass_report_count": int(sum(1 for row in rows if row.status == "pass")),
        "fail_report_count": int(len(fail_reports)),
        "total_checks": {
            "pass": int(sum(row.pass_count for row in rows)),
            "fail": int(sum(row.fail_count for row in rows)),
            "skip": int(sum(row.skip_count for row in rows)),
        },
        "rows": [row.__dict__ for row in rows],
        "failed_reports": [row.source_file for row in fail_reports],
        "triage": {
            "top_failed_checks": top_failed_checks,
            "failed_reports_by_backend": {
                backend: int(count)
                for backend, count in sorted(failed_reports_by_backend.items(), key=lambda item: item[0])
            },
            "failed_checks_by_backend": failed_checks_by_backend,
        },
    }
    return summary


def _to_markdown(*, summary: dict[str, object], title: str) -> str:
    rows_raw = summary.get("rows")
    rows: list[dict[str, object]] = rows_raw if isinstance(rows_raw, list) else []

    lines = [f"## {title}", ""]
    lines.append(f"- Generated (UTC): `{summary.get('timestamp_utc', 'unknown')}`")
    lines.append(f"- Reports: `{summary.get('report_count', 0)}`")
    lines.append(f"- Passed reports: `{summary.get('pass_report_count', 0)}`")
    lines.append(f"- Failed reports: `{summary.get('fail_report_count', 0)}`")
    total_checks = summary.get("total_checks")
    if isinstance(total_checks, dict):
        lines.append(
            "- Total checks: "
            f"`pass={total_checks.get('pass', 0)}` "
            f"`fail={total_checks.get('fail', 0)}` "
            f"`skip={total_checks.get('skip', 0)}`"
        )
    triage = summary.get("triage")
    if isinstance(triage, dict):
        lines.append("")
        lines.append("### Failure Triage")
        top_failed_checks = triage.get("top_failed_checks")
        if isinstance(top_failed_checks, list) and top_failed_checks:
            for item in top_failed_checks[:10]:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"- `{item.get('check', 'unknown')}` "
                    f"[`{item.get('severity', 'unknown')}`:{item.get('severity_score', 0)}] "
                    f"→ `{item.get('count', 0)}`"
                )
        else:
            lines.append("- No failed checks detected.")
        failed_reports_by_backend = triage.get("failed_reports_by_backend")
        failed_checks_by_backend = triage.get("failed_checks_by_backend")
        if isinstance(failed_reports_by_backend, dict):
            lines.append("")
            lines.append("| backend | failed reports | top failed checks |")
            lines.append("| --- | ---: | --- |")
            for backend in sorted(failed_reports_by_backend.keys()):
                failed_reports = failed_reports_by_backend.get(backend, 0)
                top_backend_checks = "none"
                if isinstance(failed_checks_by_backend, dict):
                    checks = failed_checks_by_backend.get(backend)
                    if isinstance(checks, list) and checks:
                        top_backend_checks = ", ".join(
                            f"{item.get('check', 'unknown')} "
                            f"({item.get('count', 0)}, {item.get('severity', 'unknown')})"
                            for item in checks[:3]
                            if isinstance(item, dict)
                        )
                lines.append(f"| `{backend}` | `{failed_reports}` | {top_backend_checks} |")
    lines.append("")
    lines.append("| report | status | device arg | resolved | auto | torch | checks (pass/fail/skip) | failed checks |")
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | --- |")
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_cell = "✅ pass" if status == "pass" else f"❌ {status}"
        failed = row.get("failed_checks")
        failed_checks = ""
        if isinstance(failed, (list, tuple)):
            failed_checks = ", ".join(str(item) for item in failed) if failed else "none"
        else:
            failed_checks = "none"
        counts = (
            f"{int(row.get('pass_count', 0))}/"
            f"{int(row.get('fail_count', 0))}/"
            f"{int(row.get('skip_count', 0))}"
        )
        lines.append(
            "| "
            f"`{row.get('source_file', 'unknown')}` | "
            f"{status_cell} | "
            f"`{row.get('device_arg', 'unknown')}` | "
            f"`{row.get('resolved_device', 'unknown')}` | "
            f"`{row.get('auto_device', 'unknown')}` | "
            f"`{row.get('torch_version', 'unknown')}` | "
            f"`{counts}` | "
            f"{failed_checks} |"
        )
    lines.append("")
    return "\n".join(lines)


def _backend_from_device_string(device: str) -> str:
    text = str(device).strip()
    if not text:
        return "unknown"
    if ":" in text:
        return text.split(":", 1)[0]
    return text


def _sorted_failed_check_items(*, counter: Counter[str]) -> list[dict[str, object]]:
    def key(item: tuple[str, int]) -> tuple[int, int, str]:
        check_name, count = item
        severity_score = _severity_score_for_check(check_name)
        return (-severity_score, -int(count), check_name)

    sorted_items = sorted(counter.items(), key=key)
    return [
        {
            "check": check_name,
            "count": int(count),
            "severity_score": int(_severity_score_for_check(check_name)),
            "severity": _severity_label(_severity_score_for_check(check_name)),
        }
        for check_name, count in sorted_items
    ]


def _severity_score_for_check(check_name: str) -> int:
    name = str(check_name)
    if name.startswith("device.resolve("):
        return 100
    if name == "capabilities.detect":
        return 95
    if name.startswith("tensor.smoke("):
        return 90
    if name.startswith("compile.smoke("):
        return 80
    if name.startswith("autocast.smoke("):
        return 70
    if name == "import.torchvision":
        return 60
    if name.startswith("import."):
        return 40
    return 50


def _severity_label(score: int) -> str:
    if score >= 90:
        return "critical"
    if score >= 75:
        return "high"
    if score >= 60:
        return "medium"
    return "low"


if __name__ == "__main__":
    main()
