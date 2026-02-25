from __future__ import annotations

import json
from pathlib import Path

from scripts import summarize_runtime_preflight as summary_module


def _write_preflight(path: Path, *, status: str, device_arg: str, resolved: str, auto_device: str, fail_checks: list[str]) -> None:
    checks: list[dict[str, object]] = [
        {
            "name": "capabilities.detect",
            "status": "pass",
            "details": {"auto_device": auto_device},
        },
        {
            "name": f"device.resolve({device_arg})",
            "status": "pass" if status == "pass" else "fail",
            "details": {"resolved_device": resolved},
        },
    ]
    checks.extend({"name": item, "status": "fail"} for item in fail_checks)
    payload = {
        "status": status,
        "pass_count": 3,
        "fail_count": len(fail_checks),
        "skip_count": 1,
        "args": {"device": device_arg},
        "torch": {"version": "2.10.0"},
        "checks": checks,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_load_rows_and_build_summary(tmp_path: Path) -> None:
    pass_path = tmp_path / "preflight_linux.json"
    fail_path = tmp_path / "preflight_macos.json"
    _write_preflight(pass_path, status="pass", device_arg="auto", resolved="cuda", auto_device="cuda", fail_checks=[])
    _write_preflight(
        fail_path,
        status="fail",
        device_arg="auto",
        resolved="cpu",
        auto_device="cpu",
        fail_checks=["tensor.smoke(selected:cpu)"],
    )

    rows = summary_module._load_rows(patterns=[str(tmp_path / "preflight_*.json")])
    summary = summary_module._build_summary(rows=rows)

    assert len(rows) == 2
    assert summary["report_count"] == 2
    assert summary["pass_report_count"] == 1
    assert summary["fail_report_count"] == 1
    assert summary["total_checks"]["fail"] == 1
    assert "preflight_macos.json" in summary["failed_reports"]
    triage = summary["triage"]
    assert triage["top_failed_checks"][0]["check"] == "device.resolve(auto)"
    assert triage["top_failed_checks"][0]["severity"] == "critical"
    assert triage["top_failed_checks"][0]["severity_score"] == 100
    top_failed = {str(item["check"]): int(item["count"]) for item in triage["top_failed_checks"]}
    assert top_failed["tensor.smoke(selected:cpu)"] == 1
    assert top_failed["device.resolve(auto)"] == 1
    assert triage["failed_reports_by_backend"]["cpu"] == 1


def test_markdown_includes_failed_check_names(tmp_path: Path) -> None:
    fail_path = tmp_path / "preflight_fail.json"
    _write_preflight(
        fail_path,
        status="fail",
        device_arg="gpu",
        resolved="cpu",
        auto_device="cpu",
        fail_checks=["device.resolve(gpu)", "tensor.smoke(selected:cpu)"],
    )
    rows = summary_module._load_rows(patterns=[str(fail_path)])
    summary = summary_module._build_summary(rows=rows)
    markdown = summary_module._to_markdown(summary=summary, title="Runtime Preflight Summary")

    assert "Runtime Preflight Summary" in markdown
    assert "❌ fail" in markdown
    assert "device.resolve(gpu)" in markdown
    assert "tensor.smoke(selected:cpu)" in markdown
    assert "Failure Triage" in markdown
    assert "[`critical`:100]" in markdown
    assert "| backend | failed reports | top failed checks |" in markdown


def test_severity_ordering_precedes_frequency(tmp_path: Path) -> None:
    _write_preflight(
        tmp_path / "a.json",
        status="fail",
        device_arg="auto",
        resolved="cuda",
        auto_device="cuda",
        fail_checks=["compile.smoke(cuda:0)"],
    )
    _write_preflight(
        tmp_path / "b.json",
        status="fail",
        device_arg="auto",
        resolved="cuda",
        auto_device="cuda",
        fail_checks=["compile.smoke(cuda:0)"],
    )
    _write_preflight(
        tmp_path / "c.json",
        status="fail",
        device_arg="auto",
        resolved="cuda",
        auto_device="cuda",
        fail_checks=["tensor.smoke(selected:cuda)"],
    )

    rows = summary_module._load_rows(patterns=[str(tmp_path / "*.json")])
    summary = summary_module._build_summary(rows=rows)
    top = summary["triage"]["top_failed_checks"]
    names = [str(item["check"]) for item in top]
    assert names.index("tensor.smoke(selected:cuda)") < names.index("compile.smoke(cuda:0)")
