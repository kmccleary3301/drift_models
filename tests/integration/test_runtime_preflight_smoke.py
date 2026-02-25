from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_runtime_preflight_strict_auto_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = tmp_path / "preflight.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/runtime_preflight.py",
            "--device",
            "auto",
            "--check-torchvision",
            "--strict",
            "--output-path",
            str(output_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "pass"
    assert payload["fail_count"] == 0
    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["status"] == "pass"
    check_names = {entry["name"] for entry in saved["checks"]}
    assert "capabilities.detect" in check_names
    assert "import.torchvision" in check_names


def test_runtime_preflight_strict_fails_on_invalid_device(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = tmp_path / "preflight_fail.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/runtime_preflight.py",
            "--device",
            "not-a-real-device",
            "--strict",
            "--output-path",
            str(output_path),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "fail"
    assert payload["fail_count"] >= 1
    assert output_path.exists()


def test_runtime_preflight_compile_disable_fallback_passes_strict(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = tmp_path / "preflight_compile_disable.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/runtime_preflight.py",
            "--device",
            "cpu",
            "--check-compile",
            "--compile-backend",
            "backend-that-does-not-exist",
            "--compile-fail-action",
            "disable",
            "--strict",
            "--output-path",
            str(output_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "pass"
    assert payload["fail_count"] == 0
    compile_checks = [item for item in payload["checks"] if str(item.get("name", "")).startswith("compile.smoke(")]
    assert compile_checks
    assert compile_checks[0]["status"] == "skip"
