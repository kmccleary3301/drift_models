from __future__ import annotations

import importlib
from pathlib import Path


def _write_core_files(run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "RUN.md").write_text("# run\n", encoding="utf-8")
    (run_root / "env_snapshot.json").write_text("{}", encoding="utf-8")
    (run_root / "codebase_fingerprint.json").write_text("{}", encoding="utf-8")


def test_validate_run_artifacts_passes_for_stable_run_with_eval_summary(tmp_path: Path) -> None:
    module = importlib.import_module("scripts.validate_run_artifacts")
    run_root = tmp_path / "stable_20260304_121314"
    _write_core_files(run_root)
    eval_dir = run_root / "claim_bundle" / "claim_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "eval_pretrained.json").write_text("{}", encoding="utf-8")

    report = module.validate_run_artifacts(
        run_root=run_root,
        lane="stable",
        eval_globs=("claim_bundle/claim_eval/eval_pretrained.json",),
        require_eval_summaries=True,
    )
    assert report["pass"] is True
    assert report["failures"] == []


def test_validate_run_artifacts_fails_without_eval_summary_when_required(tmp_path: Path) -> None:
    module = importlib.import_module("scripts.validate_run_artifacts")
    run_root = tmp_path / "exp_ablation_20260304_121314"
    _write_core_files(run_root)

    report = module.validate_run_artifacts(
        run_root=run_root,
        lane="experimental",
        eval_globs=("eval/eval_summary.json",),
        require_eval_summaries=True,
    )
    assert report["pass"] is False
    assert any("Missing eval summaries" in item for item in report["failures"])
