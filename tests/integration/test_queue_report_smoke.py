from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_queue_report_smoke(tmp_path: Path) -> None:
    summary = {
        "queue_report_level": "full",
        "queue_warmup_report": {
            "warmup_mode": "random",
            "samples_pushed": 100.0,
            "global_count": 100.0,
            "covered_classes": 3.0,
            "min_class_count_nonzero": 1.0,
            "max_class_count": 64.0,
            "mean_class_count": 0.1,
            "counts": [64, 1, 35, 0, 0],
            "count_quantiles_nonzero": {"p10": 1.0, "p50": 35.0, "p90": 64.0},
        },
        "logs": [
            {"step": 1.0, "queue_global_count": 10.0, "queue_covered_classes": 2.0},
            {"step": 2.0, "queue_global_count": 20.0, "queue_covered_classes": 3.0},
        ],
    }
    summary_path = tmp_path / "latent_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    out_path = tmp_path / "queue_report.md"

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/queue_report.py",
            "--summary-path",
            str(summary_path),
            "--output-path",
            str(out_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "Queue Report" in text

