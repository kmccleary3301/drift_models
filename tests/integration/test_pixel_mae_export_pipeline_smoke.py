import json
import subprocess
import sys
from pathlib import Path


def test_pixel_mae_export_pipeline_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_root = tmp_path / "pipeline"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_pixel_mae_export_pipeline.py",
            "--output-root",
            str(out_root),
            "--device",
            "cpu",
            "--pixel-config",
            "configs/pixel/smoke_feature_queue_mae.yaml",
            "--mae-steps",
            "2",
            "--pixel-steps",
            "2",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert (out_root / "pipeline_summary.json").exists()
    assert Path(payload["paths"]["export_path"]).exists()
    assert Path(payload["paths"]["pixel_summary_path"]).exists()

