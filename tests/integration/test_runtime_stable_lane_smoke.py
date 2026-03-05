import json
import subprocess
import sys
from pathlib import Path


def test_runtime_stable_lane_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    base_dir = tmp_path / "imagenet"
    timestamp = "20260305_010101"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/runtime_stable_lane.py",
            "--device",
            "cpu",
            "--base-dir",
            str(base_dir),
            "--timestamp",
            timestamp,
            "--steps",
            "1",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    run_root = Path(payload["run_root"])
    assert run_root == (base_dir / f"stable_{timestamp}").resolve()
    assert (run_root / "stable_lane_summary.json").exists()
    assert (run_root / "artifact_validation.json").exists()
    assert (run_root / "latent_summary.json").exists()
