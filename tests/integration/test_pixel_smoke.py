import json
import subprocess
import sys
from pathlib import Path


def test_train_pixel_config_file_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_pixel.py",
            "--config",
            "configs/pixel/smoke_feature.yaml",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["train_config"]["use_feature_loss"] is True
    assert payload["train_config"]["feature_encoder"] == "tiny"
    assert len(payload["logs"]) >= 1
    assert payload["logs"][-1]["loss"] >= 0.0
    assert "perf" in payload
    assert payload["perf"]["mean_step_time_s"] > 0.0
    assert payload["logs"][0]["step_time_s"] > 0.0
