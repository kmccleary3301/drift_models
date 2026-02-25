import json
import subprocess
import sys
from pathlib import Path


def test_train_latent_config_file_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--config",
            "configs/latent/smoke_raw.yaml",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["train_config"]["steps"] == 3
    assert payload["train_config"]["groups"] == 2
    assert payload["train_config"]["use_feature_loss"] is False
    assert len(payload["logs"]) >= 1
    assert "perf" in payload
    assert payload["perf"]["mean_step_time_s"] > 0.0
    assert payload["logs"][0]["step_time_s"] > 0.0
