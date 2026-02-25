import json
import subprocess
import sys
from pathlib import Path


def test_train_mae_config_file_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_mae.py",
            "--config",
            "configs/mae/smoke_latent.yaml",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["config"]["steps"] == 6
    assert len(payload["logs"]) >= 1
    assert payload["logs"][-1]["loss"] >= 0.0
    assert payload["logs"][-1]["val_loss"] >= 0.0
    assert "mask_ratio_target" in payload["logs"][-1]
    assert payload["logs"][-1]["mean_feature_norm"] > 0.0
    assert payload["logs"][-1]["step_time_s"] > 0.0
    assert "perf" in payload
