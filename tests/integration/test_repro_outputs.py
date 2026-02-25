import json
import subprocess
import sys
from pathlib import Path


def test_train_latent_writes_repro_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "artifacts"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--config",
            "configs/latent/smoke_raw.yaml",
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["config_hash"] is not None
    assert len(payload["config_hash"]) == 64
    assert (output_dir / "latent_summary.json").exists()
    assert (output_dir / "env_fingerprint.json").exists()
