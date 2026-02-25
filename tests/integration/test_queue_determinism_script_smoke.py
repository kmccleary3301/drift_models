import json
import subprocess
import sys
from pathlib import Path


def test_check_queue_determinism_script_passes(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_path = tmp_path / "queue_determinism.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_queue_determinism.py",
            "--device",
            "cpu",
            "--seed",
            "1337",
            "--output-path",
            str(out_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["passed"] is True
    assert out_path.exists()

