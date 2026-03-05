import json
import subprocess
import sys
from pathlib import Path


def test_script_surface_registry_check_passes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "scripts/check_script_surface_registry.py"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["top_level_scripts"] > 0
