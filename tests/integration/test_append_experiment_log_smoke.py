import subprocess
import sys
from pathlib import Path


def test_append_experiment_log_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    log_path = tmp_path / "experiment_log.md"
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        '{"output_root":"outputs/x","args":{"device":"cpu","seed":1337,"train_steps":2,"sample_count":8,"inception_weights":"none"},"paths":{"checkpoint_path":"outputs/x/ckpt.pt","eval_summary_path":"outputs/x/eval.json"}}\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/append_experiment_log.py",
            "--summary-json",
            str(summary_path),
            "--experiment-log",
            str(log_path),
            "--kind",
            "end_to_end_pixel",
            "--run-id",
            "EXP-TEST-0001",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "EXP-TEST-0001" in result.stdout
    content = log_path.read_text(encoding="utf-8")
    assert "EXP-TEST-0001" in content

