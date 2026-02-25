import subprocess
import sys
from pathlib import Path


def test_export_cifar10_imagefolder_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_root = tmp_path / "cifar10_val"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_cifar10_imagefolder.py",
            "--split",
            "val",
            "--max-images",
            "20",
            "--output-root",
            str(out_root),
            "--dataset-cache",
            str(tmp_path / "cache"),
            "--download",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    class_dirs = [p for p in out_root.iterdir() if p.is_dir()]
    assert class_dirs
    assert any(any(f.is_file() for f in d.iterdir()) for d in class_dirs)

