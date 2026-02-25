import json
import subprocess
import sys
from pathlib import Path


def test_sample_pixel_writes_imagefolder(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = tmp_path / "pixel_ckpt.pt"
    output_root = tmp_path / "samples"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_pixel.py",
            "--device",
            "cpu",
            "--steps",
            "2",
            "--log-every",
            "1",
            "--groups",
            "2",
            "--negatives-per-group",
            "2",
            "--positives-per-group",
            "2",
            "--image-size",
            "16",
            "--patch-size",
            "4",
            "--hidden-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
            "--checkpoint-path",
            str(checkpoint_path),
            "--save-every",
            "1",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert checkpoint_path.exists()

    result = subprocess.run(
        [
            sys.executable,
            "scripts/sample_pixel.py",
            "--device",
            "cpu",
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-root",
            str(output_root),
            "--n-samples",
            "8",
            "--batch-size",
            "4",
            "--num-classes",
            "1000",
            "--image-size",
            "16",
            "--channels",
            "3",
            "--patch-size",
            "4",
            "--hidden-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    images_dir = Path(payload["images_dir"])
    assert images_dir.exists()
    # Should have at least one class subdir with at least one file.
    class_dirs = [p for p in images_dir.iterdir() if p.is_dir()]
    assert class_dirs
    assert any(any(f.is_file() for f in d.iterdir()) for d in class_dirs)

