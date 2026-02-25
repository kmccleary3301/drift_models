import json
import subprocess
import sys
from pathlib import Path

import torch


def test_eval_fid_is_tensorfile_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    reference_path = tmp_path / "reference.pt"
    generated_path = tmp_path / "generated.pt"
    output_path = tmp_path / "metrics.json"

    reference = {"images": torch.rand(20, 3, 64, 64)}
    generated = {"images": torch.rand(20, 3, 64, 64)}
    torch.save(reference, reference_path)
    torch.save(generated, generated_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_fid_is.py",
            "--device",
            "cpu",
            "--batch-size",
            "8",
            "--inception-weights",
            "none",
            "--inception-splits",
            "5",
            "--reference-source",
            "tensor_file",
            "--reference-tensor-file-path",
            str(reference_path),
            "--generated-source",
            "tensor_file",
            "--generated-tensor-file-path",
            str(generated_path),
            "--output-path",
            str(output_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["reference_samples"] == 20
    assert payload["generated_samples"] == 20
    assert payload["fid"] >= 0.0
    assert payload["inception_score_mean"] > 0.0
    assert output_path.exists()
