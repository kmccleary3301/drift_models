import json
import subprocess
import sys
from pathlib import Path

import torch


def test_eval_fid_is_cache_roundtrip(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    reference_path = tmp_path / "reference.pt"
    generated_path = tmp_path / "generated.pt"
    cache_path = tmp_path / "ref_stats.pt"

    reference = {"images": torch.rand(32, 3, 64, 64)}
    generated = {"images": torch.rand(32, 3, 64, 64)}
    torch.save(reference, reference_path)
    torch.save(generated, generated_path)

    first = subprocess.run(
        [
            sys.executable,
            "scripts/eval_fid_is.py",
            "--device",
            "cpu",
            "--batch-size",
            "16",
            "--inception-weights",
            "none",
            "--inception-splits",
            "4",
            "--reference-source",
            "tensor_file",
            "--reference-tensor-file-path",
            str(reference_path),
            "--generated-source",
            "tensor_file",
            "--generated-tensor-file-path",
            str(generated_path),
            "--cache-reference-stats",
            str(cache_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert cache_path.exists()
    first_payload = json.loads(first.stdout)
    assert "contract_sha256" in first_payload["protocol"]
    assert first_payload["reference_stats_provenance"]["loaded_contract_sha256"] is None

    second = subprocess.run(
        [
            sys.executable,
            "scripts/eval_fid_is.py",
            "--device",
            "cpu",
            "--batch-size",
            "16",
            "--inception-weights",
            "none",
            "--inception-splits",
            "4",
            "--reference-source",
            "tensor_file",
            "--reference-tensor-file-path",
            str(reference_path),
            "--generated-source",
            "tensor_file",
            "--generated-tensor-file-path",
            str(generated_path),
            "--load-reference-stats",
            str(cache_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    second_payload = json.loads(second.stdout)
    assert abs(first_payload["fid"] - second_payload["fid"]) < 1e-9
    assert (
        second_payload["reference_stats_provenance"]["loaded_contract_sha256"]
        == first_payload["protocol"]["contract_sha256"]
    )


def test_eval_fid_is_rejects_contract_mismatch(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    reference_path = tmp_path / "reference.pt"
    generated_path = tmp_path / "generated.pt"
    cache_path = tmp_path / "ref_stats.pt"

    reference = {"images": torch.rand(16, 3, 64, 64)}
    generated = {"images": torch.rand(16, 3, 64, 64)}
    torch.save(reference, reference_path)
    torch.save(generated, generated_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/eval_fid_is.py",
            "--device",
            "cpu",
            "--batch-size",
            "8",
            "--inception-weights",
            "none",
            "--reference-source",
            "tensor_file",
            "--reference-tensor-file-path",
            str(reference_path),
            "--generated-source",
            "tensor_file",
            "--generated-tensor-file-path",
            str(generated_path),
            "--cache-reference-stats",
            str(cache_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = torch.load(cache_path, map_location="cpu")
    payload["contract_sha256"] = "deadbeef"
    torch.save(payload, cache_path)

    failed = subprocess.run(
        [
            sys.executable,
            "scripts/eval_fid_is.py",
            "--device",
            "cpu",
            "--batch-size",
            "8",
            "--inception-weights",
            "none",
            "--reference-source",
            "tensor_file",
            "--reference-tensor-file-path",
            str(reference_path),
            "--generated-source",
            "tensor_file",
            "--generated-tensor-file-path",
            str(generated_path),
            "--load-reference-stats",
            str(cache_path),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert "Reference stats contract mismatch" in failed.stderr

    passed = subprocess.run(
        [
            sys.executable,
            "scripts/eval_fid_is.py",
            "--device",
            "cpu",
            "--batch-size",
            "8",
            "--inception-weights",
            "none",
            "--reference-source",
            "tensor_file",
            "--reference-tensor-file-path",
            str(reference_path),
            "--generated-source",
            "tensor_file",
            "--generated-tensor-file-path",
            str(generated_path),
            "--load-reference-stats",
            str(cache_path),
            "--allow-reference-contract-mismatch",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = json.loads(passed.stdout)
    assert parsed["reference_samples"] == 16
