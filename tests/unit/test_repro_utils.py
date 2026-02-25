from pathlib import Path

from drifting_models.utils import codebase_fingerprint, environment_fingerprint, environment_snapshot, file_sha256, payload_sha256, write_json


def test_payload_sha256_is_stable() -> None:
    payload = {"b": 2, "a": 1}
    first = payload_sha256(payload)
    second = payload_sha256({"a": 1, "b": 2})
    assert first == second
    assert len(first) == 64


def test_file_sha256_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("drifting-models", encoding="utf-8")
    digest = file_sha256(path)
    assert len(digest) == 64


def test_environment_fingerprint_fields() -> None:
    fingerprint = environment_fingerprint()
    assert "python" in fingerprint
    assert "torch" in fingerprint
    assert "cuda_available" in fingerprint


def test_environment_snapshot_fields(tmp_path: Path) -> None:
    snapshot = environment_snapshot(paths=[tmp_path])
    assert "timestamp" in snapshot
    assert "env_fingerprint" in snapshot
    assert "disk_usage" in snapshot


def test_codebase_fingerprint_fields() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fp = codebase_fingerprint(repo_root=repo_root)
    assert "sha256" in fp
    assert fp["file_count"] > 0


def test_write_json_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "payload.json"
    write_json(path, {"ok": True})
    assert path.exists()
