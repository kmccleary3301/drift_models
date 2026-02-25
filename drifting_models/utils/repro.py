from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def environment_fingerprint() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    cudnn_available = bool(torch.backends.cudnn.is_available())
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "cudnn_available": cudnn_available,
        "cudnn_version": int(torch.backends.cudnn.version()) if cudnn_available else None,
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }


def _try_run(argv: list[str]) -> dict[str, Any] | None:
    try:
        result = subprocess.run(argv, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return {
        "argv": argv,
        "returncode": int(result.returncode),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def environment_snapshot(*, paths: list[Path] | None = None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "env_fingerprint": environment_fingerprint(),
        "python_executable": sys.executable,
        "cpu_count": int(os.cpu_count() or 0),
    }

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
        snapshot["ram_bytes"] = int(page_size * phys_pages)
    except (OSError, ValueError, AttributeError):
        snapshot["ram_bytes"] = None

    if paths:
        disk: dict[str, Any] = {}
        for path in paths:
            try:
                usage = shutil.disk_usage(path)
                disk[str(path)] = {"total_bytes": int(usage.total), "used_bytes": int(usage.used), "free_bytes": int(usage.free)}
            except FileNotFoundError:
                disk[str(path)] = {"error": "not_found"}
        snapshot["disk_usage"] = disk

    if torch.cuda.is_available():
        gpus: list[dict[str, Any]] = []
        for idx in range(int(torch.cuda.device_count())):
            props = torch.cuda.get_device_properties(idx)
            gpus.append(
                {
                    "index": int(idx),
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory_bytes": int(getattr(props, "total_memory", 0)),
                    "multi_processor_count": int(getattr(props, "multi_processor_count", 0)),
                    "major": int(getattr(props, "major", 0)),
                    "minor": int(getattr(props, "minor", 0)),
                }
            )
        snapshot["gpus"] = gpus

    nvidia = _try_run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader,nounits"])
    if nvidia is not None:
        snapshot["nvidia_smi_query"] = nvidia

    git_head = _try_run(["git", "rev-parse", "HEAD"])
    if git_head is not None and git_head.get("returncode") == 0 and git_head.get("stdout"):
        snapshot["git_head"] = git_head.get("stdout")

    return snapshot


def codebase_fingerprint(
    *,
    repo_root: Path,
    include_suffixes: tuple[str, ...] = (".py", ".md", ".yaml", ".yml", ".toml", ".json", ".txt", ".lock"),
    include_filenames: tuple[str, ...] = ("uv.lock", "pyproject.toml"),
    exclude_dirs: tuple[str, ...] = (".git", ".venv", "__pycache__", "outputs", "data"),
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    excluded = set(exclude_dirs)

    paths: list[Path] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(repo_root).parts
        if rel_parts and rel_parts[0] in excluded:
            continue
        if path.name in include_filenames or path.suffix in include_suffixes:
            paths.append(path)

    digest = hashlib.sha256()
    relpaths: list[str] = []
    for path in sorted(paths, key=lambda p: str(p.relative_to(repo_root))):
        rel = str(path.relative_to(repo_root))
        relpaths.append(rel)
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("utf-8"))
        digest.update(b"\0")

    manifest_digest = hashlib.sha256("\n".join(relpaths).encode("utf-8")).hexdigest()
    return {
        "repo_root": str(repo_root),
        "sha256": digest.hexdigest(),
        "file_count": int(len(relpaths)),
        "include_suffixes": list(include_suffixes),
        "include_filenames": list(include_filenames),
        "exclude_dirs": list(exclude_dirs),
        "manifest_sha256": manifest_digest,
        "files_sample": relpaths[:50],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
