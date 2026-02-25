from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from drifting_models.utils import add_device_argument


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_a_dir = Path(args.run_a_dir)
    run_b_dir = Path(args.run_b_dir)
    output_path = Path(args.output_path)

    if args.clean:
        if run_a_dir.exists():
            shutil.rmtree(run_a_dir)
        if run_b_dir.exists():
            shutil.rmtree(run_b_dir)

    run_a = run_train_toy(
        repo_root=repo_root,
        config=Path(args.config),
        output_dir=run_a_dir,
        ablation=args.ablation,
        device=args.device,
    )
    run_b = run_train_toy(
        repo_root=repo_root,
        config=Path(args.config),
        output_dir=run_b_dir,
        ablation=args.ablation,
        device=args.device,
    )

    hashes_a = collect_hashes(run_a_dir)
    hashes_b = collect_hashes(run_b_dir)
    common_paths = sorted(set(hashes_a).intersection(hashes_b))
    differing_hashes = [path for path in common_paths if hashes_a[path]["sha256"] != hashes_b[path]["sha256"]]
    only_in_a = sorted(set(hashes_a).difference(hashes_b))
    only_in_b = sorted(set(hashes_b).difference(hashes_a))
    files_identical = len(differing_hashes) == 0 and len(only_in_a) == 0 and len(only_in_b) == 0

    summary_compare = compare_summary_json(run_a_dir / "toy_results.json", run_b_dir / "toy_results.json")
    ablation_metrics = extract_ablation_metrics(run_a_dir / "toy_results.json", args.ablation)

    report = {
        "config": str(args.config),
        "ablation": args.ablation,
        "device": args.device,
        "runs": {
            "a": run_a,
            "b": run_b,
        },
        "file_hash_comparison": {
            "files_identical": files_identical,
            "file_count_a": len(hashes_a),
            "file_count_b": len(hashes_b),
            "only_in_a": only_in_a,
            "only_in_b": only_in_b,
            "differing_hashes": differing_hashes,
        },
        "summary_json_comparison": summary_compare,
        "ablation_final_metrics_run_a": ablation_metrics,
        "artifact_hashes_run_a": hashes_a,
        "artifact_hashes_run_b": hashes_b,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "files_identical": files_identical, "summary_equal": summary_compare["equal"]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run toy training twice and compare artifact hashes for determinism.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ablation", type=str, required=True)
    add_device_argument(parser, default="cpu")
    parser.add_argument("--run-a-dir", type=str, required=True)
    parser.add_argument("--run-b-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def run_train_toy(
    *,
    repo_root: Path,
    config: Path,
    output_dir: Path,
    ablation: str,
    device: str,
) -> dict[str, object]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/train_toy.py",
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
        "--ablation",
        ablation,
        "--device",
        device,
    ]
    completed = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"train_toy failed for output_dir={output_dir} with code={completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    return {
        "output_dir": str(output_dir),
        "argv": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def collect_hashes(root: Path) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = str(path.relative_to(root))
        digest = sha256_file(path)
        output[rel] = {"sha256": digest, "size_bytes": path.stat().st_size}
    return output


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compare_summary_json(path_a: Path, path_b: Path) -> dict[str, object]:
    if not path_a.exists() or not path_b.exists():
        return {
            "exists_a": path_a.exists(),
            "exists_b": path_b.exists(),
            "equal": False,
            "reason": "missing_summary_file",
        }
    payload_a = json.loads(path_a.read_text(encoding="utf-8"))
    payload_b = json.loads(path_b.read_text(encoding="utf-8"))
    return {
        "exists_a": True,
        "exists_b": True,
        "equal": payload_a == payload_b,
        "sha256_a": sha256_file(path_a),
        "sha256_b": sha256_file(path_b),
    }


def extract_ablation_metrics(path: Path, ablation: str) -> dict[str, float] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    for entry in results:
        if entry.get("ablation") == ablation:
            metrics = entry.get("final_metrics", {})
            if isinstance(metrics, dict):
                return {str(key): float(value) for key, value in metrics.items()}
    return None


if __name__ == "__main__":
    main()
