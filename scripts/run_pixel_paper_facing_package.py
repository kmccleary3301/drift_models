from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from drifting_models.utils import add_device_argument, codebase_fingerprint, environment_fingerprint, environment_snapshot, write_json
from drifting_models.utils.checkpoints import select_last_k_checkpoints
from drifting_models.utils.run_md import write_run_md


@dataclass(frozen=True)
class CommandRecord:
    name: str
    argv: list[str]
    returncode: int
    duration_seconds: float


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_path = (repo_root / args.checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing --checkpoint-path: {checkpoint_path}")
    checkpoint_dir = None if args.checkpoint_dir is None else (repo_root / args.checkpoint_dir).resolve()
    config_path = None if args.config is None else (repo_root / args.config).resolve()
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    reference_imagefolder_root = (repo_root / args.reference_imagefolder_root).resolve()
    reference_tensor_file = (repo_root / args.reference_tensor_file).resolve()
    load_reference_stats = (repo_root / args.load_reference_stats).resolve()
    if args.eval_profile == "proxy":
        if not reference_tensor_file.exists():
            raise FileNotFoundError(f"Missing --reference-tensor-file: {reference_tensor_file}")
    if args.eval_profile == "pretrained_cached":
        if not reference_imagefolder_root.exists():
            raise FileNotFoundError(f"Missing --reference-imagefolder-root: {reference_imagefolder_root}")
        if not load_reference_stats.exists():
            raise FileNotFoundError(f"Missing --load-reference-stats: {load_reference_stats}")

    write_json(output_root / "env_snapshot.json", environment_snapshot(paths=[output_root]))
    write_json(output_root / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))
    write_json(output_root / "env_fingerprint.json", environment_fingerprint())

    commands: list[CommandRecord] = []
    last_k_results: list[dict[str, object]] = []
    if (not args.skip_last_k) and checkpoint_dir is not None and checkpoint_dir.exists():
        for item in select_last_k_checkpoints(checkpoint_dir, k=int(args.last_k)):
            step_root = output_root / "last_k" / f"step_{item.step:08d}"
            step_root.mkdir(parents=True, exist_ok=True)
            result = _sample_eval_package(
                repo_root=repo_root,
                checkpoint_path=item.path,
                config_path=config_path,
                output_root=step_root,
                alpha=float(args.claim_alpha),
                n_samples=int(args.last_k_samples),
                sample_batch_size=int(args.last_k_batch_size),
                eval_batch_size=int(args.eval_batch_size),
                device=str(args.device),
                eval_profile=str(args.eval_profile),
                reference_imagefolder_root=reference_imagefolder_root,
                reference_tensor_file=reference_tensor_file,
                load_reference_stats=load_reference_stats,
                allow_reference_contract_mismatch=bool(args.allow_reference_contract_mismatch),
                command_records=commands,
                name_prefix=f"last_k_{item.step}",
            )
            result["step"] = int(item.step)
            result["checkpoint_path"] = str(item.path)
            last_k_results.append(result)

    alpha_sweep_results: list[dict[str, object]] = []
    if not args.skip_alpha_sweep:
        for alpha in args.alphas:
            alpha_key = _format_alpha(float(alpha))
            alpha_root = output_root / "alpha_sweep" / f"alpha_{alpha_key}"
            alpha_root.mkdir(parents=True, exist_ok=True)
            result = _sample_eval_package(
                repo_root=repo_root,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                output_root=alpha_root,
                alpha=float(alpha),
                n_samples=int(args.alpha_sweep_samples),
                sample_batch_size=int(args.alpha_sweep_batch_size),
                eval_batch_size=int(args.eval_batch_size),
                device=str(args.device),
                eval_profile=str(args.eval_profile),
                reference_imagefolder_root=reference_imagefolder_root,
                reference_tensor_file=reference_tensor_file,
                load_reference_stats=load_reference_stats,
                allow_reference_contract_mismatch=bool(args.allow_reference_contract_mismatch),
                command_records=commands,
                name_prefix=f"alpha_{alpha_key}",
            )
            result["alpha"] = float(alpha)
            alpha_sweep_results.append(result)

    claim_root = output_root / "claim"
    claim_root.mkdir(parents=True, exist_ok=True)
    claim_result = _sample_eval_package(
        repo_root=repo_root,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        output_root=claim_root,
        alpha=float(args.claim_alpha),
        n_samples=int(args.claim_sample_count),
        sample_batch_size=int(args.claim_sample_batch_size),
        eval_batch_size=int(args.eval_batch_size),
        device=str(args.device),
        eval_profile=str(args.eval_profile),
        reference_imagefolder_root=reference_imagefolder_root,
        reference_tensor_file=reference_tensor_file,
        load_reference_stats=load_reference_stats,
        allow_reference_contract_mismatch=bool(args.allow_reference_contract_mismatch),
        command_records=commands,
        name_prefix="claim",
    )

    nn_output_path = claim_root / "nn_audit.json"
    commands.append(
        _run(
            name="claim_nn_audit",
            argv=[
                sys.executable,
                "scripts/audit_nearest_neighbors.py",
                "--generated-root",
                str(claim_root / "samples" / "images"),
                "--reference-root",
                str(reference_imagefolder_root),
                "--device",
                str(args.device),
                "--max-generated",
                str(int(args.nn_max_generated)),
                "--max-reference",
                str(int(args.nn_max_reference)),
                "--output-path",
                str(nn_output_path),
            ],
            cwd=repo_root,
        )
    )

    summary = {
        "timestamp": _timestamp(),
        "package_type": "pixel_paper_facing_package",
        "eval_profile": str(args.eval_profile),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
        "config_path": None if config_path is None else str(config_path),
        "paths": {
            "output_root": str(output_root),
            "claim_root": str(claim_root),
            "claim_nn_audit_json": str(nn_output_path),
            "summary_json": str(output_root / "paper_facing_package_summary.json"),
        },
        "args": vars(args),
        "last_k_results": last_k_results,
        "alpha_sweep_results": alpha_sweep_results,
        "claim_result": claim_result,
        "commands": [record.__dict__ for record in commands],
        "env_fingerprint": environment_fingerprint(),
    }
    write_json(output_root / "paper_facing_package_summary.json", summary)
    (output_root / "paper_facing_package_summary.md").write_text(_to_markdown(summary), encoding="utf-8")
    write_run_md(output_root / "RUN.md", summary)
    print(json.dumps({"output_root": str(output_root), "commands": len(commands)}, indent=2))


def _sample_eval_package(
    *,
    repo_root: Path,
    checkpoint_path: Path,
    config_path: Path | None,
    output_root: Path,
    alpha: float,
    n_samples: int,
    sample_batch_size: int,
    eval_batch_size: int,
    device: str,
    eval_profile: str,
    reference_imagefolder_root: Path,
    reference_tensor_file: Path,
    load_reference_stats: Path,
    allow_reference_contract_mismatch: bool,
    command_records: list[CommandRecord],
    name_prefix: str,
) -> dict[str, object]:
    sample_root = output_root / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    sample_cmd = [
        sys.executable,
        "scripts/sample_pixel.py",
        "--device",
        device,
        "--checkpoint-path",
        str(checkpoint_path),
        "--output-root",
        str(sample_root),
        "--n-samples",
        str(int(n_samples)),
        "--batch-size",
        str(int(sample_batch_size)),
        "--alpha",
        str(float(alpha)),
        "--alpha-schedule",
        "constant",
    ]
    if config_path is not None:
        sample_cmd += ["--config", str(config_path)]
    command_records.append(_run(name=f"{name_prefix}_sample", argv=sample_cmd, cwd=repo_root))

    eval_summary_path = output_root / ("eval_pretrained.json" if eval_profile == "pretrained_cached" else "eval_proxy.json")
    eval_cmd = [
        sys.executable,
        "scripts/eval_fid_is.py",
        "--device",
        device,
        "--batch-size",
        str(int(eval_batch_size)),
        "--generated-source",
        "imagefolder",
        "--generated-imagefolder-root",
        str(sample_root / "images"),
        "--max-generated-samples",
        str(int(n_samples)),
        "--output-path",
        str(eval_summary_path),
    ]
    if eval_profile == "proxy":
        eval_cmd += [
            "--inception-weights",
            "none",
            "--reference-source",
            "tensor_file",
            "--reference-tensor-file-path",
            str(reference_tensor_file),
            "--max-reference-samples",
            "2048",
        ]
    else:
        eval_cmd += [
            "--inception-weights",
            "pretrained",
            "--reference-source",
            "imagefolder",
            "--reference-imagefolder-root",
            str(reference_imagefolder_root),
            "--load-reference-stats",
            str(load_reference_stats),
        ]
        if allow_reference_contract_mismatch:
            eval_cmd.append("--allow-reference-contract-mismatch")
    command_records.append(_run(name=f"{name_prefix}_eval", argv=eval_cmd, cwd=repo_root))

    sample_summary = json.loads((sample_root / "sample_summary.json").read_text(encoding="utf-8"))
    eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
    return {
        "alpha": float(alpha),
        "n_samples": int(n_samples),
        "paths": {
            "sample_root": str(sample_root),
            "sample_summary_json": str(sample_root / "sample_summary.json"),
            "eval_summary_json": str(eval_summary_path),
        },
        "sample_saved_images": int(sample_summary.get("saved_images", 0)),
        "fid": eval_summary.get("fid"),
        "inception_score_mean": eval_summary.get("inception_score_mean"),
        "inception_score_std": eval_summary.get("inception_score_std"),
    }


def _run(*, name: str, argv: list[str], cwd: Path) -> CommandRecord:
    started = time.time()
    result = subprocess.run(argv, cwd=cwd, check=False)
    duration_seconds = float(time.time() - started)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed name={name} rc={result.returncode}")
    return CommandRecord(
        name=name,
        argv=list(map(str, argv)),
        returncode=int(result.returncode),
        duration_seconds=duration_seconds,
    )


def _format_alpha(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _to_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Pixel Paper-Facing Package Summary",
        "",
        f"- Eval profile: `{summary.get('eval_profile')}`",
        f"- Checkpoint: `{summary.get('checkpoint_path')}`",
        "",
        "## Claim",
    ]
    claim = summary.get("claim_result", {})
    if isinstance(claim, dict):
        lines.append(
            f"- alpha={claim.get('alpha')} n={claim.get('n_samples')} fid={claim.get('fid')} "
            f"is={claim.get('inception_score_mean')}"
        )
    lines.append("")
    lines.append("## Last-K")
    lines.append("| step | alpha | n | fid | inception_score |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for item in summary.get("last_k_results", []):
        if not isinstance(item, dict):
            continue
        lines.append(
            f"| {item.get('step')} | {item.get('alpha')} | {item.get('n_samples')} | "
            f"{item.get('fid')} | {item.get('inception_score_mean')} |"
        )
    lines.append("")
    lines.append("## Alpha Sweep")
    lines.append("| alpha | n | fid | inception_score |")
    lines.append("| ---: | ---: | ---: | ---: |")
    for item in summary.get("alpha_sweep_results", []):
        if not isinstance(item, dict):
            continue
        lines.append(
            f"| {item.get('alpha')} | {item.get('n_samples')} | "
            f"{item.get('fid')} | {item.get('inception_score_mean')} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a pixel paper-facing evaluation package from checkpoints.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-root", type=str, required=True)
    add_device_argument(parser, default="cpu")
    parser.add_argument("--eval-profile", choices=("proxy", "pretrained_cached"), default="pretrained_cached")
    parser.add_argument("--reference-imagefolder-root", type=str, default="outputs/datasets/imagenet1k_raw/val")
    parser.add_argument("--reference-tensor-file", type=str, default="outputs/stage8_eval_smoke/ref.pt")
    parser.add_argument("--load-reference-stats", type=str, default="outputs/datasets/imagenet1k_val_reference_stats_pretrained.pt")
    parser.add_argument("--allow-reference-contract-mismatch", action="store_true")
    parser.add_argument("--skip-last-k", action="store_true")
    parser.add_argument("--last-k", type=int, default=2)
    parser.add_argument("--last-k-samples", type=int, default=256)
    parser.add_argument("--last-k-batch-size", type=int, default=64)
    parser.add_argument("--skip-alpha-sweep", action="store_true")
    parser.add_argument("--alphas", nargs="+", type=float, default=[1.0, 1.5, 2.0, 2.5, 3.0])
    parser.add_argument("--alpha-sweep-samples", type=int, default=256)
    parser.add_argument("--alpha-sweep-batch-size", type=int, default=64)
    parser.add_argument("--claim-alpha", type=float, default=1.5)
    parser.add_argument("--claim-sample-count", type=int, default=1024)
    parser.add_argument("--claim-sample-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--nn-max-generated", type=int, default=256)
    parser.add_argument("--nn-max-reference", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    main()
