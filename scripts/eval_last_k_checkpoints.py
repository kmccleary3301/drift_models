from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from drifting_models.utils import add_device_argument, codebase_fingerprint, environment_fingerprint, environment_snapshot, write_json
from drifting_models.utils.checkpoints import select_last_k_checkpoints
from drifting_models.utils.experiment_log import (
    append_experiment_log,
    build_entry_from_summary,
    default_run_id,
)
from drifting_models.utils.run_md import write_run_md


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "env_snapshot.json", environment_snapshot(paths=[output_root]))
    write_json(output_root / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))

    checkpoint_dir = Path(args.checkpoint_dir)
    selected = select_last_k_checkpoints(checkpoint_dir, k=int(args.k))
    if not selected:
        raise FileNotFoundError(f"No step checkpoints found in {checkpoint_dir}")

    reference_stats_path = output_root / "reference_stats.pt"
    use_reference_cache = bool(args.reference_cache)
    preload_reference_stats = None if args.load_reference_stats is None else Path(args.load_reference_stats)
    if preload_reference_stats is not None and not preload_reference_stats.exists():
        raise FileNotFoundError(f"Missing --load-reference-stats file: {preload_reference_stats}")
    overwrite = bool(args.overwrite)

    results: list[dict[str, object]] = []
    for item in selected:
        run_root = output_root / f"step_{item.step:08d}"
        sample_root = run_root / "samples"
        eval_root = run_root / "eval"
        eval_root.mkdir(parents=True, exist_ok=True)
        eval_summary_path = eval_root / "eval_summary.json"

        if eval_summary_path.exists() and not overwrite:
            results.append(
                {
                    "step": int(item.step),
                    "checkpoint_path": str(item.path),
                    "skipped": True,
                    "paths": {"run_root": str(run_root), "eval_summary_path": str(eval_summary_path)},
                }
            )
            continue

        sample_root.mkdir(parents=True, exist_ok=True)
        sample_cmd = _build_sample_cmd(args=args, checkpoint_path=item.path, sample_root=sample_root)
        sample_result = _run(sample_cmd, cwd=repo_root)
        if sample_result.returncode != 0:
            raise RuntimeError(sample_result.stderr)

        generated_root = sample_root / "images"
        if not generated_root.exists():
            raise FileNotFoundError(f"Missing generated ImageFolder: {generated_root}")

        eval_cmd = _build_eval_cmd(
            args=args,
            generated_root=generated_root,
            eval_summary_path=eval_summary_path,
            reference_stats_path=reference_stats_path,
            use_reference_cache=use_reference_cache,
            preload_reference_stats=preload_reference_stats,
        )
        eval_result = _run(eval_cmd, cwd=repo_root)
        if eval_result.returncode != 0:
            raise RuntimeError(eval_result.stderr)

        metrics = json.loads(eval_summary_path.read_text(encoding="utf-8"))
        results.append(
            {
                "step": int(item.step),
                "checkpoint_path": str(item.path),
                "skipped": False,
                "metrics": metrics,
                "paths": {
                    "run_root": str(run_root),
                    "sample_root": str(sample_root),
                    "generated_root": str(generated_root),
                    "eval_summary_path": str(eval_summary_path),
                },
                "commands": {"sample": sample_result.__dict__, "eval": eval_result.__dict__},
            }
        )

    summary = {
        "output_root": str(output_root),
        "args": vars(args),
        "selected_checkpoints": [{"step": int(item.step), "path": str(item.path)} for item in selected],
        "reference_stats_path": str(reference_stats_path),
        "load_reference_stats": str(preload_reference_stats) if preload_reference_stats is not None else None,
        "env_fingerprint": environment_fingerprint(),
        "results": results,
    }
    write_json(output_root / "last_k_summary.json", summary)
    (output_root / "last_k_summary.md").write_text(_render_md(summary), encoding="utf-8")
    write_run_md(output_root / "RUN.md", summary)
    if args.append_experiment_log:
        run_id = args.experiment_run_id or default_run_id(args.experiment_run_id_prefix)
        entry = build_entry_from_summary(
            kind="last_k",
            run_id=run_id,
            date=args.experiment_date,
            summary=summary,
            notes=args.experiment_notes,
        )
        append_experiment_log(log_path=Path(args.experiment_log_path), entry_md=entry)
    print(json.dumps(summary, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the last K step checkpoints: sample -> eval per checkpoint")
    parser.add_argument("--output-root", type=str, required=True)
    add_device_argument(parser, default="cpu")
    parser.add_argument("--mode", choices=("pixel", "latent"), required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--allow-config-mismatch", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--postprocess-mode", choices=("clamp_0_1", "tanh_to_0_1", "sigmoid", "identity"), default=None)

    parser.add_argument("--decode-mode", choices=("identity", "conv", "sd_vae"), default="conv")
    parser.add_argument("--decode-image-size", type=int, default=32)
    parser.add_argument("--sd-vae-model-id", type=str, default=None)
    parser.add_argument("--sd-vae-subfolder", type=str, default=None)
    parser.add_argument("--sd-vae-revision", type=str, default="31f26fdeee1355a5c34592e401dd41e45d25a493")
    parser.add_argument("--sd-vae-scaling-factor", type=float, default=0.18215)
    parser.add_argument("--sd-vae-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")

    parser.add_argument("--reference-imagefolder-root", type=str, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--inception-weights", choices=("none", "pretrained"), default="none")
    parser.add_argument("--reference-cache", action="store_true")
    parser.add_argument("--load-reference-stats", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--append-experiment-log", action="store_true")
    parser.add_argument("--experiment-log-path", type=str, default="docs/experiment_log.md")
    parser.add_argument("--experiment-run-id", type=str, default=None)
    parser.add_argument("--experiment-run-id-prefix", type=str, default="EXP-LAST-K")
    parser.add_argument("--experiment-date", type=str, default=None)
    parser.add_argument("--experiment-notes", type=str, default=None)
    return parser.parse_args()


def _build_sample_cmd(*, args: argparse.Namespace, checkpoint_path: Path, sample_root: Path) -> list[str]:
    if args.mode == "pixel":
        cmd = [
            sys.executable,
            "scripts/sample_pixel.py",
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-root",
            str(sample_root),
            "--n-samples",
            str(args.n_samples),
            "--batch-size",
            str(args.batch_size),
        ]
        if args.config is not None:
            cmd += ["--config", args.config]
        if args.allow_config_mismatch:
            cmd += ["--allow-config-mismatch"]
        if args.postprocess_mode is not None:
            cmd += ["--postprocess-mode", args.postprocess_mode]
        return cmd

    if args.mode == "latent":
        if args.decode_mode == "sd_vae" and (args.sd_vae_model_id is None or not str(args.sd_vae_model_id).strip()):
            raise ValueError("--sd-vae-model-id must be provided when --decode-mode sd_vae")
        cmd = [
            sys.executable,
            "scripts/sample_latent.py",
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-root",
            str(sample_root),
            "--n-samples",
            str(args.n_samples),
            "--batch-size",
            str(args.batch_size),
            "--write-imagefolder",
            "--decode-mode",
            args.decode_mode,
            "--decode-image-size",
            str(args.decode_image_size),
        ]
        if args.decode_mode == "sd_vae":
            cmd += [
                "--sd-vae-model-id",
                str(args.sd_vae_model_id),
                "--sd-vae-scaling-factor",
                str(float(args.sd_vae_scaling_factor)),
                "--sd-vae-dtype",
                str(args.sd_vae_dtype),
            ]
            if args.sd_vae_subfolder is not None:
                cmd += ["--sd-vae-subfolder", str(args.sd_vae_subfolder)]
            if args.sd_vae_revision is not None:
                cmd += ["--sd-vae-revision", str(args.sd_vae_revision)]
        if args.config is not None:
            cmd += ["--config", args.config]
        if args.allow_config_mismatch:
            cmd += ["--allow-config-mismatch"]
        if args.postprocess_mode is not None:
            cmd += ["--postprocess-mode", args.postprocess_mode]
        return cmd

    raise ValueError(f"Unsupported mode: {args.mode}")


def _build_eval_cmd(
    *,
    args: argparse.Namespace,
    generated_root: Path,
    eval_summary_path: Path,
    reference_stats_path: Path,
    use_reference_cache: bool,
    preload_reference_stats: Path | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/eval_fid_is.py",
        "--device",
        args.device,
        "--batch-size",
        str(args.eval_batch_size),
        "--inception-weights",
        args.inception_weights,
        "--reference-source",
        "imagefolder",
        "--reference-imagefolder-root",
        args.reference_imagefolder_root,
        "--generated-source",
        "imagefolder",
        "--generated-imagefolder-root",
        str(generated_root),
        "--output-path",
        str(eval_summary_path),
    ]
    if preload_reference_stats is not None:
        cmd += ["--load-reference-stats", str(preload_reference_stats)]
    elif use_reference_cache:
        if reference_stats_path.exists():
            cmd += ["--load-reference-stats", str(reference_stats_path)]
        else:
            cmd += ["--cache-reference-stats", str(reference_stats_path)]
    return cmd


def _render_md(payload: dict[str, object]) -> str:
    results = payload.get("results", [])
    if not isinstance(results, list):
        results = []
    lines = [
        "# Last-K Checkpoint Evaluation Summary",
        "",
        f"- Output root: `{payload.get('output_root', '')}`",
        f"- Reference stats cache: `{payload.get('reference_stats_path', '')}`",
        "",
        "| step | fid | inception_score | skipped | checkpoint |",
        "| ---: | ---: | ---: | :---: | :--- |",
    ]
    for entry in results:
        if not isinstance(entry, dict):
            continue
        step = entry.get("step", "")
        skipped = bool(entry.get("skipped", False))
        metrics = entry.get("metrics", {}) if isinstance(entry.get("metrics", {}), dict) else {}
        fid = metrics.get("fid", "")
        inc = metrics.get("inception_score_mean", "")
        ckpt = entry.get("checkpoint_path", "")
        lines.append(f"| {step} | {fid} | {inc} | {str(skipped).lower()} | `{ckpt}` |")
    lines.append("")
    return "\n".join(lines)


def _run(argv: list[str], *, cwd: Path) -> CommandResult:
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    return CommandResult(argv=list(map(str, argv)), returncode=int(result.returncode), stdout=result.stdout, stderr=result.stderr)


if __name__ == "__main__":
    main()
