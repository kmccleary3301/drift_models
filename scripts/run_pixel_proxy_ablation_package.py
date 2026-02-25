from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from drifting_models.utils import add_device_argument, codebase_fingerprint, environment_fingerprint, environment_snapshot, write_json
from drifting_models.utils.run_md import write_run_md


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_tensor = Path(args.reference_tensor_file).expanduser().resolve()
    reference_imagefolder_root = Path(args.reference_imagefolder_root).expanduser().resolve()
    reference_stats_path = Path(args.load_reference_stats).expanduser().resolve()
    if args.eval_profile == "proxy":
        if not reference_tensor.exists():
            raise FileNotFoundError(f"Missing reference tensor file: {reference_tensor}")
    if args.eval_profile == "pretrained_cached":
        if not reference_imagefolder_root.exists():
            raise FileNotFoundError(f"Missing reference imagefolder root: {reference_imagefolder_root}")
        if not reference_stats_path.exists():
            raise FileNotFoundError(f"Missing reference stats file: {reference_stats_path}")

    write_json(output_dir / "env_snapshot.json", environment_snapshot(paths=[output_dir]))
    write_json(output_dir / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))
    write_json(output_dir / "env_fingerprint.json", environment_fingerprint())

    variants = _build_variants()
    commands: dict[str, dict[str, object]] = {}
    results: list[dict[str, object]] = []
    for variant in variants:
        name = str(variant["name"])
        variant_root = output_dir / name
        train_root = variant_root / "train"
        sample_root = variant_root / "samples"
        eval_root = variant_root / "eval"
        for directory in (train_root, sample_root, eval_root):
            directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = train_root / "checkpoint.pt"

        train_argv = [
            sys.executable,
            "scripts/train_pixel.py",
            "--device",
            str(args.device),
            "--steps",
            str(int(args.train_steps)),
            "--log-every",
            str(int(max(1, args.train_steps // 2))),
            "--groups",
            "1",
            "--negatives-per-group",
            "1",
            "--positives-per-group",
            "1",
            "--num-classes",
            "1000",
            "--image-size",
            "32",
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
            "--register-tokens",
            "8",
            "--learning-rate",
            "1e-4",
            "--scheduler",
            "none",
            "--use-feature-loss",
            "--output-dir",
            str(train_root),
            "--checkpoint-path",
            str(checkpoint_path),
            "--save-every",
            str(int(args.train_steps)),
            *[str(token) for token in variant["extra_train_args"]],
        ]
        _run(train_argv, cwd=repo_root)
        commands[f"{name}_train"] = {"argv": list(map(str, train_argv)), "returncode": 0}

        sample_argv = [
            sys.executable,
            "scripts/sample_pixel.py",
            "--device",
            str(args.device),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-root",
            str(sample_root),
            "--n-samples",
            str(int(args.sample_count)),
            "--batch-size",
            str(int(args.sample_batch_size)),
            *[str(token) for token in variant["extra_sample_args"]],
        ]
        _run(sample_argv, cwd=repo_root)
        commands[f"{name}_sample"] = {"argv": list(map(str, sample_argv)), "returncode": 0}

        eval_argv = [
            sys.executable,
            "scripts/eval_fid_is.py",
            "--device",
            str(args.device),
            "--batch-size",
            str(int(args.eval_batch_size)),
        ]
        if args.eval_profile == "proxy":
            eval_argv.extend(
                [
                    "--inception-weights",
                    "none",
                    "--reference-source",
                    "tensor_file",
                    "--reference-tensor-file-path",
                    str(reference_tensor),
                    "--generated-source",
                    "imagefolder",
                    "--generated-imagefolder-root",
                    str(sample_root / "images"),
                    "--max-reference-samples",
                    str(int(args.max_reference_samples)),
                    "--max-generated-samples",
                    str(int(args.sample_count)),
                    "--output-path",
                    str(eval_root / "eval_summary.json"),
                ]
            )
        else:
            eval_argv.extend(
                [
                    "--inception-weights",
                    "pretrained",
                    "--reference-source",
                    "imagefolder",
                    "--reference-imagefolder-root",
                    str(reference_imagefolder_root),
                    "--generated-source",
                    "imagefolder",
                    "--generated-imagefolder-root",
                    str(sample_root / "images"),
                    "--max-generated-samples",
                    str(int(args.sample_count)),
                    "--load-reference-stats",
                    str(reference_stats_path),
                    "--output-path",
                    str(eval_root / "eval_pretrained.json"),
                ]
            )
            if args.allow_reference_contract_mismatch:
                eval_argv.append("--allow-reference-contract-mismatch")
        _run(eval_argv, cwd=repo_root)
        commands[f"{name}_eval"] = {"argv": list(map(str, eval_argv)), "returncode": 0}

        train_summary = json.loads((train_root / "pixel_summary.json").read_text(encoding="utf-8"))
        eval_summary_path = eval_root / ("eval_summary.json" if args.eval_profile == "proxy" else "eval_pretrained.json")
        eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
        final_train_log = train_summary["logs"][-1]
        results.append(
            {
                "name": name,
                "description": variant["description"],
                "feature_encoder": train_summary["train_config"]["feature_encoder"],
                "train_loss": float(final_train_log["loss"]),
                "train_mean_drift_norm": float(final_train_log["mean_drift_norm"]),
                "train_mean_step_time_s": float(train_summary["perf"]["mean_step_time_s"]),
                "train_images_per_sec": float(train_summary["perf"]["mean_generated_images_per_sec"]),
                "proxy_fid": eval_summary.get("fid"),
                "proxy_is_mean": eval_summary.get("inception_score_mean"),
                "proxy_is_std": eval_summary.get("inception_score_std"),
                "paths": {
                    "variant_root": str(variant_root),
                    "train_summary_json": str(train_root / "pixel_summary.json"),
                    "sample_summary_json": str(sample_root / "sample_summary.json"),
                    "eval_summary_json": str(eval_summary_path),
                },
            }
        )

    summary = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "device": args.device,
        "package_type": f"pixel_{args.eval_profile}_ablation",
        "notes": (
            "Proxy evaluator (inception-weights=none, tensor-file reference) for relative ablation only; not claim-facing."
            if args.eval_profile == "proxy"
            else "Pretrained Inception + cached ImageNet val reference stats; small-sample ablation package (not paper-scale)."
        ),
        "results": results,
    }
    summary_json = output_dir / "pixel_proxy_ablation_summary.json"
    summary_md = output_dir / "pixel_proxy_ablation_summary.md"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(_to_markdown(summary), encoding="utf-8")
    write_run_md(
        output_dir / "RUN.md",
        {
            "output_root": str(output_dir),
            "args": vars(args),
            "paths": {
                "summary_json": str(summary_json),
                "summary_markdown": str(summary_md),
                "env_snapshot_json": str(output_dir / "env_snapshot.json"),
                "codebase_fingerprint_json": str(output_dir / "codebase_fingerprint.json"),
                "env_fingerprint_json": str(output_dir / "env_fingerprint.json"),
            },
            "commands": commands,
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "variant_count": len(results)}, indent=2))


def _build_variants() -> list[dict[str, object]]:
    return [
        {
            "name": "tiny",
            "description": "Tiny feature encoder baseline",
            "extra_train_args": ["--feature-encoder", "tiny"],
            "extra_sample_args": [],
        },
        {
            "name": "mae",
            "description": "MAE feature encoder path",
            "extra_train_args": [
                "--feature-encoder",
                "mae",
                "--mae-encoder-arch",
                "paper_resnet34_unet",
                "--feature-stages",
                "4",
            ],
            "extra_sample_args": [],
        },
        {
            "name": "convnext_tiny",
            "description": "ConvNeXt Tiny feature encoder path",
            "extra_train_args": ["--feature-encoder", "convnext_tiny", "--convnext-weights", "none"],
            "extra_sample_args": [],
        },
    ]


def _run(argv: list[str], *, cwd: Path) -> None:
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        raise RuntimeError(
            f"Command failed rc={result.returncode}\nargv={' '.join(map(str, argv))}\nstdout={stdout}\nstderr={stderr}"
        )


def _to_markdown(summary: dict[str, object]) -> str:
    rows = [
        "| variant | description | feature_encoder | train_loss | drift_norm | step_time_s | img/s | proxy_fid | proxy_is_mean |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in summary["results"]:
        rows.append(
            f"| {entry['name']} | {entry['description']} | {entry['feature_encoder']} | "
            f"{entry['train_loss']:.6f} | {entry['train_mean_drift_norm']:.6f} | "
            f"{entry['train_mean_step_time_s']:.4f} | {entry['train_images_per_sec']:.4f} | "
            f"{float(entry['proxy_fid']):.6f} | {float(entry['proxy_is_mean']):.6f} |"
        )
    header = "# Pixel Ablation Summary\n\n"
    notes = f"{summary['notes']}\n\n"
    return header + notes + "\n".join(rows) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pixel proxy ablation package (train -> sample -> proxy eval)")
    add_device_argument(parser, default="cpu")
    parser.add_argument("--output-dir", type=str, default="outputs/feature_ablations/pixel_proxy_ablation")
    parser.add_argument("--eval-profile", choices=("proxy", "pretrained_cached"), default="pretrained_cached")
    parser.add_argument("--reference-tensor-file", type=str, default="outputs/stage8_eval_smoke/ref.pt")
    parser.add_argument("--reference-imagefolder-root", type=str, default="outputs/datasets/imagenet1k_raw/val")
    parser.add_argument("--load-reference-stats", type=str, default="outputs/datasets/imagenet1k_val_reference_stats_pretrained.pt")
    parser.add_argument("--allow-reference-contract-mismatch", action="store_true")
    parser.add_argument("--train-steps", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--sample-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-reference-samples", type=int, default=2048)
    return parser.parse_args()


if __name__ == "__main__":
    main()
