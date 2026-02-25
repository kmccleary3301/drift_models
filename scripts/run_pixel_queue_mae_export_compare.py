from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    export_path = Path(args.mae_encoder_path)
    if not export_path.exists():
        raise FileNotFoundError(
            f"MAE export payload not found: {export_path}. "
            "Create it with scripts/train_mae.py --export-encoder-path first."
        )

    output_root = Path(args.output_root)
    no_export_dir = output_root / "no_export"
    with_export_dir = output_root / "with_export"
    no_export_dir.mkdir(parents=True, exist_ok=True)
    with_export_dir.mkdir(parents=True, exist_ok=True)

    no_export_payload = _run_train_pixel(
        repo_root=repo_root,
        config_path=args.pixel_config,
        output_dir=no_export_dir,
        extra_args=[],
    )
    with_export_payload = _run_train_pixel(
        repo_root=repo_root,
        config_path=args.pixel_config,
        output_dir=with_export_dir,
        extra_args=["--mae-encoder-path", str(export_path)],
    )

    compare = {
        "pixel_config": args.pixel_config,
        "config_hash": no_export_payload.get("config_hash"),
        "no_export": _summarize(no_export_payload),
        "with_export": _summarize(with_export_payload),
    }
    (output_root / "compare.json").write_text(json.dumps(compare, indent=2), encoding="utf-8")
    (output_root / "compare.md").write_text(_format_markdown(compare), encoding="utf-8")
    print(json.dumps(compare, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pixel queue MAE runs with/without loading MAE export weights")
    parser.add_argument("--pixel-config", type=str, default="configs/pixel/smoke_feature_queue_mae.yaml")
    parser.add_argument("--mae-encoder-path", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="outputs/stage7_pixel_queue_mae_export_compare")
    return parser.parse_args()


def _run_train_pixel(
    *,
    repo_root: Path,
    config_path: str,
    output_dir: Path,
    extra_args: list[str],
) -> dict[str, object]:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_pixel.py",
            "--config",
            config_path,
            "--output-dir",
            str(output_dir),
            *extra_args,
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _summarize(payload: dict[str, object]) -> dict[str, object]:
    train_config = payload.get("train_config") if isinstance(payload.get("train_config"), dict) else {}
    logs = payload.get("logs") if isinstance(payload.get("logs"), list) else []
    last_loss = None
    if logs and isinstance(logs[-1], dict) and "loss" in logs[-1]:
        last_loss = float(logs[-1]["loss"])
    perf = payload.get("perf") if isinstance(payload.get("perf"), dict) else {}
    return {
        "mae_encoder_path": train_config.get("mae_encoder_path") if isinstance(train_config, dict) else None,
        "last_loss": last_loss,
        "mean_step_time_s": perf.get("mean_step_time_s"),
        "mean_generated_images_per_sec": perf.get("mean_generated_images_per_sec"),
    }


def _format_markdown(compare: dict[str, object]) -> str:
    no_export = compare["no_export"]
    with_export = compare["with_export"]
    return (
        "# Pixel Queue MAE Export Compare\n\n"
        f"- Pixel config: `{compare['pixel_config']}`\n"
        f"- Config hash: `{compare['config_hash']}`\n"
        f"- No export: last_loss={no_export['last_loss']}, mean_step_time_s={no_export['mean_step_time_s']}, "
        f"mean_img_per_sec={no_export['mean_generated_images_per_sec']}\n"
        f"- With export: last_loss={with_export['last_loss']}, mean_step_time_s={with_export['mean_step_time_s']}, "
        f"mean_img_per_sec={with_export['mean_generated_images_per_sec']}\n"
    )


if __name__ == "__main__":
    main()

