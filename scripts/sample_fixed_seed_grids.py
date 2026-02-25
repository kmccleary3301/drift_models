from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from drifting_models.utils import add_device_argument, write_json


def main() -> None:
    args = _parse_args()
    checkpoint_paths = _discover_checkpoints(
        checkpoint_dir=Path(args.checkpoint_dir),
        max_checkpoints=int(args.max_checkpoints),
    )
    if not checkpoint_paths:
        raise ValueError(f"No checkpoints found in {args.checkpoint_dir}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for checkpoint_path in checkpoint_paths:
        step_token = _checkpoint_step_token(checkpoint_path)
        sample_root = output_root / f"samples_{step_token}"
        sample_root.mkdir(parents=True, exist_ok=True)
        cmd = _build_sample_command(args=args, checkpoint_path=checkpoint_path, sample_root=sample_root)
        result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=False)
        row: dict[str, object] = {
            "checkpoint_path": str(checkpoint_path),
            "sample_root": str(sample_root),
            "argv": cmd,
            "returncode": int(result.returncode),
            "stderr_tail": result.stderr[-2000:],
        }
        if result.returncode == 0:
            images = sorted((sample_root / "images").rglob("*.jpg")) + sorted((sample_root / "images").rglob("*.png"))
            if not images:
                row["returncode"] = 1
                row["stderr_tail"] = "No decoded images found after sampling."
            else:
                grid_path = output_root / f"grid_{step_token}.png"
                _write_grid(images=images[: int(args.grid_samples)], output_path=grid_path, columns=int(args.grid_cols))
                row["grid_path"] = str(grid_path)
                row["image_count"] = len(images)
        rows.append(row)

    summary = {
        "checkpoint_dir": args.checkpoint_dir,
        "output_root": str(output_root),
        "seed": int(args.seed),
        "grid_samples": int(args.grid_samples),
        "grid_cols": int(args.grid_cols),
        "rows": rows,
    }
    summary_path = output_root / "fixed_seed_grid_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary_path": str(summary_path), "checkpoint_count": len(rows)}, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample fixed-seed grids across checkpoint trajectory")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    add_device_argument(parser, default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--grid-samples", type=int, default=16)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-checkpoints", type=int, default=6)
    parser.add_argument("--decode-mode", choices=("sd_vae", "conv", "identity"), default="sd_vae")
    parser.add_argument("--decode-image-size", type=int, default=256)
    parser.add_argument("--sd-vae-model-id", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--sd-vae-subfolder", type=str, default=None)
    parser.add_argument("--sd-vae-revision", type=str, default="31f26fdeee1355a5c34592e401dd41e45d25a493")
    return parser.parse_args()


def _discover_checkpoints(*, checkpoint_dir: Path, max_checkpoints: int) -> list[Path]:
    paths = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
    if max_checkpoints > 0 and len(paths) > max_checkpoints:
        stride = max(1, len(paths) // max_checkpoints)
        sampled = paths[::stride]
        if sampled[-1] != paths[-1]:
            sampled.append(paths[-1])
        paths = sampled[:max_checkpoints]
    return paths


def _build_sample_command(*, args: argparse.Namespace, checkpoint_path: Path, sample_root: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/sample_latent.py",
        "--checkpoint-path",
        str(checkpoint_path),
        "--output-root",
        str(sample_root),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
        "--alpha",
        str(float(args.alpha)),
        "--n-samples",
        str(int(args.grid_samples)),
        "--batch-size",
        str(int(args.batch_size)),
        "--write-imagefolder",
        "--decode-mode",
        str(args.decode_mode),
        "--decode-image-size",
        str(int(args.decode_image_size)),
        "--postprocess-mode",
        "clamp_0_1",
        "--image-format",
        "jpg",
    ]
    if args.config is not None:
        cmd.extend(["--config", str(args.config)])
    if args.decode_mode == "sd_vae":
        cmd.extend(["--sd-vae-model-id", str(args.sd_vae_model_id)])
        if args.sd_vae_subfolder is not None:
            cmd.extend(["--sd-vae-subfolder", str(args.sd_vae_subfolder)])
        if args.sd_vae_revision is not None:
            cmd.extend(["--sd-vae-revision", str(args.sd_vae_revision)])
    return cmd


def _checkpoint_step_token(path: Path) -> str:
    stem = path.stem
    if stem.startswith("checkpoint_step_"):
        return stem[len("checkpoint_step_") :]
    return stem


def _write_grid(*, images: list[Path], output_path: Path, columns: int) -> None:
    if not images:
        raise ValueError("No images provided for grid")
    columns = max(1, columns)
    opened = [Image.open(path).convert("RGB") for path in images]
    try:
        width, height = opened[0].size
        rows = (len(opened) + columns - 1) // columns
        canvas = Image.new("RGB", (columns * width, rows * height))
        for index, image in enumerate(opened):
            row = index // columns
            col = index % columns
            canvas.paste(image, box=(col * width, row * height))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
    finally:
        for image in opened:
            image.close()


if __name__ == "__main__":
    main()
