from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    baseline_dir = source_root / args.baseline_ablation
    compare_dir = source_root / args.compare_ablation

    if not baseline_dir.exists():
        raise FileNotFoundError(f"Missing baseline directory: {baseline_dir}")
    if not compare_dir.exists():
        raise FileNotFoundError(f"Missing compare directory: {compare_dir}")

    baseline_images = index_step_images(baseline_dir)
    compare_images = index_step_images(compare_dir)
    common_steps = sorted(set(baseline_images).intersection(compare_images))
    if not common_steps:
        raise RuntimeError("No shared checkpoint scatter images found between the two ablations.")

    output_root.mkdir(parents=True, exist_ok=True)
    panels: list[dict[str, object]] = []
    for step in common_steps:
        panel_path = output_root / f"step_{step:06d}_panel.png"
        build_panel(
            left_path=baseline_images[step],
            right_path=compare_images[step],
            left_label=args.baseline_ablation,
            right_label=args.compare_ablation,
            title=f"step {step}",
            output_path=panel_path,
        )
        panels.append(
            {
                "step": step,
                "baseline_scatter": str(baseline_images[step]),
                "compare_scatter": str(compare_images[step]),
                "panel_path": str(panel_path),
            }
        )

    summary = {
        "source_root": str(source_root),
        "baseline_ablation": args.baseline_ablation,
        "compare_ablation": args.compare_ablation,
        "panel_count": len(panels),
        "panels": panels,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "panel_count": len(panels)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build side-by-side toy trajectory panels from two ablation folders.")
    parser.add_argument("--source-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--baseline-ablation", type=str, default="baseline")
    parser.add_argument("--compare-ablation", type=str, default="attraction_only")
    return parser.parse_args()


def index_step_images(ablation_dir: Path) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for path in sorted(ablation_dir.glob("step_*_scatter.png")):
        parts = path.stem.split("_")
        if len(parts) < 2:
            continue
        try:
            step = int(parts[1])
        except ValueError:
            continue
        mapping[step] = path
    return mapping


def build_panel(
    *,
    left_path: Path,
    right_path: Path,
    left_label: str,
    right_label: str,
    title: str,
    output_path: Path,
) -> None:
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    width = left.width + right.width
    height = max(left.height, right.height) + 56
    panel = Image.new("RGB", (width, height), (16, 18, 24))
    panel.paste(left, (0, 56))
    panel.paste(right, (left.width, 56))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 12), title, fill=(235, 235, 235))
    draw.text((12, 34), left_label, fill=(120, 200, 255))
    draw.text((left.width + 12, 34), right_label, fill=(255, 170, 120))
    panel.save(output_path)


if __name__ == "__main__":
    main()
