from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main() -> None:
    args = _parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    output_root.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args.seed))
    sigma = float(args.sigma)
    if sigma < 0.0:
        raise ValueError("--sigma must be >= 0")

    count = 0
    for src_path, rel_path in _iter_image_paths(input_root):
        if args.max_images is not None and count >= int(args.max_images):
            break
        dst_path = output_root / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists() and not args.overwrite:
            count += 1
            continue
        image = Image.open(src_path).convert("RGB")
        array = np.array(image, dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(array).float().div(255.0)
        noise = torch.randn_like(tensor) * sigma
        noisy = torch.clamp(tensor + noise, 0.0, 1.0)
        out = (noisy.mul(255.0).round().to(torch.uint8)).cpu().numpy()
        Image.fromarray(out, mode="RGB").save(dst_path)
        count += 1


def _iter_image_paths(root: Path) -> list[tuple[Path, Path]]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    pairs: list[tuple[Path, Path]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        rel = path.relative_to(root)
        pairs.append((path, rel))
    return pairs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a noisy copy of an ImageFolder dataset")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
