from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from torchvision.datasets import CIFAR10
from PIL import Image


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root)
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_cache = Path(args.dataset_cache)
    dataset = CIFAR10(root=str(dataset_cache), train=(args.split == "train"), download=bool(args.download))
    limit = len(dataset) if args.max_images is None else min(int(args.max_images), len(dataset))

    for idx in range(limit):
        image, label = dataset[idx]
        if args.image_size is not None:
            target = int(args.image_size)
            if image.size != (target, target):
                image = image.resize((target, target), resample=Image.BICUBIC)
        class_dir = output_root / str(int(label))
        class_dir.mkdir(parents=True, exist_ok=True)
        image.save(class_dir / f"{idx:06d}.png")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CIFAR-10 to ImageFolder layout")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None, help="Optional resize to square image size")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--dataset-cache", type=str, default="outputs/datasets/_torchvision_cache")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
