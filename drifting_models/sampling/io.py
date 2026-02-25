from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def write_imagefolder(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    root: Path,
    start_index: int = 0,
    image_format: str = "png",
    overwrite: bool = False,
) -> int:
    if images.ndim != 4:
        raise ValueError(f"images must be [N, C, H, W], got {tuple(images.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be [N], got {tuple(labels.shape)}")
    if images.shape[0] != labels.shape[0]:
        raise ValueError("images and labels must share batch dimension")
    if images.shape[1] not in (1, 3):
        raise ValueError(f"Expected 1 or 3 channels, got {images.shape[1]}")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    fmt = image_format.lower().lstrip(".")
    if fmt not in {"png", "jpg", "jpeg"}:
        raise ValueError(f"Unsupported image_format: {image_format}")

    root.mkdir(parents=True, exist_ok=True)
    saved = 0
    for idx in range(images.shape[0]):
        label = int(labels[idx].item())
        class_dir = root / str(label)
        class_dir.mkdir(parents=True, exist_ok=True)
        image_index = start_index + idx
        out_path = class_dir / f"{image_index:06d}.{fmt}"
        if out_path.exists() and not overwrite:
            continue
        pil = _to_pil(images[idx])
        pil.save(out_path)
        saved += 1
    return saved


def _to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim != 3:
        raise ValueError(f"image must be [C, H, W], got {tuple(image.shape)}")
    values = image.detach().cpu().float()
    values = values.clamp(0.0, 1.0)
    values = (values * 255.0).round().to(torch.uint8)
    if values.shape[0] == 1:
        array = values[0].numpy()
        return Image.fromarray(array, mode="L")
    array = values.permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")

