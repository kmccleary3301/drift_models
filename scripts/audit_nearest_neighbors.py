from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import v2

from drifting_models.utils import add_device_argument, resolve_device, write_json


def main() -> None:
    args = _parse_args()
    device = resolve_device(args.device)
    feature_model = _build_feature_model(device=device)
    transform = _build_transform()

    gen_dataset = ImageFolder(root=str(Path(args.generated_root)), transform=transform)
    ref_dataset = ImageFolder(root=str(Path(args.reference_root)), transform=transform)
    if len(gen_dataset) == 0:
        raise ValueError("Generated dataset is empty")
    if len(ref_dataset) == 0:
        raise ValueError("Reference dataset is empty")

    gen_samples = _subset_samples(gen_dataset.samples, int(args.max_generated))
    ref_samples = _subset_samples(ref_dataset.samples, int(args.max_reference))
    gen_features = _extract_features(
        samples=gen_samples,
        feature_model=feature_model,
        transform=transform,
        batch_size=int(args.batch_size),
        device=device,
    )
    ref_features = _extract_features(
        samples=ref_samples,
        feature_model=feature_model,
        transform=transform,
        batch_size=int(args.batch_size),
        device=device,
    )
    gen_features = torch.nn.functional.normalize(gen_features, dim=-1)
    ref_features = torch.nn.functional.normalize(ref_features, dim=-1)

    similarity = gen_features @ ref_features.T
    best_values, best_indices = torch.max(similarity, dim=1)

    pairs: list[dict[str, object]] = []
    for index in range(similarity.shape[0]):
        ref_index = int(best_indices[index].item())
        gen_path, gen_label = gen_samples[index]
        ref_path, ref_label = ref_samples[ref_index]
        pairs.append(
            {
                "generated_path": gen_path,
                "generated_label": int(gen_label),
                "nearest_reference_path": ref_path,
                "nearest_reference_label": int(ref_label),
                "cosine_similarity": float(best_values[index].item()),
                "label_match": bool(int(gen_label) == int(ref_label)),
            }
        )

    summary = {
        "generated_root": args.generated_root,
        "reference_root": args.reference_root,
        "generated_count": len(gen_samples),
        "reference_count": len(ref_samples),
        "mean_cosine_similarity": float(best_values.mean().item()),
        "median_cosine_similarity": float(best_values.median().item()),
        "p95_cosine_similarity": float(torch.quantile(best_values, 0.95).item()),
        "label_match_rate": float(sum(1 for entry in pairs if entry["label_match"]) / max(1, len(pairs))),
        "pairs": pairs,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, summary)
    print(json.dumps({"output_path": str(output_path), "pairs": len(pairs)}, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nearest-neighbor audit between generated and reference ImageFolder sets")
    parser.add_argument("--generated-root", type=str, required=True)
    parser.add_argument("--reference-root", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--max-generated", type=int, default=256)
    parser.add_argument("--max-reference", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    add_device_argument(parser, default="auto")
    return parser.parse_args()


def _build_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _build_feature_model(*, device: torch.device) -> torch.nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    model.fc = torch.nn.Identity()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _subset_samples(samples: list[tuple[str, int]], limit: int) -> list[tuple[str, int]]:
    if limit <= 0 or limit >= len(samples):
        return list(samples)
    return list(samples[:limit])


def _extract_features(
    *,
    samples: list[tuple[str, int]],
    feature_model: torch.nn.Module,
    transform: v2.Compose,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    dataset = _PathImageDataset(samples=samples, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    outputs = []
    with torch.no_grad():
        for image_batch in loader:
            output = feature_model(image_batch.to(device))
            outputs.append(output.detach().cpu())
    return torch.cat(outputs, dim=0)


class _PathImageDataset(Dataset[torch.Tensor]):
    def __init__(self, *, samples: list[tuple[str, int]], transform: v2.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        from torchvision.io import read_image

        path = self.samples[index][0]
        image = read_image(path)
        return self.transform(image)


if __name__ == "__main__":
    main()
