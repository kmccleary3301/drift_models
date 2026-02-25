from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from PIL import Image, ImageDraw

from drifting_models.drift_field import DriftFieldConfig
from drifting_models.drift_loss import DriftingLossConfig, drifting_stopgrad_loss
from drifting_models.utils.runtime import resolve_device, seed_everything


@dataclass(frozen=True)
class ToyTrainConfig:
    seed: int = 1337
    latent_dim: int = 2
    data_dim: int = 2
    hidden_dim: int = 128
    depth: int = 3
    train_steps: int = 5000
    batch_size: int = 512
    learning_rate: float = 1e-4
    temperature: float = 0.05
    log_every: int = 200
    sample_count_eval: int = 4096
    target_std: float = 0.35
    center_a_x: float = -2.5
    center_a_y: float = 0.0
    center_b_x: float = 2.5
    center_b_y: float = 0.0
    normalize_over_x: bool = True
    mask_self_negatives: bool = True


@dataclass(frozen=True)
class AblationConfig:
    name: str
    attraction_scale: float
    repulsion_scale: float


def default_ablation_suite() -> list[AblationConfig]:
    return [
        AblationConfig(name="baseline", attraction_scale=1.0, repulsion_scale=1.0),
        AblationConfig(name="attraction_1_5x", attraction_scale=1.5, repulsion_scale=1.0),
        AblationConfig(name="repulsion_1_5x", attraction_scale=1.0, repulsion_scale=1.5),
        AblationConfig(name="attraction_2_0x", attraction_scale=2.0, repulsion_scale=1.0),
        AblationConfig(name="repulsion_2_0x", attraction_scale=1.0, repulsion_scale=2.0),
        AblationConfig(name="attraction_only", attraction_scale=1.0, repulsion_scale=0.0),
    ]


class ToyGenerator(nn.Module):
    def __init__(self, latent_dim: int, data_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)


def run_toy_ablation(
    *,
    config: ToyTrainConfig,
    ablation: AblationConfig,
    device: torch.device,
    output_dir: Path | None = None,
) -> dict[str, object]:
    _set_seed(config.seed)
    generator = ToyGenerator(
        latent_dim=config.latent_dim,
        data_dim=config.data_dim,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
    ).to(device)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=config.learning_rate)
    drift_field = DriftFieldConfig(
        temperature=config.temperature,
        normalize_over_x=config.normalize_over_x,
        mask_self_negatives=config.mask_self_negatives,
    )
    loss_config = DriftingLossConfig(
        drift_field=drift_field,
        attraction_scale=ablation.attraction_scale,
        repulsion_scale=ablation.repulsion_scale,
        stopgrad_target=True,
    )
    fixed_eval_noise = torch.randn(config.sample_count_eval, config.latent_dim, device=device)
    centers = _target_centers(config, device=device)

    history: list[dict[str, float]] = []
    artifact_dir = None
    if output_dir is not None:
        artifact_dir = output_dir / ablation.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
    for step in range(config.train_steps):
        noise = torch.randn(config.batch_size, config.latent_dim, device=device)
        generated = generator(noise)
        positives = sample_target_distribution(config, config.batch_size, device=device)

        loss, drift, stats = drifting_stopgrad_loss(
            generated,
            positives,
            generated,
            config=loss_config,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        should_log = (step + 1) % config.log_every == 0 or step == 0 or step + 1 == config.train_steps
        if should_log:
            with torch.no_grad():
                eval_samples = generator(fixed_eval_noise)
                eval_metrics = evaluate_toy_samples(eval_samples, centers)
            if artifact_dir is not None:
                _save_snapshot_artifacts(
                    artifact_dir=artifact_dir,
                    step=step + 1,
                    samples=eval_samples.detach().cpu(),
                    centers=centers.detach().cpu(),
                )
            history.append(
                {
                    "step": float(step + 1),
                    "loss": float(stats["loss"]),
                    "drift_norm": float(stats["drift_norm"]),
                    "coverage_a": float(eval_metrics["coverage_a"]),
                    "coverage_b": float(eval_metrics["coverage_b"]),
                    "center_distance": float(eval_metrics["center_distance"]),
                    "mean_distance_to_target": float(eval_metrics["mean_distance_to_target"]),
                    "mode_balance_error": float(eval_metrics["mode_balance_error"]),
                }
            )

    with torch.no_grad():
        final_samples = generator(fixed_eval_noise)
        final_metrics = evaluate_toy_samples(final_samples, centers)
    if artifact_dir is not None:
        _save_snapshot_artifacts(
            artifact_dir=artifact_dir,
            step=config.train_steps,
            samples=final_samples.detach().cpu(),
            centers=centers.detach().cpu(),
            suffix="final",
        )
    return {
        "ablation": ablation.name,
        "attraction_scale": ablation.attraction_scale,
        "repulsion_scale": ablation.repulsion_scale,
        "history": history,
        "final_metrics": final_metrics,
    }


def run_toy_suite(
    *,
    config: ToyTrainConfig,
    ablations: list[AblationConfig],
    output_dir: Path | None = None,
    device: torch.device | None = None,
) -> dict[str, object]:
    resolved_device = device or resolve_device("auto")
    if output_dir is not None:
        results = [
            run_toy_ablation(
                config=config,
                ablation=ablation,
                device=resolved_device,
                output_dir=output_dir,
            )
            for ablation in ablations
        ]
    else:
        results = [
            run_toy_ablation(
                config=config,
                ablation=ablation,
                device=resolved_device,
                output_dir=None,
            )
            for ablation in ablations
        ]
    summary = {
        "device": str(resolved_device),
        "config": config.__dict__,
        "results": results,
        "ranking_by_mean_distance": sorted(
            (
                {
                    "ablation": entry["ablation"],
                    "mean_distance_to_target": entry["final_metrics"]["mean_distance_to_target"],
                }
                for entry in results
            ),
            key=lambda item: item["mean_distance_to_target"],
        ),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "toy_results.json"
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_ablation_table_markdown(output_dir=output_dir, summary=summary)
    return summary


def evaluate_toy_samples(samples: torch.Tensor, centers: torch.Tensor) -> dict[str, float]:
    distances = torch.cdist(samples, centers)
    nearest = distances.argmin(dim=1)
    center_distance = torch.linalg.vector_norm(samples.mean(dim=0) - centers.mean(dim=0)).item()
    mean_distance_to_target = distances.min(dim=1).values.mean().item()
    coverage_a = (nearest == 0).float().mean().item()
    coverage_b = (nearest == 1).float().mean().item()
    mode_balance_error = abs(coverage_a - 0.5) + abs(coverage_b - 0.5)
    return {
        "coverage_a": float(coverage_a),
        "coverage_b": float(coverage_b),
        "center_distance": float(center_distance),
        "mean_distance_to_target": float(mean_distance_to_target),
        "mode_balance_error": float(mode_balance_error),
    }


def sample_target_distribution(config: ToyTrainConfig, count: int, *, device: torch.device) -> torch.Tensor:
    centers = _target_centers(config, device=device)
    selector = torch.randint(0, 2, (count,), device=device)
    selected_centers = centers[selector]
    noise = torch.randn(count, config.data_dim, device=device) * config.target_std
    return selected_centers + noise


def parse_simple_yaml_config(path: Path) -> ToyTrainConfig:
    entries: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid config line: {raw_line}")
        key, value = line.split(":", 1)
        entries[key.strip()] = value.strip()
    return ToyTrainConfig(
        seed=int(entries.get("seed", ToyTrainConfig.seed)),
        latent_dim=int(entries.get("latent_dim", ToyTrainConfig.latent_dim)),
        data_dim=int(entries.get("data_dim", ToyTrainConfig.data_dim)),
        hidden_dim=int(entries.get("hidden_dim", ToyTrainConfig.hidden_dim)),
        depth=int(entries.get("depth", ToyTrainConfig.depth)),
        train_steps=int(entries.get("train_steps", ToyTrainConfig.train_steps)),
        batch_size=int(entries.get("batch_size", ToyTrainConfig.batch_size)),
        learning_rate=float(entries.get("learning_rate", ToyTrainConfig.learning_rate)),
        temperature=float(entries.get("temperature", ToyTrainConfig.temperature)),
        log_every=int(entries.get("log_every", ToyTrainConfig.log_every)),
        sample_count_eval=int(entries.get("sample_count_eval", ToyTrainConfig.sample_count_eval)),
        target_std=float(entries.get("target_std", ToyTrainConfig.target_std)),
        center_a_x=float(entries.get("center_a_x", ToyTrainConfig.center_a_x)),
        center_a_y=float(entries.get("center_a_y", ToyTrainConfig.center_a_y)),
        center_b_x=float(entries.get("center_b_x", ToyTrainConfig.center_b_x)),
        center_b_y=float(entries.get("center_b_y", ToyTrainConfig.center_b_y)),
        normalize_over_x=_parse_bool(entries.get("normalize_over_x"), ToyTrainConfig.normalize_over_x),
        mask_self_negatives=_parse_bool(entries.get("mask_self_negatives"), ToyTrainConfig.mask_self_negatives),
    )


def _target_centers(config: ToyTrainConfig, *, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            [config.center_a_x, config.center_a_y],
            [config.center_b_x, config.center_b_y],
        ],
        device=device,
        dtype=torch.float32,
    )


def _set_seed(seed: int) -> None:
    seed_everything(seed)


def _parse_bool(value: str | None, fallback: bool) -> bool:
    if value is None:
        return fallback
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _save_snapshot_artifacts(
    *,
    artifact_dir: Path,
    step: int,
    samples: torch.Tensor,
    centers: torch.Tensor,
    suffix: str | None = None,
) -> None:
    stem = f"step_{step:06d}"
    if suffix is not None:
        stem = f"{stem}_{suffix}"
    sample_path = artifact_dir / f"{stem}_samples.pt"
    image_path = artifact_dir / f"{stem}_scatter.png"
    torch.save(samples, sample_path)
    image = _render_scatter_image(samples=samples, centers=centers, size=512)
    image.save(image_path)


def _render_scatter_image(samples: torch.Tensor, centers: torch.Tensor, *, size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), (10, 12, 16))
    draw = ImageDraw.Draw(image)
    all_points = torch.cat([samples, centers], dim=0)
    min_xy = all_points.min(dim=0).values
    max_xy = all_points.max(dim=0).values
    span = torch.clamp(max_xy - min_xy, min=1e-5)

    def map_point(point: torch.Tensor) -> tuple[int, int]:
        x_norm = float((point[0] - min_xy[0]) / span[0])
        y_norm = float((point[1] - min_xy[1]) / span[1])
        x = int(20 + x_norm * (size - 40))
        y = int(20 + (1.0 - y_norm) * (size - 40))
        return x, y

    for point in samples:
        x, y = map_point(point)
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 140, 40))

    center_colors = [(90, 180, 255), (120, 240, 120)]
    for index, center in enumerate(centers):
        x, y = map_point(center)
        color = center_colors[index % len(center_colors)]
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=2)

    return image


def _write_ablation_table_markdown(*, output_dir: Path, summary: dict[str, object]) -> None:
    rows = ["| ablation | mean_distance_to_target | mode_balance_error |", "| --- | ---: | ---: |"]
    for result in summary["results"]:
        metrics = result["final_metrics"]
        rows.append(
            f"| {result['ablation']} | "
            f"{metrics['mean_distance_to_target']:.6f} | {metrics['mode_balance_error']:.6f} |"
        )
    table = "# Toy Ablation Summary\n\n" + "\n".join(rows) + "\n"
    (output_dir / "toy_ablation_table.md").write_text(table, encoding="utf-8")
