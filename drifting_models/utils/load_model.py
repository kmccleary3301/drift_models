from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import torch
from torch import nn

from drifting_models.models import DiTLikeConfig, DiTLikeGenerator


def load_pixel_generator_from_checkpoint(
    *,
    checkpoint_path: Path,
    device: torch.device,
    override_config: DiTLikeConfig | None = None,
    strict: bool = True,
) -> DiTLikeGenerator:
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_model_state_dict(payload)
    config = override_config if override_config is not None else _extract_dit_config(payload)
    model = DiTLikeGenerator(config).to(device)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model


def _extract_model_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
        if not isinstance(state, dict):
            raise ValueError("model_state_dict is not a dict")
        return state
    if isinstance(payload, dict) and all(isinstance(k, str) for k in payload.keys()):
        tensor_values = [v for v in payload.values() if isinstance(v, torch.Tensor)]
        if tensor_values and len(tensor_values) == len(payload):
            return payload  # raw state_dict
    raise ValueError("Unsupported checkpoint payload; expected training checkpoint or raw state_dict")


def _extract_dit_config(payload: object) -> DiTLikeConfig:
    if isinstance(payload, dict) and "extra" in payload and isinstance(payload["extra"], dict):
        extra = payload["extra"]
        if "model_config" in extra and isinstance(extra["model_config"], dict):
            return _coerce_dit_config(extra["model_config"])
    raise ValueError("Checkpoint does not include extra.model_config; pass override_config")


def _coerce_dit_config(raw: dict[str, object]) -> DiTLikeConfig:
    allowed = {field.name for field in fields(DiTLikeConfig)}
    values: dict[str, object] = {}
    for key, value in raw.items():
        if key not in allowed:
            continue
        values[key] = value
    return DiTLikeConfig(**values)  # type: ignore[arg-type]

