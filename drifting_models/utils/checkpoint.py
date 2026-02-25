from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "cpu_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    cpu_state = state.get("cpu_rng_state")
    if cpu_state is not None:
        torch.set_rng_state(_coerce_rng_state_tensor(cpu_state, to_device="cpu"))
    cuda_state_all = state.get("cuda_rng_state_all")
    if cuda_state_all is not None and torch.cuda.is_available():
        if not isinstance(cuda_state_all, list):
            raise TypeError("cuda_rng_state_all must be a list of RNG state tensors")
        coerced = [_coerce_rng_state_tensor(item, to_device="cpu") for item in cuda_state_all]
        device_count = int(torch.cuda.device_count())
        for idx in range(min(device_count, len(coerced))):
            torch.cuda.set_rng_state(coerced[idx], idx)



def _coerce_rng_state_tensor(value: Any, *, to_device: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=to_device)
        if tensor.dtype != torch.uint8:
            tensor = tensor.to(dtype=torch.uint8)
        return tensor.contiguous()
    if isinstance(value, (bytes, bytearray)):
        return torch.tensor(list(value), dtype=torch.uint8, device=to_device)
    if isinstance(value, list):
        return torch.tensor(value, dtype=torch.uint8, device=to_device)
    raise TypeError(f"Unsupported RNG state type: {type(value).__name__}")


def save_training_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    extra: dict[str, Any] | None = None,
    queue_state: dict[str, Any] | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    ema_state_dict: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
    }
    if extra is not None:
        payload["extra"] = extra
    if queue_state is not None:
        payload["queue_state"] = queue_state
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if ema_state_dict is not None:
        payload["ema_state_dict"] = ema_state_dict
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_training_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in payload:
        scaler.load_state_dict(payload["scaler_state_dict"])
    rng_state = payload.get("rng_state")
    if isinstance(rng_state, dict):
        restore_rng_state(rng_state)
    return payload
