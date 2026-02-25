from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelEMA:
    decay: float
    shadow: dict[str, torch.Tensor]

    @classmethod
    def create(cls, *, model: nn.Module, decay: float) -> "ModelEMA":
        if not (0.0 < decay < 1.0):
            raise ValueError("decay must be in (0, 1)")
        shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        return cls(decay=decay, shadow=shadow)

    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=(1.0 - self.decay))

    def copy_to(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    parameter.copy_(self.shadow[name])

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        return {
            "decay": float(self.decay),
            "shadow": {name: value.clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: dict[str, torch.Tensor | float]) -> None:
        decay_value = state.get("decay")
        shadow_value = state.get("shadow")
        if not isinstance(decay_value, float):
            raise ValueError("EMA state missing float 'decay'")
        if not isinstance(shadow_value, dict):
            raise ValueError("EMA state missing dict 'shadow'")
        self.decay = decay_value
        self.shadow = {name: tensor.clone() for name, tensor in shadow_value.items() if isinstance(tensor, torch.Tensor)}
