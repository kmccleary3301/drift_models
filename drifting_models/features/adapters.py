from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class HookedFeatureAdapterConfig:
    return_nodes: tuple[str, ...]


class HookedFeatureAdapter(nn.Module):
    def __init__(self, backbone: nn.Module, config: HookedFeatureAdapterConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        self._node_outputs: dict[str, torch.Tensor] = {}
        self._handles = []
        name_to_module = dict(self.backbone.named_modules())
        for node_name in self.config.return_nodes:
            if node_name not in name_to_module:
                raise ValueError(f"return node '{node_name}' not found in backbone")
            module = name_to_module[node_name]
            handle = module.register_forward_hook(self._make_hook(node_name))
            self._handles.append(handle)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        self._node_outputs.clear()
        _ = self.backbone(images)
        missing = [node for node in self.config.return_nodes if node not in self._node_outputs]
        if missing:
            raise RuntimeError(f"Missing hooked outputs for nodes: {missing}")
        outputs = [self._node_outputs[node] for node in self.config.return_nodes]
        for output in outputs:
            if output.ndim != 4:
                raise ValueError("Hooked features must be [B, C, H, W]")
        return outputs

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, node_name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self._node_outputs[node_name] = output

        return hook


class OutputAdapter(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        output = self.backbone(images)
        if isinstance(output, torch.Tensor):
            if output.ndim != 4:
                raise ValueError("Output tensor must be [B, C, H, W]")
            return [output]
        if isinstance(output, (list, tuple)):
            tensors = [tensor for tensor in output if isinstance(tensor, torch.Tensor)]
            for tensor in tensors:
                if tensor.ndim != 4:
                    raise ValueError("Output tensors must be [B, C, H, W]")
            return tensors
        raise ValueError("Unsupported backbone output type")
