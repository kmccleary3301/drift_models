import torch
from torch import nn

from drifting_models.features import HookedFeatureAdapter, HookedFeatureAdapterConfig, OutputAdapter


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x


def test_hooked_feature_adapter_returns_requested_nodes() -> None:
    backbone = TinyBackbone()
    adapter = HookedFeatureAdapter(
        backbone=backbone,
        config=HookedFeatureAdapterConfig(return_nodes=("conv1", "conv2")),
    )
    images = torch.randn(2, 4, 16, 16)
    features = adapter(images)
    assert len(features) == 2
    assert features[0].shape == (2, 8, 16, 16)
    assert features[1].shape == (2, 16, 8, 8)
    adapter.close()


def test_output_adapter_wraps_tensor_output() -> None:
    backbone = TinyBackbone()
    adapter = OutputAdapter(backbone)
    images = torch.randn(3, 4, 16, 16)
    outputs = adapter(images)
    assert len(outputs) == 1
    assert outputs[0].shape == (3, 16, 8, 8)
