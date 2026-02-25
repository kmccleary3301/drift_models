import torch

from drifting_models.sampling import postprocess_images


def test_postprocess_clamp_0_1_bounded() -> None:
    x = torch.tensor([[-2.0, 0.5, 2.0]]).view(1, 1, 1, 3)
    y = postprocess_images(x, mode="clamp_0_1")
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 1.0


def test_postprocess_tanh_to_0_1_bounded() -> None:
    x = torch.randn(2, 3, 4, 4) * 10.0
    y = postprocess_images(x, mode="tanh_to_0_1")
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 1.0


def test_postprocess_sigmoid_bounded() -> None:
    x = torch.randn(2, 3, 4, 4) * 10.0
    y = postprocess_images(x, mode="sigmoid")
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 1.0


def test_postprocess_identity_passthrough() -> None:
    x = torch.randn(2, 3, 4, 4)
    y = postprocess_images(x, mode="identity")
    assert torch.allclose(x.float(), y)

