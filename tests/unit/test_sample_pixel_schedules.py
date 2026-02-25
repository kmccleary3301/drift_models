import torch

from scripts.sample_pixel import _build_alpha


class _Args:
    def __init__(self) -> None:
        self.n_samples = 8
        self.alpha = 2.0
        self.alpha_schedule = "constant"
        self.alpha_start = None
        self.alpha_end = None
        self.alpha_values = []


def test_alpha_constant_schedule() -> None:
    args = _Args()
    device = torch.device("cpu")
    global_indices = torch.arange(0, 4, device=device)
    alpha = _build_alpha(args, batch=4, device=device, global_indices=global_indices)
    assert torch.allclose(alpha, torch.full((4,), 2.0))


def test_alpha_linear_schedule_endpoints() -> None:
    args = _Args()
    args.alpha_schedule = "linear"
    args.alpha_start = 1.0
    args.alpha_end = 5.0
    device = torch.device("cpu")
    global_indices = torch.tensor([0, 7], device=device, dtype=torch.long)
    alpha = _build_alpha(args, batch=2, device=device, global_indices=global_indices)
    assert torch.allclose(alpha, torch.tensor([1.0, 5.0]))


def test_alpha_list_schedule_cycles() -> None:
    args = _Args()
    args.alpha_schedule = "list"
    args.alpha_values = [1.0, 3.0]
    device = torch.device("cpu")
    global_indices = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.long)
    alpha = _build_alpha(args, batch=4, device=device, global_indices=global_indices)
    assert torch.allclose(alpha, torch.tensor([1.0, 3.0, 1.0, 3.0]))

