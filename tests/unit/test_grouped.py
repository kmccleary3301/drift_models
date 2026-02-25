import torch

from drifting_models.drift_field import DriftFieldConfig
from drifting_models.train import compute_grouped_v, flatten_grouped, infer_grouped_shapes


def test_grouped_compute_v_shape() -> None:
    config = DriftFieldConfig(temperature=0.05, normalize_over_x=True, mask_self_negatives=False)
    x_grouped = torch.randn(4, 8, 3)
    y_pos_grouped = torch.randn(4, 10, 3)
    y_neg_grouped = torch.randn(4, 8, 3)

    grouped_v = compute_grouped_v(x_grouped, y_pos_grouped, y_neg_grouped, config=config)
    assert grouped_v.shape == x_grouped.shape


def test_grouped_shape_inference_and_flatten() -> None:
    x_grouped = torch.randn(3, 5, 2)
    y_pos_grouped = torch.randn(3, 7, 2)
    y_neg_grouped = torch.randn(3, 5, 2)
    shapes = infer_grouped_shapes(x_grouped, y_pos_grouped, y_neg_grouped)

    assert shapes.groups == 3
    assert shapes.negatives_per_group == 5
    assert shapes.positives_per_group == 7
    assert shapes.feature_dim == 2
    flattened = flatten_grouped(x_grouped)
    assert flattened.shape == (15, 2)
