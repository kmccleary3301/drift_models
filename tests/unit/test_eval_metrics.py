import torch

from drifting_models.eval import frechet_distance, gaussian_statistics, inception_score_from_logits


def test_frechet_distance_zero_for_identical_stats() -> None:
    features = torch.randn(64, 8)
    mean, cov = gaussian_statistics(features)
    fid = frechet_distance(mean, cov, mean, cov)
    assert abs(fid) < 1e-10


def test_frechet_distance_increases_with_mean_shift() -> None:
    base = torch.randn(64, 8)
    shifted = base + 0.5
    mean_a, cov_a = gaussian_statistics(base)
    mean_b, cov_b = gaussian_statistics(shifted)
    fid = frechet_distance(mean_a, cov_a, mean_b, cov_b)
    assert fid > 0.0


def test_inception_score_uniform_logits_near_one() -> None:
    logits = torch.zeros(128, 10)
    score_mean, score_std = inception_score_from_logits(logits, splits=8)
    assert score_mean == 1.0
    assert score_std == 0.0


def test_inception_score_peaked_logits_above_one() -> None:
    logits = torch.zeros(128, 5)
    logits[:, 0] = 10.0
    logits[::2, 1] = 9.0
    score_mean, _score_std = inception_score_from_logits(logits, splits=8)
    assert score_mean > 1.0
