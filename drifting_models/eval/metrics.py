from __future__ import annotations

import math

import torch


def gaussian_statistics(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2:
        raise ValueError(f"features must be rank-2 [N, D], got {tuple(features.shape)}")
    if features.shape[0] < 2:
        raise ValueError("features must contain at least 2 samples")
    values = features.to(dtype=torch.float64)
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.t().matmul(centered) / float(values.shape[0] - 1)
    return mean, covariance


def frechet_distance(
    mean_a: torch.Tensor,
    cov_a: torch.Tensor,
    mean_b: torch.Tensor,
    cov_b: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> float:
    if mean_a.shape != mean_b.shape:
        raise ValueError(f"Mean shape mismatch: {tuple(mean_a.shape)} vs {tuple(mean_b.shape)}")
    if cov_a.shape != cov_b.shape:
        raise ValueError(f"Covariance shape mismatch: {tuple(cov_a.shape)} vs {tuple(cov_b.shape)}")
    if cov_a.ndim != 2 or cov_a.shape[0] != cov_a.shape[1]:
        raise ValueError(f"cov_a must be square matrix, got {tuple(cov_a.shape)}")

    mean_a64 = mean_a.to(dtype=torch.float64)
    mean_b64 = mean_b.to(dtype=torch.float64)
    cov_a64 = _symmetrize(cov_a.to(dtype=torch.float64))
    cov_b64 = _symmetrize(cov_b.to(dtype=torch.float64))

    mean_delta = mean_a64 - mean_b64
    mean_term = float(mean_delta.dot(mean_delta).item())
    trace_term = float(torch.trace(cov_a64 + cov_b64).item())

    sqrt_cov_a = _matrix_sqrt_psd(cov_a64, eps=eps)
    middle = sqrt_cov_a.matmul(cov_b64).matmul(sqrt_cov_a)
    covmean_trace = float(torch.trace(_matrix_sqrt_psd(middle, eps=eps)).item())
    fid = mean_term + trace_term - (2.0 * covmean_trace)
    return max(0.0, float(fid))


def inception_score_from_logits(
    logits: torch.Tensor,
    *,
    splits: int = 10,
    eps: float = 1e-8,
) -> tuple[float, float]:
    if logits.ndim != 2:
        raise ValueError(f"logits must be rank-2 [N, C], got {tuple(logits.shape)}")
    total = int(logits.shape[0])
    if total < 2:
        raise ValueError("logits must contain at least 2 samples")
    split_count = max(1, min(splits, total))
    probs = torch.softmax(logits.to(dtype=torch.float64), dim=1).clamp_min(eps)
    split_scores: list[float] = []
    for split in torch.tensor_split(probs, split_count, dim=0):
        if split.numel() == 0:
            continue
        marginal = split.mean(dim=0, keepdim=True).clamp_min(eps)
        kl = split * (split.log() - marginal.log())
        score = torch.exp(kl.sum(dim=1).mean())
        split_scores.append(float(score.item()))
    if not split_scores:
        raise ValueError("No non-empty splits available for inception score")
    if len(split_scores) == 1:
        return split_scores[0], 0.0
    mean = sum(split_scores) / len(split_scores)
    variance = sum((value - mean) ** 2 for value in split_scores) / len(split_scores)
    return float(mean), float(math.sqrt(variance))


def _matrix_sqrt_psd(matrix: torch.Tensor, *, eps: float) -> torch.Tensor:
    sym = _symmetrize(matrix)
    evals, evecs = torch.linalg.eigh(sym)
    clamped = torch.clamp(evals, min=eps)
    sqrt_vals = torch.sqrt(clamped)
    return (evecs * sqrt_vals.unsqueeze(0)).matmul(evecs.t())


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.t())
