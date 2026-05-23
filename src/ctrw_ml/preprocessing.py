from __future__ import annotations

import numpy as np


def fill_missing_by_interpolation(values: np.ndarray) -> np.ndarray:
    observed = np.isfinite(values)
    if observed.sum() == 0:
        return np.zeros_like(values)
    if observed.sum() == 1:
        return np.full_like(values, values[observed][0])

    idx = np.arange(values.size)
    filled = values.copy()
    filled[~observed] = np.interp(idx[~observed], idx[observed], values[observed])
    return filled


def zscore(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    return (values - mean) / (std + eps)


def condition_trajectory(
    trajectory: np.ndarray,
    rng: np.random.Generator,
    noise_sigma: float = 0.05,
    dropout_prob: float = 0.10,
) -> np.ndarray:
    """Apply frame dropout, interpolation, zero-mean/unit-variance scaling, and noise."""
    x = trajectory.astype(np.float32).copy()

    if dropout_prob > 0:
        missing = rng.uniform(size=x.size) < dropout_prob
        missing[0] = False
        missing[-1] = False
        x[missing] = np.nan
        x = fill_missing_by_interpolation(x)

    x = zscore(x).astype(np.float32)

    if noise_sigma > 0:
        x = x + rng.normal(0.0, noise_sigma, size=x.shape).astype(np.float32)

    return x.astype(np.float32)
