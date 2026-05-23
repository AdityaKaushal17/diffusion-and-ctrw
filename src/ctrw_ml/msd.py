from __future__ import annotations

import numpy as np


def time_averaged_msd(x: np.ndarray, max_lag: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if max_lag is None:
        max_lag = max(4, x.size // 4)
    lags = np.arange(1, max_lag + 1)
    msd = np.array([np.mean((x[lag:] - x[:-lag]) ** 2) for lag in lags], dtype=np.float64)
    return lags, msd


def estimate_msd_slope(x: np.ndarray) -> float:
    lags, msd = time_averaged_msd(x)
    valid = msd > 0
    if valid.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(lags[valid]), np.log(msd[valid]), deg=1)
    return float(slope)


def ensemble_msd_by_alpha_bin(
    x: np.ndarray,
    lengths: np.ndarray,
    alpha: np.ndarray,
    n_bins: int = 5,
) -> list[dict[str, np.ndarray | float]]:
    bins = np.linspace(float(alpha.min()), float(alpha.max()), n_bins + 1)
    results = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (alpha >= lo) & (alpha <= hi)
        if mask.sum() == 0:
            continue
        max_common = int(np.min(lengths[mask]))
        sq = []
        for traj, n in zip(x[mask], lengths[mask]):
            y = traj[: min(int(n), max_common)]
            sq.append((y - y[0]) ** 2)
        mean_msd = np.mean(np.vstack(sq), axis=0)
        t = np.arange(max_common)
        valid = (t > 0) & (mean_msd > 0)
        slope, _ = np.polyfit(np.log(t[valid]), np.log(mean_msd[valid]), deg=1)
        results.append(
            {
                "alpha_mean": float(alpha[mask].mean()),
                "slope": float(slope),
                "t": t,
                "msd": mean_msd,
            }
        )

    return results
