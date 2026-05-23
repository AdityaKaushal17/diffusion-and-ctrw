from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import ProjectConfig
from .preprocessing import condition_trajectory


def sample_power_law_waiting_times(
    rng: np.random.Generator,
    alpha: float,
    size: int,
    tau0: float = 1.0,
) -> np.ndarray:
    """Sample Pareto waiting times with asymptotic density psi(tau) ~ tau^(-1-alpha)."""
    u = rng.uniform(0.0, 1.0, size=size)
    return tau0 * (1.0 - u) ** (-1.0 / alpha)


def simulate_ctrw_trajectory(
    rng: np.random.Generator,
    alpha: float,
    n_steps: int,
    jump_sigma: float = 1.0,
    tau0: float = 1.0,
) -> np.ndarray:
    """Generate one 1D CTRW trajectory and interpolate it onto a uniform grid."""
    target_t = np.arange(n_steps, dtype=np.float64)
    event_times = [0.0]
    positions = [0.0]

    while event_times[-1] < target_t[-1]:
        tau = sample_power_law_waiting_times(rng, alpha, size=1, tau0=tau0)[0]
        jump = rng.normal(0.0, jump_sigma)
        event_times.append(event_times[-1] + tau)
        positions.append(positions[-1] + jump)

    return np.interp(target_t, np.asarray(event_times), np.asarray(positions)).astype(np.float32)


def generate_dataset(
    config: ProjectConfig,
    output_path: str | Path,
    robust: bool = True,
) -> Path:
    """Generate padded CTRW trajectories, sequence lengths, and ground-truth alpha labels."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)

    x = np.zeros((config.n_trajectories, config.max_steps), dtype=np.float32)
    lengths = np.zeros(config.n_trajectories, dtype=np.int64)
    alpha = rng.uniform(config.alpha_min, config.alpha_max, size=config.n_trajectories).astype(np.float32)

    for i in range(config.n_trajectories):
        n_steps = int(rng.integers(config.min_steps, config.max_steps + 1))
        clean = simulate_ctrw_trajectory(
            rng,
            float(alpha[i]),
            n_steps,
            jump_sigma=config.jump_sigma,
            tau0=config.tau0,
        )
        conditioned = condition_trajectory(
            clean,
            rng=rng,
            noise_sigma=config.noise_sigma if robust else 0.0,
            dropout_prob=config.dropout_prob if robust else 0.0,
        )
        x[i, :n_steps] = conditioned
        lengths[i] = n_steps

    np.savez_compressed(
        output_path,
        x=x,
        lengths=lengths,
        alpha=alpha,
        config=asdict(config),
    )
    return output_path
