from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ProjectConfig
from .model import CTRWLSTMRegressor, freeze_lstm_layers
from .preprocessing import condition_trajectory
from .train import CTRWDataset, run_epoch


def microscopy_csv_to_npz(
    csv_path: Path,
    output_path: Path,
    config: ProjectConfig = ProjectConfig(),
) -> Path:
    """Convert labeled microscopy tracks to the same padded NPZ format used by training."""
    df = pd.read_csv(csv_path)
    required = {"trajectory_id", "time", "x", "alpha"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    rng = np.random.default_rng(config.seed)
    groups = list(df.sort_values(["trajectory_id", "time"]).groupby("trajectory_id"))
    x = np.zeros((len(groups), config.max_steps), dtype=np.float32)
    lengths = np.zeros(len(groups), dtype=np.int64)
    alpha = np.zeros(len(groups), dtype=np.float32)

    for i, (_, g) in enumerate(groups):
        t = g["time"].to_numpy(dtype=np.float64)
        y = g["x"].to_numpy(dtype=np.float32)
        n = int(np.clip(len(g), config.min_steps, config.max_steps))
        uniform_t = np.linspace(t.min(), t.max(), n)
        interpolated = np.interp(uniform_t, t, y).astype(np.float32)
        conditioned = condition_trajectory(interpolated, rng, noise_sigma=0.0, dropout_prob=0.0)
        x[i, :n] = conditioned
        lengths[i] = n
        alpha[i] = float(g["alpha"].iloc[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, x=x, lengths=lengths, alpha=alpha)
    return output_path


def fine_tune_experimental(
    pretrained_model_path: Path,
    experimental_npz: Path,
    output_model_path: Path,
    epochs: int = 20,
    lr: float = 5e-4,
    batch_size: int = 16,
) -> Path:
    dataset = CTRWDataset(experimental_npz)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"

    model = CTRWLSTMRegressor().to(device)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    freeze_lstm_layers(model)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        loss = run_epoch(model, loader, criterion, optimizer, device)
        print(f"fine_tune_epoch={epoch:03d} mse={loss:.5f}")

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_model_path)
    return output_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune CTRW LSTM on microscopy trajectories.")
    parser.add_argument("--csv", required=True, help="CSV with trajectory_id,time,x,alpha columns.")
    parser.add_argument("--pretrained-model", default="models/ctrw_lstm.pt")
    parser.add_argument("--experimental-data", default="data/experimental_tracks.npz")
    parser.add_argument("--output-model", default="models/ctrw_lstm_finetuned.pt")
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    microscopy_csv_to_npz(Path(args.csv), Path(args.experimental_data))
    fine_tune_experimental(
        Path(args.pretrained_model),
        Path(args.experimental_data),
        Path(args.output_model),
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
