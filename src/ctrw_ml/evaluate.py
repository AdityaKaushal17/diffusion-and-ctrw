from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs/.cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from .model import CTRWLSTMRegressor
from .msd import ensemble_msd_by_alpha_bin, estimate_msd_slope
from .train import CTRWDataset, predict


def rmse_score(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(true, pred)))


def save_prediction_plot(true: np.ndarray, pred: np.ndarray, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(true, pred, s=14, alpha=0.65, edgecolor="none")
    lo = min(float(true.min()), float(pred.min()))
    hi = max(float(true.max()), float(pred.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.2, label="ideal")
    ax.set_xlabel("Ground-truth alpha")
    ax.set_ylabel("Predicted alpha")
    ax.set_title("CTRW LSTM Regression")
    ax.legend()
    ax.grid(alpha=0.25)
    path = output_dir / "predicted_vs_true_alpha.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_msd_plot(data_path: Path, output_dir: Path) -> Path:
    data = np.load(data_path, allow_pickle=True)
    bins = ensemble_msd_by_alpha_bin(data["x"], data["lengths"], data["alpha"])
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    for item in bins:
        t = item["t"][1:]
        msd = item["msd"][1:]
        ax.loglog(
            t,
            msd,
            linewidth=1.4,
            label=f"alpha={item['alpha_mean']:.2f}, slope={item['slope']:.2f}",
        )
    ax.set_xlabel("time lag t")
    ax.set_ylabel("<x^2(t)>")
    ax.set_title("MSD Scaling Check")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    path = output_dir / "msd_scaling.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_msd_slope_plot(data_path: Path, output_dir: Path, max_samples: int = 300) -> Path:
    data = np.load(data_path, allow_pickle=True)
    rng = np.random.default_rng(123)
    idx = rng.choice(len(data["alpha"]), size=min(max_samples, len(data["alpha"])), replace=False)
    slopes = np.array(
        [estimate_msd_slope(data["x"][i, : data["lengths"][i]]) for i in idx],
        dtype=np.float64,
    )
    true = data["alpha"][idx]
    valid = np.isfinite(slopes)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(true[valid], slopes[valid], s=14, alpha=0.65, edgecolor="none")
    ax.plot([0.1, 1.0], [0.1, 1.0], color="black", linewidth=1.2, label="theory")
    ax.set_xlabel("Ground-truth alpha")
    ax.set_ylabel("Single-trajectory MSD slope")
    ax.set_title("Physical Benchmark: MSD Slope")
    ax.legend()
    ax.grid(alpha=0.25)
    path = output_dir / "msd_slope_benchmark.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def evaluate_project(data_path: Path, model_path: Path, output_dir: Path) -> dict[str, float]:
    dataset = CTRWDataset(data_path)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    data = np.load(data_path, allow_pickle=True)
    alpha_min = float(data["alpha"].min())
    alpha_max = float(data["alpha"].max())
    model = CTRWLSTMRegressor(alpha_min=alpha_min, alpha_max=alpha_max).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    pred, true = predict(model, loader, device)
    mae = float(mean_absolute_error(true, pred))
    rmse = rmse_score(true, pred)
    save_prediction_plot(true, pred, output_dir)
    save_msd_plot(data_path, output_dir)
    save_msd_slope_plot(data_path, output_dir)
    return {"mae": mae, "rmse": rmse}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained CTRW LSTM model.")
    parser.add_argument("--data", default="data/ctrw_dataset.npz")
    parser.add_argument("--model", default="models/ctrw_lstm.pt")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_project(Path(args.data), Path(args.model), Path(args.output_dir))
    print(f"mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
