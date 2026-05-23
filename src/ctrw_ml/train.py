from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from .config import DEMO_CONFIG, ProjectConfig
from .data import generate_dataset
from .model import CTRWLSTMRegressor
from .msd import estimate_msd_slope


def rmse_score(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(true, pred)))


class CTRWDataset(Dataset):
    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path, allow_pickle=True)
        self.x = data["x"].astype(np.float32)
        self.lengths = data["lengths"].astype(np.int64)
        self.alpha = data["alpha"].astype(np.float32)
        self.aux = np.zeros((self.alpha.size, 10), dtype=np.float32)
        for i, n in enumerate(self.lengths):
            y = self.x[i, : int(n)]
            dx = np.diff(y)
            abs_dx = np.abs(dx)
            slope = estimate_msd_slope(y)
            if not np.isfinite(slope):
                slope = 0.55
            self.aux[i] = np.array(
                [
                    np.clip(slope, 0.0, 1.5),
                    np.mean(abs_dx < 0.02),
                    np.mean(abs_dx < 0.05),
                    np.mean(abs_dx < 0.10),
                    np.percentile(abs_dx, 25),
                    np.percentile(abs_dx, 50),
                    np.percentile(abs_dx, 75),
                    np.std(dx),
                    np.mean(abs_dx),
                    np.max(abs_dx),
                ],
                dtype=np.float32,
            )

    def __len__(self) -> int:
        return self.alpha.size

    def __getitem__(self, idx: int):
        n = int(self.lengths[idx])
        x = self.x[idx].copy()
        dx = np.zeros_like(x)
        dx[1:n] = x[1:n] - x[: n - 1]
        sequence = np.stack([x, dx], axis=-1).astype(np.float32)
        aux = self.aux[idx].astype(np.float32)
        return (
            torch.from_numpy(sequence),
            torch.tensor(self.lengths[idx], dtype=torch.long),
            torch.from_numpy(aux),
            torch.tensor(self.alpha[idx], dtype=torch.float32),
        )


def split_indices(n: int, config: ProjectConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    idx = rng.permutation(n)
    n_train = int(config.train_fraction * n)
    n_val = int(config.val_fraction * n)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    device="cpu",
    grad_clip: float | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses = []

    for x, lengths, aux, y in tqdm(loader, leave=False):
        x, lengths, aux, y = x.to(device), lengths.to(device), aux.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            pred = model(x, lengths, aux)
            loss = criterion(pred, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses))


def predict(model, loader, device="cpu") -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, lengths, aux, y in loader:
            pred = model(x.to(device), lengths.to(device), aux.to(device))
            preds.append(pred.cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def train_project(config: ProjectConfig, data_path: Path, model_path: Path) -> dict[str, float]:
    if not data_path.exists():
        generate_dataset(config, data_path, robust=True)

    dataset = CTRWDataset(data_path)
    train_idx, val_idx, test_idx = split_indices(len(dataset), config)
    loader_kwargs = dict(batch_size=config.batch_size, num_workers=0)
    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(Subset(dataset, test_idx), shuffle=False, **loader_kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"

    torch.manual_seed(config.seed)
    model = CTRWLSTMRegressor(
        hidden_size=config.hidden_size,
        dense_size=config.dense_size,
        dropout=config.dropout,
        alpha_min=config.alpha_min,
        alpha_max=config.alpha_max,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )
    criterion = nn.MSELoss()

    best_val = float("inf")
    epochs_without_improvement = 0
    model_path.parent.mkdir(parents=True, exist_ok=True)

    total_epochs = config.pretrain_epochs + config.robust_epochs
    for epoch in range(1, total_epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip=config.grad_clip,
        )
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step(val_loss)
        print(f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        if val_loss < best_val - config.min_delta:
            best_val = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"early_stopping epoch={epoch:03d} best_val={best_val:.5f}")
                break

    model.load_state_dict(torch.load(model_path, map_location=device))
    pred, true = predict(model, test_loader, device)
    rmse = rmse_score(true, pred)
    mae = float(mean_absolute_error(true, pred))
    print(f"test_mae={mae:.4f} test_rmse={rmse:.4f}")
    return {"mae": mae, "rmse": rmse}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM on CTRW trajectories.")
    parser.add_argument("--demo", action="store_true", help="Run a small fast demo.")
    parser.add_argument("--data", default="data/ctrw_dataset.npz")
    parser.add_argument("--model", default="models/ctrw_lstm.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DEMO_CONFIG if args.demo else ProjectConfig()
    train_project(config, Path(args.data), Path(args.model))


if __name__ == "__main__":
    main()
