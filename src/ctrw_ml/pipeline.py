from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEMO_CONFIG, ProjectConfig
from .data import generate_dataset
from .evaluate import evaluate_project
from .train import train_project


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full CTRW alpha-regression pipeline.")
    parser.add_argument("--demo", action="store_true", help="Use a small quick dataset.")
    parser.add_argument("--data", default="data/ctrw_dataset.npz")
    parser.add_argument("--model", default="models/ctrw_lstm.pt")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--regenerate", action="store_true", help="Overwrite existing dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DEMO_CONFIG if args.demo else ProjectConfig()
    data_path = Path(args.data)
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)

    if args.regenerate or not data_path.exists():
        print(f"generating dataset: {data_path}")
        generate_dataset(config, data_path, robust=True)

    print("training LSTM")
    train_project(config, data_path, model_path)
    print("evaluating and plotting")
    metrics = evaluate_project(data_path, model_path, output_dir)
    print(f"done mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
