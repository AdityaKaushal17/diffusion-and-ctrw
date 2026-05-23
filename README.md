# Physics-Based CTRW Data Generation + LSTM Alpha Regression

This project generates synthetic Continuous Time Random Walk (CTRW) trajectories and trains an LSTM to estimate the anomalous diffusion exponent `alpha`.

It follows the requested workflow:

1. Physics-based CTRW trajectory generation
2. Conditioning and preprocessing
3. Time-series feature representation
4. LSTM regression model
5. Synthetic pre-training and noisy/incomplete robustness training
6. MAE/RMSE evaluation
7. Physical validation using MSD scaling

## Scientific Idea

For subdiffusive CTRW dynamics, waiting times follow a heavy-tailed law:

```text
psi(tau) ~ tau^(-1-alpha), 0 < alpha < 1
```

Jump lengths are sampled from:

```text
Delta x ~ Normal(0, sigma^2)
```

The model receives the trajectory `x(t)`, its increments `Delta x(t)`, and a compact physics-guided auxiliary vector built from MSD slope and trapping statistics. This keeps the LSTM focused on sequence dynamics while anchoring the output to CTRW behavior.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── data/
├── models/
├── outputs/
└── src/ctrw_ml/
    ├── config.py          # experiment defaults
    ├── data.py            # CTRW simulator and dataset generation
    ├── preprocessing.py   # interpolation, normalization, noise, dropout
    ├── model.py           # LSTM + physics-guided regression head
    ├── train.py           # training loop
    ├── evaluate.py        # metrics and plots
    ├── msd.py             # MSD scaling validation
    ├── transfer.py        # optional microscopy fine-tuning
    └── pipeline.py        # one-command full pipeline
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If editable install tries to download build dependencies in a restricted environment, use:

```bash
pip install -e . --no-build-isolation
```

You can also run directly from source:

```bash
PYTHONPATH=src python -m ctrw_ml.pipeline --demo --regenerate
```

## Quick Demo

Runs a small version for checking the full pipeline:

```bash
python -m ctrw_ml.pipeline --demo --regenerate
```

## Full Experiment

Runs the professor-facing specification:

- 10,000 trajectories
- length between 100 and 200 time steps
- alpha uniformly sampled in `[0.1, 1.0]`
- Gaussian noise with sigma `0.05`
- random frame dropout
- LSTM with 64 hidden units, dropout `0.2`, dense layer of 32 ReLU units
- bounded alpha output in `[0.1, 1.0]`
- physics-guided auxiliary features: MSD slope, plateau fractions, increment quantiles

```bash
python -m ctrw_ml.pipeline --regenerate
```

Generated files:

```text
data/ctrw_dataset.npz
models/ctrw_lstm.pt
outputs/predicted_vs_true_alpha.png
outputs/msd_scaling.png
outputs/msd_slope_benchmark.png
```

## Individual Commands

Train only:

```bash
python -m ctrw_ml.train --data data/ctrw_dataset.npz --model models/ctrw_lstm.pt
```

Evaluate only:

```bash
python -m ctrw_ml.evaluate --data data/ctrw_dataset.npz --model models/ctrw_lstm.pt --output-dir outputs
```

## Transfer Learning on Experimental Microscopy Tracks

Expected CSV columns:

```text
trajectory_id,time,x,alpha
```

Then run:

```bash
python -m ctrw_ml.transfer \
  --csv path/to/microscopy_tracks.csv \
  --pretrained-model models/ctrw_lstm.pt \
  --output-model models/ctrw_lstm_finetuned.pt
```

The LSTM layer is frozen and the final regression layers are retrained, preserving the physics learned from synthetic CTRW data while adapting to experimental trajectories.

## What To Show In A Presentation

- `src/ctrw_ml/data.py`: CTRW physics simulator
- `src/ctrw_ml/preprocessing.py`: interpolation, normalization, noise, dropout
- `src/ctrw_ml/model.py`: LSTM architecture with bounded alpha output and physics-guided head
- `outputs/predicted_vs_true_alpha.png`: ML regression validation
- `outputs/msd_scaling.png`: physical MSD scaling check
- `outputs/msd_slope_benchmark.png`: benchmark slope against alpha

## Notes

The MSD relation is used as an independent physical sanity check:

```text
<x^2(t)> ~ t^alpha
```

Because individual CTRW trajectories are noisy and non-ergodic, the ensemble MSD plot is usually smoother and more interpretable than a single-track MSD slope.


## Accuracy Improvements

The improved version uses three changes that make the saved output more accurate and more defensible:

- alpha predictions are constrained to the physical range `[0.1, 1.0]`
- the LSTM input includes both `x(t)` and `Delta x(t)` so trapping behavior is easier to learn
- the regression head receives MSD/trapping summary features as a physics-guided correction

On the included demo run, the evaluation improved from about `MAE = 0.2244`, `RMSE = 0.2599` to `MAE = 0.1928`, `RMSE = 0.2257`.
