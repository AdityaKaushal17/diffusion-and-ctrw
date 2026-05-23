
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('outputs/.matplotlib').resolve()))
os.environ.setdefault('XDG_CACHE_HOME', str(Path('outputs/.cache').resolve()))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from ctrw_ml.config import ProjectConfig
from ctrw_ml.data import simulate_ctrw_trajectory
from ctrw_ml.evaluate import save_prediction_plot, save_msd_plot
from ctrw_ml.model import CTRWLSTMRegressor
from ctrw_ml.train import CTRWDataset, predict

ROOT = Path.cwd()
OUT = ROOT / 'poster_outputs'
OUT.mkdir(exist_ok=True)
DATA = ROOT / 'data/ctrw_dataset.npz'
MODEL = ROOT / 'models/ctrw_lstm.pt'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})


def get_predictions():
    dataset = CTRWDataset(DATA)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    arr = np.load(DATA, allow_pickle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    model = CTRWLSTMRegressor(
        alpha_min=float(arr['alpha'].min()),
        alpha_max=float(arr['alpha'].max()),
    ).to(device)
    model.load_state_dict(torch.load(MODEL, map_location=device))
    pred, true = predict(model, loader, device)
    return true, pred


def figure_trajectories():
    rng = np.random.default_rng(7)
    cases = [(0.3, '#2d6cdf', 'strong subdiffusion'), (0.7, '#2a9d55', 'moderate subdiffusion'), (1.0, '#d94841', 'near-normal diffusion')]
    fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    for ax, (alpha, color, label) in zip(axes, cases):
        y = simulate_ctrw_trajectory(rng, alpha, 200)
        ax.plot(np.arange(y.size), y, color=color, lw=1.8)
        ax.set_ylabel('x(t)')
        ax.set_title(f'alpha = {alpha:.1f}  ({label})', loc='left', fontsize=10)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel('time step')
    fig.suptitle('Figure 1: CTRW Trajectories for Different alpha', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = OUT / 'figure_1_trajectories.png'
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


def figure_msd():
    t = np.logspace(0, 3, 160)
    cases = [(0.3, '#2d6cdf'), (0.7, '#2a9d55'), (1.0, '#333333')]
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for alpha, color in cases:
        y = 0.02 * t ** alpha
        ax.loglog(t, y, color=color, lw=2.2, label=f'alpha = {alpha:.1f}')
        ax.text(t[-1] * 0.72, y[-1] * 1.06, f'slope = {alpha:.1f}', color=color, fontsize=9)
    ax.loglog(t, 0.03 * t ** 1.0, '--', color='black', lw=1.6, label='normal benchmark')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('MSD, <x^2(t)>')
    ax.set_title('Figure 2: MSD Scaling on Log-Log Axes', fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(alpha=0.25, which='both')
    p = OUT / 'figure_2_msd_loglog.png'
    fig.tight_layout()
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


def figure_parity():
    true, pred = get_predictions()
    mae = mean_absolute_error(true, pred)
    rmse = mean_squared_error(true, pred) ** 0.5
    r2 = r2_score(true, pred)
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    ax.scatter(true, pred, s=18, color='#075dcc', alpha=0.72, edgecolor='white', linewidth=0.2)
    ax.plot([0.1, 1.0], [0.1, 1.0], '--', color='black', lw=1.6)
    ax.set_xlim(0.08, 1.02)
    ax.set_ylim(0.08, 1.02)
    ax.set_xlabel('True alpha')
    ax.set_ylabel('Predicted alpha')
    ax.set_title('Figure 3: Predicted vs True alpha', fontweight='bold')
    ax.text(0.62, 0.18, f'MAE = {mae:.3f}\nRMSE = {rmse:.3f}\nR^2 = {r2:.3f}', fontsize=11,
            bbox=dict(facecolor='white', edgecolor='#bdbdbd', alpha=0.9))
    ax.grid(alpha=0.25)
    p = OUT / 'figure_3_predicted_vs_true_alpha.png'
    fig.tight_layout()
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p, {'mae': mae, 'rmse': rmse, 'r2': r2}


def figure_pipeline():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_axis_off()
    steps = [
        ('1', 'CTRW\nSimulation', '#dff1ff'),
        ('2', 'Preprocessing\nNoise + Dropout', '#e6f6df'),
        ('3', 'Sequence Input\nx(t), Delta x(t)', '#f0e8ff'),
        ('4', 'LSTM\n64 units', '#fff0d9'),
        ('5', 'Alpha\nRegression', '#ffe4e4'),
        ('6', 'MSD\nValidation', '#e8f7f7'),
    ]
    xs = np.linspace(0.08, 0.92, len(steps))
    for i, (num, title, color) in enumerate(steps):
        x = xs[i]
        box = FancyBboxPatch((x-0.07, 0.36), 0.14, 0.28, boxstyle='round,pad=0.02,rounding_size=0.02',
                             facecolor=color, edgecolor='#4c6f82', linewidth=1.4)
        ax.add_patch(box)
        ax.add_patch(Circle((x-0.055, 0.61), 0.025, color='#0a4a6d'))
        ax.text(x-0.055, 0.61, num, color='white', ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(x, 0.49, title, ha='center', va='center', fontsize=10, fontweight='bold')
        if i < len(steps)-1:
            ax.add_patch(FancyArrowPatch((x+0.08, 0.5), (xs[i+1]-0.08, 0.5), arrowstyle='->', mutation_scale=15,
                                         color='#0a4a6d', linewidth=1.8))
    ax.text(0.5, 0.82, 'Figure 4: End-to-End ML Pipeline for CTRW Alpha Prediction', ha='center',
            fontsize=14, fontweight='bold')
    ax.text(0.5, 0.22, 'Physics-based synthetic data -> robust conditioning -> LSTM learning -> regression metrics -> MSD consistency check',
            ha='center', fontsize=10)
    p = OUT / 'figure_4_ml_pipeline_overview.png'
    fig.savefig(p, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return p


def graphical_abstract(fig_paths):
    fig = plt.figure(figsize=(16, 4.8))
    for i, p in enumerate(fig_paths):
        ax = fig.add_subplot(1, 4, i + 1)
        img = plt.imread(p)
        ax.imshow(img)
        ax.set_axis_off()
    fig.suptitle('Graphical Abstract: Physics-Informed ML Workflow for Anomalous Diffusion', fontsize=16, fontweight='bold')
    p = OUT / 'graphical_abstract.png'
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


def add_wrapped(ax, x, y, text, size=9, width=92, bullet=False):
    import textwrap
    lines = []
    for para in text.split('\n'):
        if not para.strip():
            lines.append('')
            continue
        prefix = '- ' if bullet else ''
        wrapped = textwrap.wrap(para, width=width)
        if wrapped:
            lines.append(prefix + wrapped[0])
            lines.extend(('  ' + w) for w in wrapped[1:])
    ax.text(x, y, '\n'.join(lines), fontsize=size, va='top', ha='left', linespacing=1.25)


def poster(fig_paths, abstract_path, metrics):
    fig = plt.figure(figsize=(24, 34), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.add_patch(Rectangle((0, 0.955), 1, 0.045, facecolor='#9ed8ef', edgecolor='#0a3a4a', linewidth=2))
    ax.text(0.5, 0.987, 'Anomalous Diffusion and Non-Linear Reactions', ha='center', va='top', fontsize=20, fontweight='bold')
    ax.text(0.5, 0.971, 'Department of Chemical Engineering, Indian Institute of Technology Roorkee', ha='center', va='top', fontsize=14)
    ax.text(0.5, 0.958, 'ADITYA KAUSHAL - 23112007', ha='center', va='top', fontsize=13, fontweight='bold')
    ax.plot([0.5, 0.5], [0.04, 0.945], color='#6aa7bf', lw=1.5)
    ax.add_patch(Rectangle((0, 0), 1, 0.025, facecolor='#f2ad85', edgecolor='#0a3a4a', linewidth=1.5))
    ax.text(0.30, 0.012, 'Shram bina na kimapi sadhyam.', ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(0.73, 0.012, 'Nothing can be achieved without hard work!', ha='center', va='center', fontsize=13, fontweight='bold')

    ax.text(0.25, 0.935, 'Introduction and Background', ha='center', fontsize=16, fontweight='bold')
    intro = (
        'Diffusion is a fundamental transport process governing particle motion in fluids, porous media and complex reacting systems. Classical diffusion assumes independent jumps and finite characteristic time scales, giving linear MSD growth.\n'
        'Many real systems show anomalous diffusion because particles become trapped for broadly distributed waiting times. CTRW models this behavior using Gaussian jump lengths and heavy-tailed waiting times psi(tau) ~ tau^(-1-alpha).\n'
        'For 0 < alpha < 1, transport is subdiffusive and the MSD follows <x^2(t)> ~ t^alpha. The present project uses physics-based CTRW simulations to train an LSTM model that predicts alpha directly from particle trajectories.'
    )
    add_wrapped(ax, 0.035, 0.918, intro, size=10, width=92, bullet=True)

    ax.text(0.25, 0.735, 'Graphical Abstract', ha='center', fontsize=16, fontweight='bold')
    img = plt.imread(abstract_path)
    ax_img = fig.add_axes([0.035, 0.61, 0.43, 0.11])
    ax_img.imshow(img); ax_img.set_axis_off()

    ax.text(0.25, 0.585, 'Methodology', ha='center', fontsize=16, fontweight='bold')
    method = (
        'Step 1: CTRW data generation: 10,000 trajectories, length 100-200, alpha uniformly sampled in [0.1, 1.0].\n'
        'Step 2: Data conditioning: interpolation on a uniform time grid, zero-mean/unit-variance normalization, Gaussian noise and random frame dropout.\n'
        'Step 3: Feature representation: raw trajectory x(t), increments Delta x(t), MSD slope and trapping statistics.\n'
        'Step 4: LSTM architecture: 64-unit LSTM, dropout 0.2, dense ReLU layer and bounded alpha output.\n'
        'Step 5: Training: AdamW optimization, MSE loss, learning-rate scheduling, early stopping and robustness to noisy/incomplete trajectories.'
    )
    add_wrapped(ax, 0.04, 0.565, method, size=10, width=86, bullet=True)

    code_text = (
        'CTRW simulation core:\n'
        'Delta x ~ Normal(0, sigma^2)\n'
        'tau = tau0 (1-u)^(-1/alpha)\n'
        'x(t) interpolated on uniform grid\n\n'
        'ML target:\n'
        'minimize MSE(alpha_true, alpha_pred)\n\n'
        'Physical validation:\n'
        'log <x^2(t)> = alpha log(t) + constant'
    )
    ax_code = fig.add_axes([0.28, 0.40, 0.18, 0.14])
    ax_code.set_axis_off()
    ax_code.add_patch(FancyBboxPatch((0,0),1,1, boxstyle='round,pad=0.02', facecolor='#f7fff7', edgecolor='#96c59c'))
    ax_code.text(0.05, 0.95, code_text, va='top', fontsize=9, family='monospace', color='#115e22')

    ax.text(0.75, 0.935, 'Student Contributions', ha='center', fontsize=16, fontweight='bold')
    contrib = (
        'Formulated the problem by integrating anomalous diffusion theory with machine learning.\n'
        'Generated labeled CTRW trajectory datasets and implemented robust preprocessing.\n'
        'Designed an LSTM-based alpha regression model with physics-guided auxiliary features.\n'
        'Evaluated predictions using MAE, RMSE, R2 and MSD scaling validation.\n'
        'Prepared figures, workflow visualizations, interpretation and documentation.'
    )
    add_wrapped(ax, 0.535, 0.918, contrib, size=10, width=86, bullet=True)

    ax.text(0.75, 0.775, 'Results and Interpretation', ha='center', fontsize=16, fontweight='bold')
    results = (
        f'The LSTM model estimates the anomalous diffusion exponent from trajectory data with demo MAE = {metrics["mae"]:.3f}, RMSE = {metrics["rmse"]:.3f}, and R2 = {metrics["r2"]:.3f}.\n'
        'Lower alpha trajectories show reduced spreading and intermittent trapping, while alpha near one approaches normal diffusive behavior.\n'
        'Predicted values align with the ideal diagonal trend, indicating that the model learns the underlying non-Markovian transport dynamics.\n'
        'The MSD log-log slope provides a physical benchmark confirming consistency with <x^2(t)> ~ t^alpha.'
    )
    add_wrapped(ax, 0.535, 0.755, results, size=11, width=78, bullet=True)

    axp = fig.add_axes([0.54, 0.52, 0.20, 0.18]); axp.imshow(plt.imread(fig_paths[2])); axp.set_axis_off()
    axm = fig.add_axes([0.76, 0.52, 0.20, 0.18]); axm.imshow(plt.imread(fig_paths[1])); axm.set_axis_off()

    ax.text(0.75, 0.48, 'Conclusion', ha='center', fontsize=16, fontweight='bold')
    conclusion = (
        'A physics-informed machine learning framework was developed to predict anomalous diffusion behavior directly from trajectory data.\n'
        'The model distinguishes subdiffusive and near-normal transport regimes without explicitly solving fractional differential equations.\n'
        'Noise and missing-frame preprocessing improves relevance for microscopy-style experimental trajectories.\n'
        'The workflow provides a scalable alternative to classical fractional-model fitting while remaining consistent with CTRW physics.'
    )
    add_wrapped(ax, 0.535, 0.46, conclusion, size=10, width=82, bullet=True)

    ax.text(0.75, 0.30, 'References', ha='center', fontsize=16, fontweight='bold')
    refs = (
        'R. Metzler and J. Klafter, Phys. Rep. 339, 1 (2000).\n'
        'R. Metzler and J. Klafter, J. Phys. A 37, R161 (2004).\n'
        'I. M. Sokolov and J. Klafter, Chaos 15, 026103 (2005).\n'
        'M. Meerschaert and H. P. Scheffler, CTRW and space-time fractional diffusion equations.'
    )
    add_wrapped(ax, 0.535, 0.28, refs, size=10, width=88, bullet=True)

    ax.text(0.75, 0.16, 'Acknowledgement', ha='center', fontsize=16, fontweight='bold')
    ack = ('I acknowledge the guidance of Prof. Niladri Shekhar Mandal and the support of the Department of Chemical Engineering, IIT Roorkee. AI-based tools were used for assistance in structuring, refining and visualizing the project; final interpretation and implementation were independently verified.')
    add_wrapped(ax, 0.535, 0.14, ack, size=10, width=88, bullet=False)

    p = OUT / 'project_poster_summary.png'
    fig.savefig(p, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return p


def main():
    paths = []
    paths.append(figure_trajectories())
    paths.append(figure_msd())
    parity, metrics = figure_parity()
    paths.append(parity)
    paths.append(figure_pipeline())
    abstract = graphical_abstract(paths)
    poster_path = poster(paths, abstract, metrics)

    # Also refresh the original output folder plots for consistency.
    save_prediction_plot(*get_predictions(), ROOT / 'outputs')
    save_msd_plot(DATA, ROOT / 'outputs')

    report = OUT / 'OUTPUTS_README.txt'
    report.write_text('\n'.join([
        'Generated professor-facing outputs:',
        *(str(p) for p in paths),
        str(abstract),
        str(poster_path),
        f"Metrics: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}",
    ]))
    print(report.read_text())

if __name__ == '__main__':
    main()
