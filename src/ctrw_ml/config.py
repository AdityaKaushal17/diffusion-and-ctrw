from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectConfig:
    n_trajectories: int = 10_000
    min_steps: int = 100
    max_steps: int = 200
    alpha_min: float = 0.1
    alpha_max: float = 1.0
    jump_sigma: float = 1.0
    tau0: float = 1.0
    noise_sigma: float = 0.05
    dropout_prob: float = 0.10
    train_fraction: float = 0.80
    val_fraction: float = 0.10
    batch_size: int = 32
    pretrain_epochs: int = 50
    robust_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_size: int = 64
    dense_size: int = 32
    dropout: float = 0.2
    grad_clip: float = 1.0
    patience: int = 12
    min_delta: float = 1e-5
    seed: int = 42


DEMO_CONFIG = ProjectConfig(
    n_trajectories=1_200,
    pretrain_epochs=10,
    robust_epochs=5,
)
