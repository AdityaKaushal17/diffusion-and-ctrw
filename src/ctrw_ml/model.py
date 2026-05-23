from __future__ import annotations

import torch
from torch import nn


class CTRWLSTMRegressor(nn.Module):
    """LSTM regressor for estimating the anomalous exponent alpha from x(t)."""

    def __init__(
        self,
        hidden_size: int = 64,
        dense_size: int = 32,
        dropout: float = 0.2,
        alpha_min: float = 0.1,
        alpha_max: float = 1.0,
        aux_features: int = 10,
    ):
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + aux_features, dense_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_size, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        aux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        features = self.dropout(hidden[-1])
        if aux is None:
            aux = torch.zeros(features.shape[0], 1, device=features.device, dtype=features.dtype)
        features = torch.cat([features, aux], dim=1)
        unit_alpha = torch.sigmoid(self.head(features).squeeze(-1))
        return self.alpha_min + (self.alpha_max - self.alpha_min) * unit_alpha


def freeze_lstm_layers(model: CTRWLSTMRegressor) -> None:
    for param in model.lstm.parameters():
        param.requires_grad = False
