"""
1D CNN architectures for Hurst exponent estimation.
Replicates the Stone (2020) architecture from Quantitative Finance.
"""

import torch
import torch.nn as nn


class HurstCNN(nn.Module):
    """
    CNN following Stone (2020) for Hurst exponent calibration.

    Architecture:
        Conv1d(1→32, k=5) + ReLU + MaxPool(2)
        Conv1d(32→64, k=5) + ReLU + MaxPool(2)
        Conv1d(64→128, k=3) + ReLU + MaxPool(2)
        Flatten → Dense(→128) + ReLU + Dropout → Dense(→1)
    """

    def __init__(self, input_size: int = 100, dropout: float = 0.15):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # Block 1: detect short-range patterns
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2: combine short-range into medium-range
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3: combine into long-range features
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_size)
            conv_out = self.conv_layers(dummy)
            self.flat_size = conv_out.view(1, -1).shape[1]

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x shape: (batch, 100) → need (batch, 1, 100) for Conv1d
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
