"""
Ensemble (stacking) model that combines Dense and CNN predictions.

The meta-learner takes the predictions of both base models as input
and learns to optimally combine them.
"""

import torch
import torch.nn as nn


class EnsembleMetaLearner(nn.Module):
    """
    Small network that takes 2 inputs (H_dense, H_cnn) and outputs final H.
    Intentionally kept small to avoid overfitting on just 2 features.
    """

    def __init__(self, n_inputs: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class EnsembleWithFeatures(nn.Module):
    """
    Enhanced ensemble that also receives uncertainty estimates
    and the agreement between models as features.

    Input: [H_dense, H_cnn, std_dense, std_cnn, |H_dense - H_cnn|]
    """

    def __init__(self, n_inputs: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
