"""
src/models/neuro_fuzzy.py
--------------------------
ANFIS-style Neuro-Fuzzy classifier with learnable Gaussian membership functions.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NeuroFuzzyLayer(nn.Module):
    """Adaptive Neuro-Fuzzy Inference System (ANFIS) approximation.

    Each input feature is fuzzified by ``n_mf`` Gaussian membership functions
    with learnable centres and widths.  The resulting firing strengths are
    flattened and mapped to a binary output via a linear sigmoid head.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    n_mf : int
        Number of membership functions per feature (default: 3).
    """

    def __init__(self, input_dim: int, n_mf: int = 3) -> None:
        super().__init__()
        self.means = nn.Parameter(torch.randn(input_dim, n_mf))
        self.stds  = nn.Parameter(torch.ones(input_dim, n_mf))
        self.fc    = nn.Linear(input_dim * n_mf, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor  shape (batch, input_dim)

        Returns
        -------
        torch.Tensor  shape (batch, 1)
        """
        x_exp   = x.unsqueeze(-1)                                          # (B, D, 1)
        mu      = torch.exp(
            -0.5 * ((x_exp - self.means) / (self.stds.abs() + 1e-5)) ** 2
        )                                                                   # (B, D, n_mf)
        mu_flat = mu.reshape(x.size(0), -1)                                # (B, D*n_mf)
        return torch.sigmoid(self.fc(mu_flat))
