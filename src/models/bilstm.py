"""
src/models/bilstm.py
--------------------
Bidirectional LSTM classifier for weekly seismicity sequences.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM for binary earthquake occurrence prediction.

    Each sample is treated as a (1, n_features) single-step sequence.
    Forward and backward hidden states are concatenated and passed through
    a linear sigmoid head.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_size : int
        LSTM units per direction (default: 32 → 64 total after concatenation).
    """

    def __init__(self, input_dim: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor  shape (batch, seq_len, input_dim)

        Returns
        -------
        torch.Tensor  shape (batch, 1)
            Predicted probability of a seismic event in the next 4 weeks.
        """
        _, (hn, _) = self.lstm(x)
        out = torch.cat((hn[-2], hn[-1]), dim=1)
        return torch.sigmoid(self.fc(out))
