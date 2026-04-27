"""Auxiliary peak prediction head for multi-task learning.

Inputs an arbitrary latent vector h ∈ R^D and predicts:
    - peak_amp (regression, 1-d) — max value of forecast horizon (z-space)
    - peak_hr  (classification, 24-class) — argmax position of forecast horizon
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PeakAuxHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, n_hours: int = 24) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.amp_head = nn.Linear(hidden, 1)
        self.hr_head = nn.Linear(hidden, n_hours)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.shared(h)
        return self.amp_head(z).squeeze(-1), self.hr_head(z)


def peak_aux_loss(
    amp_pred: torch.Tensor,
    hr_pred: torch.Tensor,
    y: torch.Tensor,
    hr_weight: float = 0.1,
) -> torch.Tensor:
    """Combined peak loss. y is z-normalized forecast horizon [B, 24]."""
    amp_true = y.max(dim=1).values
    hr_true = y.argmax(dim=1)
    return F.mse_loss(amp_pred, amp_true) + hr_weight * F.cross_entropy(hr_pred, hr_true)
