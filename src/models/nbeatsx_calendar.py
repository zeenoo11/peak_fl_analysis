"""NBEATSx with calendar feature injection.

Calendar features (forecast-start hour sin/cos + optional day-of-week sin/cos)
are concatenated with `residual` before fc1 of EVERY stack. Backcast is still
in seq_len-d (unchanged), so residual flow remains 96-d.

Aux head can be attached for peak supervision (peak_aux loss).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import D_MODEL, HORIZON, INPUT_SIZE, N_HARMONICS, N_POLYNOMIALS
from models.nbeatsx import GenericBasis, SeasonalityBasis, TrendBasis
from models.peak_aux_head import PeakAuxHead


class NBEATSxStackCal(nn.Module):
    def __init__(self, seq_len: int, n_cal: int, d_model: int,
                 n_theta: int, basis: nn.Module) -> None:
        super().__init__()
        self.seq_len = seq_len; self.n_cal = n_cal; self.d_model = d_model
        self.basis = basis
        self.fc1 = nn.Linear(seq_len + n_cal, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.fc3 = nn.Linear(d_model, d_model)
        self.fc4 = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, n_theta)

    def forward(self, residual: torch.Tensor, cal: torch.Tensor):
        x_in = torch.cat([residual, cal], dim=1)
        h = torch.relu(self.fc1(x_in))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        h = self.fc4(h)
        theta = self.proj(h)
        backcast, forecast = self.basis(theta)
        return backcast, forecast, h


class MinimalNBEATSxCal(nn.Module):
    def __init__(self, seq_len: int = INPUT_SIZE, horizon: int = HORIZON,
                 d_model: int = D_MODEL, n_polynomials: int = N_POLYNOMIALS,
                 n_harmonics: int = N_HARMONICS, n_cal: int = 4) -> None:
        super().__init__()
        self.seq_len = seq_len; self.horizon = horizon
        self.d_model = d_model; self.n_cal = n_cal

        trend_basis = TrendBasis(n_polynomials, seq_len, horizon)
        seasonal_basis = SeasonalityBasis(n_harmonics, seq_len, horizon)
        generic_basis = GenericBasis(seq_len, horizon)
        n_theta_t = 2 * (n_polynomials + 1)
        n_theta_s = 4 * n_harmonics
        n_theta_g = seq_len + horizon

        self.stack_trend = NBEATSxStackCal(seq_len, n_cal, d_model, n_theta_t, trend_basis)
        self.stack_seasonal = NBEATSxStackCal(seq_len, n_cal, d_model, n_theta_s, seasonal_basis)
        self.stack_generic = NBEATSxStackCal(seq_len, n_cal, d_model, n_theta_g, generic_basis)

    def forward(self, x: torch.Tensor, cal: torch.Tensor):
        if x.dim() == 3:
            x = x.squeeze(-1)
        residual = x.clone()
        bc_t, fc_t, h_t = self.stack_trend(residual, cal); residual = residual - bc_t
        bc_s, fc_s, h_s = self.stack_seasonal(residual, cal); residual = residual - bc_s
        bc_g, fc_g, h_g = self.stack_generic(residual, cal)
        y_hat = fc_t + fc_s + fc_g
        return y_hat, {"h_trend": h_t, "h_seasonal": h_s, "h_generic": h_g}


class NBEATSxAuxCal(nn.Module):
    """Calendar-aware NBEATSx with PeakAuxHead on h_generic."""
    def __init__(self, n_cal: int = 4) -> None:
        super().__init__()
        self.backbone = MinimalNBEATSxCal(n_cal=n_cal)
        self.aux_head = PeakAuxHead(in_dim=D_MODEL)

    def forward(self, x: torch.Tensor, cal: torch.Tensor):
        y_hat, hidd = self.backbone(x, cal)
        amp_p, hr_p = self.aux_head(hidd["h_generic"])
        return y_hat, hidd, (amp_p, hr_p)
