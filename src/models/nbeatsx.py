"""Minimal NBEATSx (3-stack: trend / seasonal / generic).

Origin: ported from Peak_Analysis/experiments/federated/v10_b2_nbeatsx_novq.py.
The state_dict layer names are kept identical so v10 b2 checkpoints load
strict=True without rewriting keys.

The only intentional change: forward() returns the per-stack hidden vectors
(h_trend, h_seasonal, h_generic) inside an info dict so probes can read them.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import D_MODEL, HORIZON, INPUT_SIZE, N_HARMONICS, N_POLYNOMIALS


class TrendBasis(nn.Module):
    """Polynomial trend basis: theta [B, 2*(p+1)] -> backcast/forecast."""

    def __init__(self, n_polynomials: int, seq_len: int, horizon: int) -> None:
        super().__init__()
        self.n_polynomials = n_polynomials
        backcast_t = torch.linspace(-1, 1, seq_len)
        forecast_t = torch.linspace(0, 1, horizon)
        back_mat = torch.stack([backcast_t ** i for i in range(n_polynomials + 1)], dim=0)
        fcast_mat = torch.stack([forecast_t ** i for i in range(n_polynomials + 1)], dim=0)
        self.register_buffer("back_mat", back_mat)
        self.register_buffer("fcast_mat", fcast_mat)

    def forward(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.n_polynomials + 1
        backcast = theta[:, :p] @ self.back_mat
        forecast = theta[:, p:] @ self.fcast_mat
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    """Harmonic seasonality basis: theta [B, 4*H] -> backcast/forecast."""

    def __init__(self, n_harmonics: int, seq_len: int, horizon: int) -> None:
        super().__init__()
        self.n_harmonics = n_harmonics
        backcast_t = torch.linspace(0, 1, seq_len)
        forecast_t = torch.linspace(0, 1, horizon)
        freqs = torch.arange(1, n_harmonics + 1).float()
        back_cos = torch.cos(2 * math.pi * freqs.unsqueeze(1) * backcast_t.unsqueeze(0))
        back_sin = torch.sin(2 * math.pi * freqs.unsqueeze(1) * backcast_t.unsqueeze(0))
        fcast_cos = torch.cos(2 * math.pi * freqs.unsqueeze(1) * forecast_t.unsqueeze(0))
        fcast_sin = torch.sin(2 * math.pi * freqs.unsqueeze(1) * forecast_t.unsqueeze(0))
        self.register_buffer("back_mat", torch.cat([back_cos, back_sin], dim=0))
        self.register_buffer("fcast_mat", torch.cat([fcast_cos, fcast_sin], dim=0))

    def forward(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h2 = 2 * self.n_harmonics
        backcast = theta[:, :h2] @ self.back_mat
        forecast = theta[:, h2:] @ self.fcast_mat
        return backcast, forecast


class GenericBasis(nn.Module):
    """Identity / generic basis: theta directly = (backcast || forecast)."""

    def __init__(self, seq_len: int, horizon: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon

    def forward(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return theta[:, : self.seq_len], theta[:, self.seq_len :]


class NBEATSxStack(nn.Module):
    """Single NBEATS stack: 4-layer MLP -> projection -> basis."""

    def __init__(
        self,
        in_features: int,
        d_model: int,
        n_theta: int,
        basis: nn.Module,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.basis = basis
        self.fc1 = nn.Linear(in_features, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.fc3 = nn.Linear(d_model, d_model)
        self.fc4 = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, n_theta)

    def forward(
        self, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = torch.relu(self.fc1(residual))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        h = self.fc4(h)
        theta = self.proj(h)
        backcast, forecast = self.basis(theta)
        return backcast, forecast, h


class MinimalNBEATSx(nn.Module):
    """3-stack NBEATSx (trend / seasonal / generic), no VQ.

    forward(x) -> (y_hat, hiddens) where
        hiddens = {'h_trend': [B, d], 'h_seasonal': [B, d], 'h_generic': [B, d]}
    """

    def __init__(
        self,
        seq_len: int = INPUT_SIZE,
        horizon: int = HORIZON,
        d_model: int = D_MODEL,
        n_polynomials: int = N_POLYNOMIALS,
        n_harmonics: int = N_HARMONICS,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.d_model = d_model

        trend_basis = TrendBasis(n_polynomials, seq_len, horizon)
        seasonal_basis = SeasonalityBasis(n_harmonics, seq_len, horizon)
        generic_basis = GenericBasis(seq_len, horizon)

        n_theta_trend = 2 * (n_polynomials + 1)
        n_theta_seasonal = 4 * n_harmonics
        n_theta_generic = seq_len + horizon

        self.stack_trend = NBEATSxStack(seq_len, d_model, n_theta_trend, trend_basis)
        self.stack_seasonal = NBEATSxStack(seq_len, d_model, n_theta_seasonal, seasonal_basis)
        self.stack_generic = NBEATSxStack(seq_len, d_model, n_theta_generic, generic_basis)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.dim() == 3:
            x = x.squeeze(-1)

        residual = x.clone()
        bc_t, fc_t, h_t = self.stack_trend(residual)
        residual = residual - bc_t

        bc_s, fc_s, h_s = self.stack_seasonal(residual)
        residual = residual - bc_s

        bc_g, fc_g, h_g = self.stack_generic(residual)

        y_hat = fc_t + fc_s + fc_g
        hiddens = {"h_trend": h_t, "h_seasonal": h_s, "h_generic": h_g}
        return y_hat, hiddens
