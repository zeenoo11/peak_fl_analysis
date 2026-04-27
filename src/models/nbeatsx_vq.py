"""NBEATSx + VQ on h_generic (MVP).

Single VQ inserted between h_generic = stack_generic.fc4(...) and the proj
layer. trend / seasonal stacks are unchanged. This isolates the VQ effect to
the residual stack — the place where peak structure should live.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import D_MODEL, HORIZON, INPUT_SIZE, N_HARMONICS, N_POLYNOMIALS
from models.nbeatsx import (
    GenericBasis,
    NBEATSxStack,
    SeasonalityBasis,
    TrendBasis,
)
from models.vector_quantizer import VectorQuantizerEMA


class GenericStackWithVQ(nn.Module):
    """stack_generic with VQ inserted on h_generic before projection."""

    def __init__(
        self,
        in_features: int,
        d_model: int,
        n_theta: int,
        basis: nn.Module,
        num_embeddings: int = 32,
        commitment_beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.basis = basis
        self.fc1 = nn.Linear(in_features, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.fc3 = nn.Linear(d_model, d_model)
        self.fc4 = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, n_theta)
        self.vq = VectorQuantizerEMA(
            num_embeddings=num_embeddings,
            embedding_dim=d_model,
            commitment_beta=commitment_beta,
        )

    def forward(
        self, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        h = torch.relu(self.fc1(residual))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        h = self.fc4(h)
        h_q, commit_loss, info = self.vq(h)
        theta = self.proj(h_q)
        backcast, forecast = self.basis(theta)
        return backcast, forecast, h, commit_loss, info


class NBEATSxVQ(nn.Module):
    """NBEATSx with VQ on the generic stack only.

    forward(x) -> (y_hat, vq_state)
        vq_state = {
            'commit_loss':  scalar tensor,
            'utilization':  float (0..1),
            'perplexity':   float,
            'indices':      [B] long,
            'h_generic':    [B, d] (post-fc4, pre-VQ),
        }
    """

    def __init__(
        self,
        seq_len: int = INPUT_SIZE,
        horizon: int = HORIZON,
        d_model: int = D_MODEL,
        n_polynomials: int = N_POLYNOMIALS,
        n_harmonics: int = N_HARMONICS,
        num_embeddings: int = 32,
        commitment_beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.d_model = d_model
        self.num_embeddings = num_embeddings

        trend_basis = TrendBasis(n_polynomials, seq_len, horizon)
        seasonal_basis = SeasonalityBasis(n_harmonics, seq_len, horizon)
        generic_basis = GenericBasis(seq_len, horizon)

        n_theta_trend = 2 * (n_polynomials + 1)
        n_theta_seasonal = 4 * n_harmonics
        n_theta_generic = seq_len + horizon

        self.stack_trend = NBEATSxStack(seq_len, d_model, n_theta_trend, trend_basis)
        self.stack_seasonal = NBEATSxStack(seq_len, d_model, n_theta_seasonal, seasonal_basis)
        self.stack_generic = GenericStackWithVQ(
            seq_len,
            d_model,
            n_theta_generic,
            generic_basis,
            num_embeddings=num_embeddings,
            commitment_beta=commitment_beta,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        if x.dim() == 3:
            x = x.squeeze(-1)

        residual = x.clone()
        bc_t, fc_t, _ = self.stack_trend(residual)
        residual = residual - bc_t

        bc_s, fc_s, _ = self.stack_seasonal(residual)
        residual = residual - bc_s

        bc_g, fc_g, h_g, commit_loss, info = self.stack_generic(residual)

        y_hat = fc_t + fc_s + fc_g
        vq_state = {
            "commit_loss": commit_loss,
            "utilization": info["utilization"],
            "perplexity": info["perplexity"],
            "indices": info["indices"],
            "h_generic": h_g,
        }
        return y_hat, vq_state
