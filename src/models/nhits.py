"""NHITS variants used across project versions.

This module hosts two NHITS implementations side-by-side:

- ``MinimalNHITS`` / ``NHITSAux`` — v01 simplified NHITS (3 stacks
  low/mid/high frequency), ported from
  ``Peak_Analysis/experiments/federated/v10_b3_nhits_novq.py``. Used by
  ``experiments/v01_peak_from_latent/13_iter5C_nhits.py``. **Do not
  modify** — v01 numbers depend on this exact definition.

- ``NHITS`` — v04 paper-faithful re-implementation following
  Challu et al., AAAI'23 (https://arxiv.org/abs/2201.12886): adds
  doubly-residual blocks, MaxPool input compression per stack,
  frequency-domain theta interpolation, and multi-rate sampling with
  ``pool_kernel_sizes=[2, 2, 1]`` / ``n_freq_downsamples=[4, 2, 1]``.
  Used by v04 NF-axis baselines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import D_MODEL, HORIZON, INPUT_SIZE
from models.peak_aux_head import PeakAuxHead

POOLING_KERNELS = (8, 4, 2)


class NHITSStack(nn.Module):
    def __init__(self, pooling_kernel: int, input_size: int = INPUT_SIZE,
                 horizon: int = HORIZON, d_hidden: int = D_MODEL) -> None:
        super().__init__()
        pooled_len = input_size // pooling_kernel
        self.pool = nn.MaxPool1d(kernel_size=pooling_kernel, stride=pooling_kernel)
        self.fc1 = nn.Linear(pooled_len, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc_out = nn.Linear(d_hidden, horizon)

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.squeeze(-1)
        pooled = self.pool(x.unsqueeze(1)).squeeze(1)
        h = F.relu(self.fc1(pooled))
        h = F.relu(self.fc2(h))
        forecast = self.fc_out(h)
        return h, forecast


class MinimalNHITS(nn.Module):
    def __init__(self, input_size: int = INPUT_SIZE, horizon: int = HORIZON,
                 d_hidden: int = D_MODEL,
                 pooling_kernels: tuple = POOLING_KERNELS) -> None:
        super().__init__()
        self.stack_low = NHITSStack(pooling_kernels[0], input_size, horizon, d_hidden)
        self.stack_mid = NHITSStack(pooling_kernels[1], input_size, horizon, d_hidden)
        self.stack_high = NHITSStack(pooling_kernels[2], input_size, horizon, d_hidden)

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.squeeze(-1)
        h_low, fc_low = self.stack_low(x)
        h_mid, fc_mid = self.stack_mid(x)
        h_high, fc_high = self.stack_high(x)
        y_hat = fc_low + fc_mid + fc_high
        return y_hat, {"h_low": h_low, "h_mid": h_mid, "h_high": h_high}


class NHITSAux(nn.Module):
    """NHITS + PeakAuxHead on h_high (highest frequency, captures peak best)."""

    def __init__(self, latent_source: str = "h_high") -> None:
        super().__init__()
        if latent_source not in ("h_low", "h_mid", "h_high", "h_concat"):
            raise ValueError(latent_source)
        self.latent_source = latent_source
        self.backbone = MinimalNHITS()
        in_dim = D_MODEL if latent_source != "h_concat" else 3 * D_MODEL
        self.aux_head = PeakAuxHead(in_dim=in_dim)

    def get_latent(self, hiddens):
        if self.latent_source == "h_concat":
            return torch.cat([hiddens["h_low"], hiddens["h_mid"], hiddens["h_high"]], dim=1)
        return hiddens[self.latent_source]

    def forward(self, x: torch.Tensor):
        y_hat, hidd = self.backbone(x)
        h = self.get_latent(hidd)
        amp_p, hr_p = self.aux_head(h)
        return y_hat, hidd, (amp_p, hr_p)


# =============================================================================
# v04 paper-faithful NHITS (Challu et al., AAAI'23)
# =============================================================================
#
# Mirrors the official Nixtla implementation
# (papers/literlature/nhits_official/nhits.py, github.com/Nixtla/neuralforecast)
# with the following deliberate simplifications:
#   - univariate (UMass apt-level kW; we drop the multivariate / exogenous axes),
#   - no PyTorch-Lightning ``BaseModel`` plumbing (this file exposes plain
#     ``nn.Module``s; training loop lives in v04 experiments/),
#   - identity basis only (the official supports linear / nearest / cubic
#     interpolation; we keep the linear default and skip the cubic batched-loop).
# Algorithmic invariants kept verbatim:
#   - one MaxPool1d per block (kernel = stride = n_pool_kernel_size, ceil_mode),
#   - single ``theta = [backcast | knots]`` output, split + interpolate via
#     ``_IdentityBasis`` (matches official lines 18-72),
#   - **Naive1 forecast initialisation** (forecast starts at last input value,
#     residuals start as flipped insample) — this is the v01 NBEATS behaviour
#     the official NHITS inherits, and it materially affects training dynamics,
#   - per-block ``forecast = forecast + block_forecast`` accumulation.

import math


class _IdentityBasis(nn.Module):
    """Identity basis: theta = [backcast (size L) | knots (size theta_f)] →
    backcast as-is, knots linearly interpolated to horizon H.

    Verbatim port of the official implementation
    (papers/literlature/nhits_official/nhits.py:_IdentityBasis), univariate
    path only (out_features=1, no cubic batched-loop).
    """

    def __init__(
        self,
        backcast_size: int,
        forecast_size: int,
        interpolation_mode: str = "linear",
    ) -> None:
        super().__init__()
        if interpolation_mode not in ("linear", "nearest"):
            raise ValueError(
                f"_IdentityBasis: only 'linear' / 'nearest' supported here, "
                f"got {interpolation_mode!r} (cubic path is in the official "
                f"implementation but not needed for v04)."
            )
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.interpolation_mode = interpolation_mode

    def forward(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # theta: [B, backcast_size + theta_f_size]
        backcast = theta[:, : self.backcast_size]                       # [B, L]
        knots = theta[:, self.backcast_size :]                          # [B, theta_f]
        # F.interpolate expects 3D input [B, C, L]; pretend C=1.
        knots = knots.reshape(len(knots), 1, -1)
        forecast = F.interpolate(
            knots, size=self.forecast_size, mode=self.interpolation_mode
        )                                                                # [B, 1, H]
        forecast = forecast.squeeze(1)                                   # [B, H]
        return backcast, forecast


class _NHITSBlock(nn.Module):
    """One paper-faithful NHITS block: pool → MLP → theta → basis."""

    def __init__(
        self,
        input_size: int,
        horizon: int,
        n_theta: int,
        mlp_hidden: tuple[int, ...],
        basis: nn.Module,
        n_pool_kernel_size: int,
        pooling_mode: str = "MaxPool1d",
        dropout: float = 0.0,
        activation: str = "ReLU",
    ) -> None:
        super().__init__()
        if pooling_mode not in ("MaxPool1d", "AvgPool1d"):
            raise ValueError(f"pooling_mode must be MaxPool1d/AvgPool1d, got {pooling_mode!r}")
        if not hasattr(nn, activation):
            raise ValueError(f"activation {activation!r} not found in torch.nn")

        # ceil_mode=True so partial windows still produce an output element
        # (matches the official; needed when input_size % pool_size != 0).
        self.pooling_layer = getattr(nn, pooling_mode)(
            kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True
        )
        pooled_input_size = math.ceil(input_size / n_pool_kernel_size)

        # MLP: [pooled_in -> h0] + [h0 -> h1, h1 -> h2, ...] + dropout, ending
        # with a final Linear(h_last, n_theta).  We mirror the official
        # ``hidden_layers + output_layer`` pattern exactly.
        activ = getattr(nn, activation)()
        layers: list[nn.Module] = [nn.Linear(pooled_input_size, mlp_hidden[0])]
        for in_dim, out_dim in zip(mlp_hidden[:-1], mlp_hidden[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activ)
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        # The official does activ + dropout after every hidden→hidden Linear
        # but not after the very first input→h0 Linear — to stay faithful, we
        # add one trailing activ here on the last hidden output before theta.
        layers.append(activ)
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(mlp_hidden[-1], n_theta))
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, insample_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # insample_y: [B, L] (univariate, no exog).
        x = insample_y.unsqueeze(1)             # [B, 1, L]  (Pool1d wants 3D)
        x = self.pooling_layer(x).squeeze(1)    # [B, pooled_L]
        theta = self.layers(x)                  # [B, n_theta]
        backcast, forecast = self.basis(theta)
        return backcast, forecast               # [B, L], [B, H]


class NHITS(nn.Module):
    """v04 paper-faithful univariate NHITS forecaster.

    Defaults follow the official ``neuralforecast.NHITS`` (and the AAAI'23
    paper §4.1 short-horizon recipe), trimmed to univariate:

        n_blocks            = (1, 1, 1)            # blocks per stack
        mlp_units           = (512, 512)           # hidden dims, all stacks
        n_pool_kernel_size  = (2, 2, 1)
        n_freq_downsample   = (4, 2, 1)
        pooling_mode        = 'MaxPool1d'
        interpolation_mode  = 'linear'
        activation          = 'ReLU'
        dropout             = 0.0

    Forward
    -------
        y_hat = NHITS()(x)   # x: [B, L]  ->  y_hat: [B, H]
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        horizon: int = HORIZON,
        n_blocks: tuple[int, ...] = (1, 1, 1),
        mlp_units: tuple[int, ...] = (512, 512),
        n_pool_kernel_size: tuple[int, ...] = (2, 2, 1),
        n_freq_downsample: tuple[int, ...] = (4, 2, 1),
        pooling_mode: str = "MaxPool1d",
        interpolation_mode: str = "linear",
        dropout: float = 0.0,
        activation: str = "ReLU",
    ) -> None:
        super().__init__()
        n_stacks = len(n_blocks)
        if len(n_pool_kernel_size) != n_stacks or len(n_freq_downsample) != n_stacks:
            raise ValueError(
                f"pool/freq lists must have length n_stacks={n_stacks}; "
                f"got {len(n_pool_kernel_size)} / {len(n_freq_downsample)}"
            )
        self.input_size = input_size
        self.horizon = horizon

        blocks: list[nn.Module] = []
        for i in range(n_stacks):
            forecast_theta_size = max(horizon // n_freq_downsample[i], 1)
            n_theta = input_size + forecast_theta_size
            for _ in range(n_blocks[i]):
                # Each block gets its own _IdentityBasis instance so the basis
                # state (none, but if cubic ever added it would need it) is
                # not shared across blocks.
                basis = _IdentityBasis(
                    backcast_size=input_size,
                    forecast_size=horizon,
                    interpolation_mode=interpolation_mode,
                )
                blocks.append(
                    _NHITSBlock(
                        input_size=input_size,
                        horizon=horizon,
                        n_theta=n_theta,
                        mlp_hidden=mlp_units,
                        basis=basis,
                        n_pool_kernel_size=n_pool_kernel_size[i],
                        pooling_mode=pooling_mode,
                        dropout=dropout,
                        activation=activation,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L] (univariate)
        if x.dim() == 3:
            x = x.squeeze(-1)
        # Naive1 forecast initialisation (last input value repeated H times),
        # residual starts as the flipped insample so the first block sees the
        # most recent observation first.  Verbatim from official forward.
        residuals = x.flip(dims=(-1,))                              # [B, L]
        forecast = x[:, -1:].repeat(1, self.horizon)                # [B, H]
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast
        return forecast
