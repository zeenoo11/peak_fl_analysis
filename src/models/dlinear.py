"""DLinear (Zeng et al., AAAI'23) — direct implementation.

Reference
---------
A. Zeng, M. Chen, L. Zhang, Q. Xu, "Are Transformers Effective for Time
Series Forecasting?", AAAI 2023. https://arxiv.org/abs/2205.13504

Architecture (univariate)
-------------------------
1. Series decomposition by a centred moving average:
       trend(t)    = MA(x; kernel=25)
       seasonal(t) = x(t) - trend(t)
2. Two independent linear projections, each from input length L=96 to
   horizon H=24:
       y_seasonal = W_s @ seasonal
       y_trend    = W_t @ trend
3. Forecast = y_seasonal + y_trend.

Univariate variant (DLinear-S/DLinear-I collapse to the same model when
the series is single-channel). UMass apt-level kW is 1-D, so this file
implements the univariate path directly.

Interface (matches v04 NF baselines convention)
-----------------------------------------------
    model = DLinear(input_size=96, horizon=24)
    y_hat = model(x)   # x: [B, L=96]  ->  y_hat: [B, H=24]

Forward returns the forecast tensor directly (no `hiddens` dict —
NF baselines do not feed Peak-VQ in v04). Loss + training loop live in
`experiments/v04_full_baseline_comparison/02_nf_train.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import HORIZON, INPUT_SIZE


class _MovingAverage(nn.Module):
    """Centred 1-D moving average with edge replication padding.

    Padding pattern reproduces the published DLinear behaviour: replicate
    the first / last value `(kernel-1)//2` times before averaging so the
    output length matches the input length.
    """

    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"DLinear MA kernel must be odd, got {kernel_size}")
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L]
        front = x[:, :1].repeat(1, self.pad)
        end = x[:, -1:].repeat(1, self.pad)
        x_padded = torch.cat([front, x, end], dim=-1)        # [B, L + 2*pad]
        return self.avg(x_padded.unsqueeze(1)).squeeze(1)    # [B, L]


class _SeriesDecomposition(nn.Module):
    """Split x into (seasonal, trend) where trend = MA(x), seasonal = x - trend."""

    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        self.ma = _MovingAverage(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.ma(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """DLinear univariate forecaster.

    Params
    ------
    input_size : int   default ``config.INPUT_SIZE`` = 96
    horizon    : int   default ``config.HORIZON``    = 24
    kernel_size: int   moving-average kernel; paper default = 25
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        horizon: int = HORIZON,
        kernel_size: int = 25,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.decomp = _SeriesDecomposition(kernel_size=kernel_size)
        self.linear_seasonal = nn.Linear(input_size, horizon)
        self.linear_trend = nn.Linear(input_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L] -> y_hat: [B, H]
        seasonal, trend = self.decomp(x)
        return self.linear_seasonal(seasonal) + self.linear_trend(trend)
