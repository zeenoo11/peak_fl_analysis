"""Minimal NHITS (3 stacks: low/mid/high frequency).

Ported from Peak_Analysis/experiments/federated/v10_b3_nhits_novq.py.
Stripped to the v11 essentials. Adds peak_aux head wrapper for fair comparison
with NBEATSxAux (T2 arm).
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
