"""NBEATSx with attached peak_aux head (arms T2/T3)."""

from __future__ import annotations

import torch
import torch.nn as nn

from config import D_MODEL
from models.nbeatsx import MinimalNBEATSx
from models.peak_aux_head import PeakAuxHead


class NBEATSxAux(nn.Module):
    """NBEATSx + PeakAuxHead. Latent source is h_generic or h_concat."""

    def __init__(self, latent_source: str = "h_generic") -> None:
        super().__init__()
        if latent_source not in ("h_generic", "h_concat"):
            raise ValueError(f"unknown latent_source: {latent_source}")
        self.latent_source = latent_source
        self.backbone = MinimalNBEATSx()
        in_dim = D_MODEL if latent_source == "h_generic" else 3 * D_MODEL
        self.aux_head = PeakAuxHead(in_dim=in_dim)

    def get_latent(self, hiddens: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.latent_source == "h_generic":
            return hiddens["h_generic"]
        return torch.cat(
            [hiddens["h_trend"], hiddens["h_seasonal"], hiddens["h_generic"]], dim=1
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict, tuple]:
        y_hat, hiddens = self.backbone(x)
        h = self.get_latent(hiddens)
        amp_pred, hr_pred = self.aux_head(h)
        return y_hat, hiddens, (amp_pred, hr_pred)
