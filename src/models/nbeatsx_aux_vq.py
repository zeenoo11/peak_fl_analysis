"""NBEATSxAux + VectorQuantizerEMA on h_generic.

Combines two existing modules without modifying either:
    - ``NBEATSxAux`` (src/models/nbeatsx_aux.py) — NBEATSx backbone + ``PeakAuxHead``
      attached to ``h_generic`` (default latent source).
    - ``GenericStackWithVQ`` (src/models/nbeatsx_vq.py) — drop-in replacement for
      ``stack_generic`` that quantises ``h_generic`` via ``VectorQuantizerEMA``.

The resulting module is what plan v09-01 §3 federates: a single VQ codebook on
``stack_generic`` lives end-to-end inside the NBEATSxAux training loop, so
backbone gradients and the EMA codebook update happen jointly per local step.
The aux head can be disabled to support the ``-noaux`` ablation cell.

State-dict layout (load-bearing):
    stack_trend.*               — unchanged trend stack
    stack_seasonal.*            — unchanged seasonal stack
    stack_generic.{fc1..fc4, proj}.*  — generic stack MLP + projection
    stack_generic.vq.codebook        — (M, D) codebook entries           [buffer]
    stack_generic.vq.ema_count       — (M,)   EMA cluster counts         [buffer]
    stack_generic.vq.ema_weight      — (M, D) EMA cluster sums           [buffer]
    aux_head.{shared, amp_head, hr_head}.*  — present only when use_aux_head=True

Forward signature (single fixed shape, all keys always present):
    forward(x) -> {
        "y_hat":    [B, H]                — point forecast (sum of stack forecasts)
        "h_generic":[B, D]                — generic stack hidden (post-fc4, pre-VQ)
        "vq_state": {
            "commit_loss": scalar tensor  — β · MSE(z, sg(q))
            "utilization": float          — fraction of unique codes used in batch
            "perplexity":  float          — exp(entropy of code histogram)
            "indices":     [B] long       — selected codebook index per sample
        }
        "aux":      (amp_pred, hr_pred) | None
                                          — None iff use_aux_head=False
    }

Latent source is forced to ``h_generic`` (matches the VQ insertion point); the
``latent_source='h_concat'`` option from ``NBEATSxAux`` is intentionally not
supported here because the VQ embedding_dim is D_MODEL=64, not 3*D_MODEL.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from config import D_MODEL, HORIZON, INPUT_SIZE, N_HARMONICS, N_POLYNOMIALS
from models.nbeatsx import (
    GenericBasis,
    NBEATSxStack,
    SeasonalityBasis,
    TrendBasis,
)
from models.nbeatsx_vq import GenericStackWithVQ
from models.peak_aux_head import PeakAuxHead


class NBEATSxAuxVQ(nn.Module):
    """NBEATSxAux with ``GenericStackWithVQ`` swapped in for ``stack_generic``.

    Args:
        seq_len:         input window length (default INPUT_SIZE=96).
        horizon:         forecast horizon (default HORIZON=24).
        d_model:         per-stack hidden dim (default D_MODEL=64).
        n_polynomials:   trend basis polynomial degree (default N_POLYNOMIALS).
        n_harmonics:     seasonal basis harmonic count (default N_HARMONICS).
        num_embeddings:  VQ codebook size M (default 32, v06 invariant).
        commitment_beta: VQ-VAE commitment loss weight (default 0.25).
        use_aux_head:    when False, do not build ``aux_head`` and return
                         ``aux=None`` from forward; saves ~3 kB of params and
                         disables the peak-aux multi-task signal entirely.

    Note:
        Latent source is fixed to ``h_generic`` because the VQ embedding_dim
        equals ``d_model``. ``h_concat`` would require a 3·d_model codebook.
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
        use_aux_head: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.d_model = d_model
        self.num_embeddings = num_embeddings
        self.use_aux_head = bool(use_aux_head)

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

        if self.use_aux_head:
            self.aux_head = PeakAuxHead(in_dim=d_model)
        # When use_aux_head=False we deliberately do not register an aux_head
        # attribute, so its parameters do not appear in state_dict() and do
        # not participate in FedAvg backbone aggregation.

    def forward(self, x: torch.Tensor) -> dict:
        """See module docstring for the return dict schema."""
        if x.dim() == 3:
            x = x.squeeze(-1)

        residual = x.clone()
        bc_t, fc_t, _ = self.stack_trend(residual)
        residual = residual - bc_t

        bc_s, fc_s, _ = self.stack_seasonal(residual)
        residual = residual - bc_s

        bc_g, fc_g, h_g, commit_loss, info = self.stack_generic(residual)

        y_hat = fc_t + fc_s + fc_g

        aux: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        if self.use_aux_head:
            amp_pred, hr_pred = self.aux_head(h_g)
            aux = (amp_pred, hr_pred)

        return {
            "y_hat": y_hat,
            "h_generic": h_g,
            "vq_state": {
                "commit_loss": commit_loss,
                "utilization": info["utilization"],
                "perplexity": info["perplexity"],
                "indices": info["indices"],
            },
            "aux": aux,
        }
