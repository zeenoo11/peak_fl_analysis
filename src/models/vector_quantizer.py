"""Standalone Vector Quantizer with EMA codebook updates.

Design choices vs. v10's `DecompCB`:
    - EMA-based codebook update (no codebook gradient) -> more stable, less
      collapse than commit+codebook gradient combo.
    - Single codebook, no per-stack split (MVP scope: h_generic only).
    - Straight-through estimator preserved so backbone gradients flow.
    - Reports codebook utilization (% of unique codes per batch).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    """VQ layer with EMA codebook updates.

    Args:
        num_embeddings:  M (codebook size).
        embedding_dim:   D (latent dim, = backbone d_model).
        commitment_beta: weight on encoder commit loss (z -> sg(q)).
        decay:           EMA decay rate (0.99 = standard).
        eps:             smoothing for codebook normalization.
    """

    def __init__(
        self,
        num_embeddings: int = 32,
        embedding_dim: int = 64,
        commitment_beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_beta = commitment_beta
        self.decay = decay
        self.eps = eps

        embed = torch.randn(num_embeddings, embedding_dim) * 0.1
        self.register_buffer("codebook", embed)
        self.register_buffer("ema_count", torch.zeros(num_embeddings))
        self.register_buffer("ema_weight", embed.clone())

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | torch.Tensor]]:
        """Quantize z via nearest-neighbor lookup; EMA-update codebook.

        Args:
            z: encoder output [B, D].

        Returns:
            z_st:        straight-through quantized output [B, D].
            commit_loss: scalar commitment loss tensor.
            info:        dict with 'utilization', 'indices', 'perplexity'.
        """
        # L2 distance: ||z - e||^2
        dist = (
            z.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * z @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1)
        )  # [B, M]
        indices = dist.argmin(dim=1)  # [B]
        z_q = self.codebook[indices]  # [B, D]

        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(indices, self.num_embeddings).float()  # [B, M]
                count = onehot.sum(dim=0)  # [M]
                self.ema_count.mul_(self.decay).add_(count, alpha=1.0 - self.decay)

                w = onehot.t() @ z.detach()  # [M, D]
                self.ema_weight.mul_(self.decay).add_(w, alpha=1.0 - self.decay)

                n = self.ema_count.sum()
                normalized = (self.ema_count + self.eps) / (
                    n + self.num_embeddings * self.eps
                ) * n
                self.codebook.copy_(self.ema_weight / normalized.unsqueeze(1))

        commit_loss = self.commitment_beta * F.mse_loss(z, z_q.detach())
        z_st = z + (z_q - z).detach()

        with torch.no_grad():
            unique = indices.unique().numel()
            utilization = unique / self.num_embeddings
            probs = onehot.mean(dim=0) if self.training else (
                F.one_hot(indices, self.num_embeddings).float().mean(dim=0)
            )
            entropy = -(probs * (probs + 1e-12).log()).sum()
            perplexity = entropy.exp().item()

        info = {
            "utilization": float(utilization),
            "indices": indices.detach(),
            "perplexity": float(perplexity),
        }
        return z_st, commit_loss, info
