"""Post-hoc KMeans++ Vector Quantizer (replaces EMA path for v11).

Usage:
    vq = VectorQuantizerKMeans(num_embeddings=32, embedding_dim=64)
    diag = vq.fit(z_train)        # z: torch.Tensor [N, D]
    z_q, indices = vq(z_query)    # NN lookup, no STE
"""

from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.cluster import KMeans


class VectorQuantizerKMeans(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 32,
        embedding_dim: int = 64,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        self.register_buffer("codebook", torch.zeros(num_embeddings, embedding_dim))
        self.register_buffer("counts", torch.zeros(num_embeddings, dtype=torch.long))
        self._is_fit = False

    def fit(self, z: torch.Tensor) -> dict:
        if z.dim() != 2:
            raise ValueError(f"expected [N, D], got {z.shape}")
        if z.shape[1] != self.embedding_dim:
            raise ValueError(f"embedding_dim mismatch: {z.shape[1]} vs {self.embedding_dim}")
        z_np = z.detach().cpu().numpy()
        km = KMeans(
            n_clusters=self.num_embeddings,
            init="k-means++",
            n_init=10,
            random_state=self.random_state,
        ).fit(z_np)
        self.codebook.copy_(torch.from_numpy(km.cluster_centers_).float())
        bincount = torch.bincount(
            torch.from_numpy(km.labels_).long(), minlength=self.num_embeddings
        )
        self.counts.copy_(bincount.long())
        self._is_fit = True
        utilization = float((self.counts > 0).sum().item()) / self.num_embeddings
        probs = self.counts.float() / max(int(self.counts.sum().item()), 1)
        entropy = -(probs * (probs + 1e-12).log()).sum().item()
        perplexity = float(torch.exp(torch.tensor(entropy)).item())
        return {
            "n_fit_samples": int(z.shape[0]),
            "utilization": utilization,
            "perplexity": perplexity,
            "k_min": int(self.counts.min().item()),
            "k_max": int(self.counts.max().item()),
            "kmeans_inertia": float(km.inertia_),
        }

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._is_fit:
            raise RuntimeError("call .fit(...) before forward")
        dist = (
            z.pow(2).sum(1, keepdim=True)
            - 2.0 * z @ self.codebook.t()
            + self.codebook.pow(2).sum(1)
        )
        idx = dist.argmin(dim=1)
        return self.codebook[idx], idx
