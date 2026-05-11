"""Federated codebook construction helpers (v05 FedCB).

This module provides three functional helpers for the v05 hierarchical
single-shot federated KMeans pipeline. The matching driver lives at
``experiments/v05_fedcb_codebook/01_fedcb_codebook.py``; the plan is
``plans/v05-01_fedcb_codebook.md``.

Why these are *functional* helpers (not a client class)
-------------------------------------------------------
The v04 ``src/fl/base.py`` style is "small, algorithm-agnostic helpers"
plus per-algorithm modules that compose them. v05 adds a *one-shot*
post-hoc step: each apt clusters its own ``h_g`` once, the server
merges, and the server requests a per-cluster residual partial sum.
There is no per-round client state machine, so a class would be
overkill.

Why sklearn KMeans directly (not VectorQuantizerKMeans.fit)
-----------------------------------------------------------
``models.vq_kmeans.VectorQuantizerKMeans.fit`` does not expose
``sample_weight``. Stage 2 needs it (``sample_weight = local cluster
counts``) so we go to sklearn directly. The Stage 2 result is then
*registered* into a ``VectorQuantizerKMeans`` buffer at the call site
(``01_fedcb_codebook.py``) for downstream API compatibility.

Public surface
--------------
- ``local_codebook_step(model, client, K_local, seed)`` — runs a frozen
  forward over one client's train segment (stride=24, batch=512, bf16
  on CUDA) and fits a local KMeans++. Returns
  ``{centroids, counts, h_g, y_hat_z, y_true_z, K_local_i}``. Reads the
  raw series from disk (v04/v05 convention).

- ``local_codebook_step_from_splits(model, train_x, train_y, K_local,
  seed, batch_size, use_amp)`` — v06 variant. Same packet shape as
  ``local_codebook_step`` but consumes a pre-windowed ``(N, INPUT_SIZE)``
  / ``(N, HORIZON)`` numpy pair (already stride=24, already z-normed)
  instead of the disk-reload + ``HouseholdDataset`` path. Used by the
  v06 Phase 2 codebook-stacking driver where ``per_client_split.pkl``
  already carved windows.

- ``merge_local_codebooks(local_packets, M_global, seed)`` — Stage 2.
  Stacks all clients' centroids weighted by their local cluster counts
  and refits to the global ``M_global`` size. Returns
  ``{codebook, stage2_inertia, n_empty_clusters, utilization, perplexity,
  k_min, k_max, stage1_mean_inertia}``.

- ``federated_residual_offsets(local_packets, codebook)`` — each client
  routes its own ``h_g`` against the global codebook (h_g 1-NN, fp32
  numpy distance), computes per-cluster residual partial sums + counts,
  and the server averages cluster-wise. Returns ``offsets ∈ R^{M × H}``.

The helpers are split so the pure-numpy paths are testable without a
real backbone — see ``_local_kmeans`` / ``_extract_h_g`` /
``_extract_h_g_from_windows``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from config import D_MODEL, HORIZON, TRAIN_RATIO
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from fl.base import DEVICE, ClientData, _NullCtx


# ============================================================================
# Stage 1 — local KMeans (pure numpy) and forward-pass extraction
# ============================================================================


def _local_kmeans(
    H_i: np.ndarray,
    K_local: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Fit a local KMeans++ on ``H_i ∈ R^{N_i × D}``.

    Tiny-client fallback: ``K_local_i = min(K_local, N_i)``. sklearn
    raises ``ValueError`` if ``n_clusters > n_samples`` so this fallback
    is required even though UMass apts are typically large enough.

    Returns
    -------
    centroids : (K_local_i, D) fp32
    counts    : (K_local_i,) int64  -- ``Σ counts == N_i``
    inertia   : float
    K_local_i : int
    """
    if H_i.ndim != 2:
        raise ValueError(f"_local_kmeans: expected 2-d H_i, got shape {H_i.shape}")
    N_i = H_i.shape[0]
    if N_i == 0:
        raise ValueError("_local_kmeans: empty H_i")
    K_local_i = int(min(int(K_local), N_i))
    km = KMeans(
        n_clusters=K_local_i,
        init="k-means++",
        n_init=10,
        random_state=int(seed),
    ).fit(H_i.astype(np.float32))
    centroids = km.cluster_centers_.astype(np.float32)
    labels = km.labels_.astype(np.int64)
    counts = np.bincount(labels, minlength=K_local_i).astype(np.int64)
    return centroids, counts, float(km.inertia_), K_local_i


def _extract_h_g(
    model: torch.nn.Module,
    client: ClientData,
    *,
    batch_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward one client's train segment through a frozen NBEATSxAux backbone.

    Builds a fresh ``HouseholdDataset`` on this client's series with the
    given stride (= 24 by default; v01/v02 codebook convention) instead
    of reusing ``client.train_set`` whose stride=1 would over-sample
    overlapping windows.

    Returns ``(h_g, y_hat_z, y_true_z)`` numpy arrays:
        h_g       (N_i, 64)  fp32
        y_hat_z   (N_i, 24)  fp32
        y_true_z  (N_i, 24)  fp32
    """
    series = load_apartment_hourly(client.apt).values.astype(np.float32)
    n = len(series)
    train_end = int(n * TRAIN_RATIO)
    seg = series[:train_end]
    ds = HouseholdDataset(seg, client.mean, client.std, stride=stride)
    if len(ds) == 0:
        return (
            np.zeros((0, D_MODEL), dtype=np.float32),
            np.zeros((0, HORIZON), dtype=np.float32),
            np.zeros((0, HORIZON), dtype=np.float32),
        )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    use_amp = DEVICE.type == "cuda"
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp
        else _NullCtx()
    )
    model.eval()
    h_chunks, yhat_chunks, ytrue_chunks = [], [], []
    for x, y in loader:
        x_dev = x.to(DEVICE, non_blocking=True)
        with torch.no_grad(), amp_ctx:
            y_hat, hiddens, _aux = model(x_dev)
        h_chunks.append(hiddens["h_generic"].float().cpu().numpy())
        yhat_chunks.append(y_hat.float().cpu().numpy())
        ytrue_chunks.append(y.numpy())
    return (
        np.concatenate(h_chunks, axis=0).astype(np.float32),
        np.concatenate(yhat_chunks, axis=0).astype(np.float32),
        np.concatenate(ytrue_chunks, axis=0).astype(np.float32),
    )


def local_codebook_step(
    model: torch.nn.Module,
    client: ClientData,
    K_local: int,
    seed: int,
    *,
    batch_size: int = 512,
    stride: int = HORIZON,
) -> dict[str, Any]:
    """Stage 1 — extract this client's ``h_g`` and fit a local KMeans++.

    Parameters
    ----------
    model      : frozen NBEATSxAux (in eval mode at the call site).
    client     : a ``ClientData`` for one train apt (provides apt name +
                 mean/std for warm-start z-norm; ``client.train_set`` is
                 *not* reused because its stride=1 would over-sample).
    K_local    : target Stage-1 cluster count. Falls back to N_i if
                 N_i < K_local (sklearn raises otherwise).
    seed       : forwarded to ``KMeans(random_state=seed)``.
    batch_size : forward-pass batch size (default 512, matches 09_fix_rerun).
    stride     : codebook fit stride (default 24, matches v01/v02).

    Returns
    -------
    dict with:
        centroids   (K_local_i, 64) fp32
        counts      (K_local_i,)    int64
        h_g         (N_i, 64)       fp32
        y_hat_z     (N_i, 24)       fp32
        y_true_z    (N_i, 24)       fp32
        K_local_i   int             effective K after tiny-client fallback
        inertia     float           Stage-1 inertia for diagnostics

    The driver never aggregates ``h_g`` / ``y_hat_z`` / ``y_true_z``
    across clients on the server — those fields stay attached to the
    packet and are only consumed by ``federated_residual_offsets`` which
    routes locally and uploads partial sums.
    """
    h_g, y_hat_z, y_true_z = _extract_h_g(
        model, client, batch_size=batch_size, stride=stride
    )
    if h_g.shape[0] == 0:
        # Empty client (extremely short series) — emit a zero packet so
        # the driver can skip it without a NoneType branch.
        return {
            "centroids": np.zeros((0, D_MODEL), dtype=np.float32),
            "counts": np.zeros((0,), dtype=np.int64),
            "h_g": h_g,
            "y_hat_z": y_hat_z,
            "y_true_z": y_true_z,
            "K_local_i": 0,
            "inertia": 0.0,
        }
    centroids, counts, inertia, K_local_i = _local_kmeans(h_g, K_local, seed)
    return {
        "centroids": centroids,
        "counts": counts,
        "h_g": h_g,
        "y_hat_z": y_hat_z,
        "y_true_z": y_true_z,
        "K_local_i": K_local_i,
        "inertia": inertia,
    }


# ============================================================================
# Stage 2 — server merge of stacked local centroids
# ============================================================================


def merge_local_codebooks(
    local_packets: list[dict[str, Any]],
    M_global: int,
    seed: int,
) -> dict[str, Any]:
    """Stage 2 — weighted KMeans++ on stacked local centroids.

    Parameters
    ----------
    local_packets : list of dicts as returned by ``local_codebook_step``.
                    Empty packets (``K_local_i == 0``) are skipped.
    M_global      : target global codebook size.
    seed          : forwarded to ``KMeans(random_state=seed)``.

    Returns
    -------
    dict with:
        codebook            (M_global, D) fp32
        stage2_inertia      float
        n_empty_clusters    int   (clusters with 0 sample-weight mass)
        utilization         float (fraction of M_global with non-zero mass)
        perplexity          float
        k_min               int
        k_max               int
        stage1_mean_inertia float (inertia mean across non-empty packets)

    Notes
    -----
    - Empty Stage-2 clusters can occur with very skewed sample-weight
      distributions; their offsets stay 0 (same convention as
      ``vq_kmeans.fit``).
    - ``utilization`` and ``perplexity`` here are computed in
      *sample-weight space* (i.e. how many train windows landed in each
      Stage-2 bucket after routing the input centroids), matching what
      a centralised KMeans would have measured on the original windows.
    """
    nonempty = [p for p in local_packets if int(p["K_local_i"]) > 0]
    if not nonempty:
        raise ValueError("merge_local_codebooks: all client packets are empty")
    P = np.vstack([p["centroids"] for p in nonempty]).astype(np.float32)
    w = np.concatenate([p["counts"] for p in nonempty]).astype(np.float64)
    if P.shape[0] < M_global:
        # Should not happen at the v05 scale (80 clients × K_local ≥ 2 = 160 ≥ M=32),
        # but guard anyway since sklearn raises if n_clusters > n_samples.
        raise ValueError(
            f"merge_local_codebooks: only {P.shape[0]} input centroids vs "
            f"M_global={M_global}; reduce M_global or raise K_local."
        )
    km = KMeans(
        n_clusters=int(M_global),
        init="k-means++",
        n_init=10,
        random_state=int(seed),
    ).fit(P, sample_weight=w)
    codebook = km.cluster_centers_.astype(np.float32)
    labels = km.labels_.astype(np.int64)

    # Sample-weight mass per Stage-2 cluster.
    mass = np.bincount(labels, weights=w, minlength=int(M_global))
    n_empty_clusters = int((mass == 0).sum())
    nonzero_clusters = int((mass > 0).sum())
    utilization = float(nonzero_clusters) / float(M_global)
    total = float(mass.sum())
    if total > 0:
        probs = mass / total
        # entropy is in nats; perplexity = exp(entropy) is in same units as
        # vq_kmeans.fit so cross-script comparisons are apples-to-apples.
        nz = probs > 0
        entropy = float(-(probs[nz] * np.log(probs[nz])).sum())
        perplexity = float(np.exp(entropy))
    else:
        perplexity = 0.0
    nonzero_mass = mass[mass > 0] if (mass > 0).any() else mass
    k_min = int(nonzero_mass.min()) if nonzero_mass.size else 0
    k_max = int(mass.max())

    stage1_mean_inertia = float(
        np.mean([float(p["inertia"]) for p in nonempty])
    )

    return {
        "codebook": codebook,
        "stage2_inertia": float(km.inertia_),
        "n_empty_clusters": n_empty_clusters,
        "utilization": utilization,
        "perplexity": perplexity,
        "k_min": k_min,
        "k_max": k_max,
        "stage1_mean_inertia": stage1_mean_inertia,
    }


# ============================================================================
# Federated residual offsets (Stage 3)
# ============================================================================


def _route_h_g_to_codebook(h_g: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """h_g 1-NN against ``codebook`` (Euclidean, fp32 numpy distance).

    Equivalent to ``eval.cold_helpers.route_R1`` but kept local so this
    module does not import from ``eval.cold_helpers`` (keeps the import
    surface inside ``src/fl`` self-contained).
    """
    # ||a - b||^2 = ||a||^2 - 2 a·b + ||b||^2; argmin over c is invariant
    # to the constant ||a||^2 term. We compute the full distance for
    # numerical clarity at v05 scale (N × M = O(thousands × 32)).
    d = (
        (h_g.astype(np.float32) ** 2).sum(axis=1, keepdims=True)
        - 2.0 * h_g.astype(np.float32) @ codebook.astype(np.float32).T
        + (codebook.astype(np.float32) ** 2).sum(axis=1)
    )
    return d.argmin(axis=1).astype(np.int64)


def federated_residual_offsets(
    local_packets: list[dict[str, Any]],
    codebook: np.ndarray,
) -> np.ndarray:
    """Stage 3 — per-cluster average residual aggregated across clients.

    For each non-empty client packet:
        1. Route ``h_g`` 1-NN against ``codebook`` → ``c*_i ∈ Z^{N_i}``.
        2. Compute per-cluster residual partial sum
           ``r_{i,c} = Σ_{j: c*_i[j]=c} (y_true_z[j] − y_hat_z[j]) ∈ R^H``
           and per-cluster count ``m_{i,c}``.
        3. Upload ``(r_{i,c}, m_{i,c})_{c=0..M-1}`` (this function
           simulates that upload by accumulating in-process).

    Server averages cluster-wise:
        ``o_c = Σ_i r_{i,c} / max(Σ_i m_{i,c}, 1)``

    Empty clusters ⇒ zero offset (matches ``vq_kmeans.fit`` and v01/v02
    convention; ``y_hat_z`` is uncorrected for that route).

    Parameters
    ----------
    local_packets : list of dicts as returned by ``local_codebook_step``.
    codebook      : (M_global, D) fp32, output of ``merge_local_codebooks``.

    Returns
    -------
    offsets : (M_global, H) fp32
    """
    M = int(codebook.shape[0])
    # Probe horizon from the first non-empty packet.
    H = HORIZON
    for p in local_packets:
        if int(p["K_local_i"]) > 0 and p["y_true_z"].shape[0] > 0:
            H = int(p["y_true_z"].shape[1])
            break

    sum_resid = np.zeros((M, H), dtype=np.float64)
    sum_count = np.zeros((M,), dtype=np.int64)
    for p in local_packets:
        if int(p["K_local_i"]) == 0 or p["h_g"].shape[0] == 0:
            continue
        idx = _route_h_g_to_codebook(p["h_g"], codebook)  # (N_i,)
        resid = (p["y_true_z"] - p["y_hat_z"]).astype(np.float64)  # (N_i, H)
        # Per-cluster sums; np.add.at handles repeated indices correctly.
        np.add.at(sum_resid, idx, resid)
        sum_count += np.bincount(idx, minlength=M)

    offsets = np.zeros((M, H), dtype=np.float32)
    nz = sum_count > 0
    offsets[nz] = (sum_resid[nz] / sum_count[nz, None]).astype(np.float32)
    return offsets


# ============================================================================
# v06 variant — splits-based extraction (no disk reload, no HouseholdDataset)
# ============================================================================


def _extract_h_g_from_windows(
    model: torch.nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    batch_size: int,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward pre-windowed (z-normed) tensors through a frozen NBEATSxAux.

    Counterpart of ``_extract_h_g`` but consumes already-windowed numpy
    arrays from the v06 ``per_client_split.pkl`` cache instead of
    re-loading the raw series and rebuilding a ``HouseholdDataset``. This
    avoids two pitfalls of the original ``_extract_h_g`` for v06:

    1. ``HouseholdDataset`` requires the raw series + per-apt mean/std and
       defaults to stride=1; v06 splits already carved windows at
       stride=24, so a re-build would over-sample.
    2. The disk reload of the CSV duplicates work the v06 driver already
       paid via ``build_per_client_splits``.

    Parameters
    ----------
    model      : frozen NBEATSxAux (eval mode at the call site).
    train_x    : (N, INPUT_SIZE) float32 z-normed input windows.
    train_y    : (N, HORIZON)    float32 z-normed target windows.
    batch_size : forward-pass batch size.
    use_amp    : if True and CUDA available, run forward under bf16 autocast.

    Returns
    -------
    h_g       : (N, 64) fp32  -- NBEATSxAux's ``h_generic`` latent.
    y_hat_z   : (N, 24) fp32  -- z-norm forecast.
    y_true_z  : (N, 24) fp32  -- copy of ``train_y`` (kept symmetric with
                                ``_extract_h_g``).
    """
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError(
            f"_extract_h_g_from_windows: train_x ({train_x.shape}) and "
            f"train_y ({train_y.shape}) row count mismatch."
        )
    n = int(train_x.shape[0])
    if n == 0:
        return (
            np.zeros((0, D_MODEL), dtype=np.float32),
            np.zeros((0, HORIZON), dtype=np.float32),
            np.zeros((0, HORIZON), dtype=np.float32),
        )
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_amp and DEVICE.type == "cuda")
        else _NullCtx()
    )
    model.eval()
    h_chunks, yhat_chunks = [], []
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(train_x[i : i + batch_size]).to(DEVICE)
        with torch.no_grad(), amp_ctx:
            ret = model(xb)
        # NBEATSxAux returns (y_hat, hiddens, aux); MinimalNBEATSx returns
        # (y_hat, hiddens). Support both.
        y_hat = ret[0]
        hiddens = ret[1]
        h_chunks.append(hiddens["h_generic"].float().cpu().numpy())
        yhat_chunks.append(y_hat.float().cpu().numpy())
    return (
        np.concatenate(h_chunks, axis=0).astype(np.float32),
        np.concatenate(yhat_chunks, axis=0).astype(np.float32),
        train_y.astype(np.float32),
    )


def local_codebook_step_from_splits(
    model: torch.nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    K_local: int,
    seed: int,
    *,
    batch_size: int = 512,
    use_amp: bool = True,
) -> dict[str, Any]:
    """Stage 1 (v06 variant) — fit a local KMeans++ from pre-windowed splits.

    Drop-in replacement for ``local_codebook_step`` in v06 contexts where
    the windows are already in memory (``per_client_split.pkl``) and the
    backbone state has been frozen for the round-level FL evaluation.

    The packet shape exactly matches ``local_codebook_step``'s output so
    ``merge_local_codebooks`` and ``federated_residual_offsets`` can be
    fed without modification — keeping the *federation contract* of v05
    intact (raw h_g and raw residuals never leave the client; only
    (centroids, counts) and (residual partial sum, count) tuples
    upload).

    Parameters
    ----------
    model      : frozen NBEATSxAux (eval mode at the call site).
    train_x    : (N, INPUT_SIZE) float32 z-normed input windows.
    train_y    : (N, HORIZON)    float32 z-normed target windows.
    K_local    : target Stage-1 cluster count. Falls back to N if
                 N < K_local (sklearn raises otherwise).
    seed       : forwarded to ``KMeans(random_state=seed)``.
    batch_size : forward-pass batch size (default 512).
    use_amp    : enable bf16 autocast on CUDA (auto-disabled on CPU).

    Returns
    -------
    Same dict shape as ``local_codebook_step``.
    """
    h_g, y_hat_z, y_true_z = _extract_h_g_from_windows(
        model, train_x, train_y, batch_size=batch_size, use_amp=use_amp
    )
    if h_g.shape[0] == 0:
        return {
            "centroids": np.zeros((0, D_MODEL), dtype=np.float32),
            "counts": np.zeros((0,), dtype=np.int64),
            "h_g": h_g,
            "y_hat_z": y_hat_z,
            "y_true_z": y_true_z,
            "K_local_i": 0,
            "inertia": 0.0,
        }
    centroids, counts, inertia, K_local_i = _local_kmeans(h_g, K_local, seed)
    return {
        "centroids": centroids,
        "counts": counts,
        "h_g": h_g,
        "y_hat_z": y_hat_z,
        "y_true_z": y_true_z,
        "K_local_i": K_local_i,
        "inertia": inertia,
    }


