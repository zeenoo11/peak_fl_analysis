"""FedProto (Tan et al., AAAI 2022) adapted to forecasting.

Reference: Yue Tan et al., "FedProto: Federated Prototype Learning across
Heterogeneous Clients", AAAI 2022. arxiv:2105.00243.

The published FedProto is a classification algorithm — each client computes
**per-class** prototypes (mean of latents over each class) and the server
aggregates these per-class prototypes by class-weighted average across
clients. The personal client model is then regularised toward the global
prototypes via an MSE-on-latents term added to the local loss.

Forecasting adaptation
----------------------
There are no class labels in residential load forecasting, so the
"per-class" abstraction is replaced with **per-cluster prototypes**:

- Clusters are defined as ``KMeans(K)`` on latents — at the start of each
  round the server holds K global prototypes; each batch's latent assigns
  to its nearest prototype, and the prototype-alignment regulariser pulls
  the latent toward the assigned prototype.
- During each round each client locally runs ``KMeans(K)`` on its own
  training-window h_g latents and reports per-cluster centroids + counts;
  the server aggregates these per-cluster centroids by **count-weighted
  average across clients** (equivalent to the per-class average in the
  original paper).
- Backbone weights are still federated by FedAvg in parallel with the
  prototype aggregation — this matches Tan et al.'s setup where both the
  representation network and the per-class prototypes are aggregated.

Cluster-to-cluster identity across clients
------------------------------------------
A subtle issue: KMeans cluster-IDs are arbitrary and not aligned across
clients. We solve this by initialising every client's local KMeans from
the round-start global prototypes (``init=global_prototypes``,
``n_init=1``); this anchors the local clusters to the global ones so
cluster ID `c` means the same thing on every client (provided KMeans
converges quickly, which it does given the warm start).

Local loss
----------
Per batch:

    L = MAE(y_hat, y) + λ_proto * MSE(h_g, global_prototype[nearest_cluster])

where ``nearest_cluster = argmin_c || h_g - global_prototype[c] ||₂``,
computed against the **round-start** global prototypes (not the
local-step ones — same convention as FedProx's prox anchor).

We default ``λ_proto = 0.1`` (small enough to leave the MAE objective
dominant; documented in the calling script).

Cold inference
--------------
For a held-out cold apt, no W5 hybrid correction is applied — FedProto's
contribution is the prototype-aligned representation, not a peak-shape
template. We forward each cold window, compute h_g, and **route via
1-NN on the global prototypes** as a diagnostic; the cold output is the
**raw forecast ŷ_base** (denormalised to kW). The diagnostic block
records the cold cluster assignment histogram so we can see whether the
prototype representation generalises across cold apts.

Public surface
--------------
- ``FedProtoConfig``     — FLConfig + ``K`` (n prototypes), ``lambda_proto``.
- ``train_fedproto(...)`` — round loop returning the same dict shape as
  the other v04 FL algorithms, with extras: ``global_prototypes``,
  ``prototype_diagnostics``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from fl.base import (
    DEVICE,
    ClientData,
    FLConfig,
    FLHistory,
    apply_state_dict,
    build_clients,
    client_loader,
    clone_state_dict,
    evaluate_cold,
    init_backbone,
    weighted_average,
)


@dataclass
class FedProtoConfig(FLConfig):
    """FedProto hyperparameters: FLConfig + ``K`` and ``lambda_proto``.

    Defaults
    --------
    ``K = 32`` to match the v01/v02 codebook size (so the prototype-axis
    sensitivity sweep in 03_m_sensitivity.py is a meaningful peer).
    ``lambda_proto = 0.1`` keeps MAE dominant; FedProto paper §5 used
    ``λ ∈ [0.01, 1]`` with similar magnitudes for image tasks.
    """

    K: int = 32
    lambda_proto: float = 0.1


# ---------------------------------------------------------------------------
# Local-step helpers
# ---------------------------------------------------------------------------


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _local_train_with_proto(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    n_epochs: int,
    use_amp: bool,
    global_prototypes: torch.Tensor,
    lambda_proto: float,
) -> dict:
    """Run ``n_epochs`` of MAE-loss SGD with prototype-alignment regulariser.

    ``global_prototypes`` is ``[K, D]`` and lives on ``DEVICE``. Per batch
    we compute ``c_batch = argmin_c || h_g - global_prototypes[c] ||₂``
    against the **round-start** global prototypes (unchanged across the
    n_epochs of this client) and add ``λ_proto * MSE(h_g, global_proto[c_batch])``.
    """
    use_amp = use_amp and (DEVICE.type == "cuda")
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp else _NullCtx()
    )
    model.train()
    n_batches = 0
    sum_main, sum_proto = 0.0, 0.0
    proto_t = global_prototypes.detach()  # frozen anchor
    for _ in range(n_epochs):
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                y_hat, hiddens = model(x)
                main = F.l1_loss(y_hat, y)
                h_g = hiddens["h_generic"]
                # Pairwise sq-distance: [B, K]. Assign each window to nearest global prototype.
                d2 = (
                    h_g.float().pow(2).sum(1, keepdim=True)
                    - 2.0 * h_g.float() @ proto_t.t()
                    + proto_t.pow(2).sum(1)
                )
                c_batch = d2.argmin(dim=1)
                target = proto_t[c_batch]
                proto_loss = F.mse_loss(h_g.float(), target)
                loss = main + lambda_proto * proto_loss
            loss.backward()
            optimizer.step()
            sum_main += float(main.item())
            sum_proto += float(proto_loss.item())
            n_batches += 1
    return {
        "n_batches": n_batches,
        "main_loss_mean": sum_main / max(n_batches, 1),
        "proto_loss_mean": sum_proto / max(n_batches, 1),
    }


def _local_h_g_centroids(
    model: torch.nn.Module,
    loader: DataLoader,
    K: int,
    init_centroids: np.ndarray,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Run KMeans(K) on this client's h_g latents, warm-started from the
    round-start global prototypes so cluster IDs stay aligned across clients.

    Returns (centroids [K, D], counts [K]). Using ``n_init=1`` because the
    warm start is the only initialisation we want; multiple inits would
    re-shuffle cluster IDs.
    """
    use_amp = use_amp and (DEVICE.type == "cuda")
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp else _NullCtx()
    )
    model.eval()
    h_chunks = []
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(DEVICE, non_blocking=True)
            with amp_ctx:
                _yh, hiddens = model(x)
            h_chunks.append(hiddens["h_generic"].float().cpu().numpy())
    h_arr = np.concatenate(h_chunks, axis=0).astype(np.float32)
    if len(h_arr) < K:
        # Edge case: if a client has fewer windows than K we pad the centroids
        # with the warm-start init for the missing slots and use counts=0.
        km = KMeans(n_clusters=len(h_arr), init="k-means++", n_init=10).fit(h_arr)
        centroids = init_centroids.copy()
        centroids[: len(h_arr)] = km.cluster_centers_.astype(np.float32)
        labels = km.labels_
        counts = np.zeros(K, dtype=np.int64)
        bc = np.bincount(labels, minlength=len(h_arr)).astype(np.int64)
        counts[: len(h_arr)] = bc
        return centroids.astype(np.float32), counts
    km = KMeans(n_clusters=K, init=init_centroids.astype(np.float32), n_init=1).fit(h_arr)
    counts = np.bincount(km.labels_, minlength=K).astype(np.int64)
    return km.cluster_centers_.astype(np.float32), counts


def _aggregate_prototypes(
    centroids_by_client: list[np.ndarray],
    counts_by_client: list[np.ndarray],
) -> np.ndarray:
    """Per-cluster count-weighted average of client centroids -> global prototypes.

    Equivalent to FedProto's per-class average: for each cluster c,
    weighted mean across clients of the c-th centroid, weights = c-th
    count on that client. Empty clusters (count=0 on every client) keep
    the average of their inputs (numerically the same as the previous
    round's prototype because counts=0 yields a uniform fallback).
    """
    K, D = centroids_by_client[0].shape
    out = np.zeros((K, D), dtype=np.float32)
    for c in range(K):
        weights = np.array([cnts[c] for cnts in counts_by_client], dtype=np.float64)
        if weights.sum() <= 0:
            # Fallback: equal-weighted mean (preserves the current cluster geometry).
            out[c] = np.mean([cs[c] for cs in centroids_by_client], axis=0)
            continue
        weights = weights / weights.sum()
        out[c] = np.sum([cs[c] * w for cs, w in zip(centroids_by_client, weights)], axis=0)
    return out


def _gather_h_g_from_one_loader(
    model: torch.nn.Module, loader: DataLoader, use_amp: bool,
) -> np.ndarray:
    use_amp = use_amp and (DEVICE.type == "cuda")
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp else _NullCtx()
    )
    h_chunks = []
    model.eval()
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(DEVICE, non_blocking=True)
            with amp_ctx:
                _yh, hiddens = model(x)
            h_chunks.append(hiddens["h_generic"].float().cpu().numpy())
    return np.concatenate(h_chunks, axis=0).astype(np.float32)


def _initial_prototypes_from_clients(
    model: torch.nn.Module,
    clients: list[ClientData],
    K: int,
    batch_size: int,
    use_amp: bool,
    seed: int,
) -> np.ndarray:
    """One-shot KMeans(K) on a pooled sample of all clients' h_g latents to
    initialise the global prototypes at round 0. Pools at most ~10000 latents
    per client to keep the cost bounded; for 80 clients × few hundred windows
    each this is the entire dataset on UMass.
    """
    h_pool = []
    for client in clients:
        loader = DataLoader(client.train_set, batch_size=batch_size, shuffle=False)
        h = _gather_h_g_from_one_loader(model, loader, use_amp=use_amp)
        if len(h) > 10_000:
            rng = np.random.default_rng(seed)
            h = h[rng.choice(len(h), 10_000, replace=False)]
        h_pool.append(h)
    h_all = np.concatenate(h_pool, axis=0)
    km = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=seed).fit(h_all)
    return km.cluster_centers_.astype(np.float32)


# ---------------------------------------------------------------------------
# Round loop
# ---------------------------------------------------------------------------


def train_fedproto(
    train_apts: list[str],
    cold_apts: list[str],
    cfg: FedProtoConfig,
) -> dict:
    """Run FedProto for ``cfg.rounds`` rounds and return a result dict.

    Same shape as ``train_fedavg`` so v04's aggregator can read it
    uniformly, with the additional ``prototype_diagnostics`` block:
    final-prototype norms, per-cluster total counts (summed across the
    last round's clients), and cold-side cluster assignment histogram.
    """
    clients: list[ClientData] = build_clients(train_apts)
    if not clients:
        raise RuntimeError("FedProto: no train clients (all apts missing?)")

    # Backbone init shared across clients — matches FedAvg.
    global_model = init_backbone(seed=cfg.seed)
    global_state = clone_state_dict(global_model.state_dict())

    # Round-0 global prototypes from a pooled KMeans on the initial latents.
    print(f"[FedProto] computing initial prototypes (K={cfg.K}) from pooled h_g...")
    apply_state_dict(global_model, global_state)
    global_prototypes_np = _initial_prototypes_from_clients(
        global_model, clients, K=cfg.K, batch_size=cfg.batch_size,
        use_amp=cfg.use_amp, seed=cfg.seed,
    )
    print(f"[FedProto] initial prototype norms: "
          f"min={np.linalg.norm(global_prototypes_np, axis=1).min():.3f}  "
          f"max={np.linalg.norm(global_prototypes_np, axis=1).max():.3f}")

    history = FLHistory()
    last_round_counts: list[np.ndarray] = []
    for r in range(1, cfg.rounds + 1):
        # Snapshot the round-start global prototypes (frozen anchor for this round).
        proto_anchor = torch.from_numpy(global_prototypes_np).to(DEVICE)

        local_states: list[dict] = []
        local_weights: list[float] = []
        local_centroids: list[np.ndarray] = []
        local_counts: list[np.ndarray] = []
        round_main_sum, round_proto_sum, round_n = 0.0, 0.0, 0

        # Optional client sampling (default = all clients participate).
        participating = clients
        if cfg.clients_per_round > 0 and cfg.clients_per_round < len(clients):
            torch.manual_seed(cfg.seed * 10_000 + r)
            idx = torch.randperm(len(clients))[: cfg.clients_per_round].tolist()
            participating = [clients[i] for i in idx]

        for client in participating:
            apply_state_dict(global_model, global_state)
            optimizer = torch.optim.Adam(
                global_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
            )
            loader = client_loader(client, cfg.batch_size, shuffle=True)

            diag = _local_train_with_proto(
                global_model, loader, optimizer,
                n_epochs=cfg.local_epochs, use_amp=cfg.use_amp,
                global_prototypes=proto_anchor, lambda_proto=cfg.lambda_proto,
            )

            # After local training, compute this client's h_g centroids
            # warm-started from the round-start global prototypes.
            no_shuffle_loader = DataLoader(client.train_set, batch_size=cfg.batch_size, shuffle=False)
            cents, cnts = _local_h_g_centroids(
                global_model, no_shuffle_loader, K=cfg.K,
                init_centroids=global_prototypes_np, use_amp=cfg.use_amp,
            )

            local_states.append(clone_state_dict(global_model.state_dict()))
            local_weights.append(float(client.n_train_windows))
            local_centroids.append(cents)
            local_counts.append(cnts)

            round_main_sum += diag["main_loss_mean"] * diag["n_batches"]
            round_proto_sum += diag["proto_loss_mean"] * diag["n_batches"]
            round_n += diag["n_batches"]

        # Backbone aggregation (FedAvg).
        global_state = weighted_average(local_states, local_weights)
        # Prototype aggregation (per-cluster count-weighted mean).
        global_prototypes_np = _aggregate_prototypes(local_centroids, local_counts)
        last_round_counts = local_counts

        history.append(
            round_idx=r,
            train_loss=round_main_sum / max(round_n, 1),
            n_clients=len(participating),
            extra={
                "proto_loss_mean": round_proto_sum / max(round_n, 1),
                "global_prototype_norm_mean": float(np.linalg.norm(global_prototypes_np, axis=1).mean()),
            },
        )

    # ---- Cold inference: forward + 1-NN routing on the global prototypes ----
    apply_state_dict(global_model, global_state)
    cold_metrics = evaluate_cold(global_model, cold_apts, use_amp=cfg.use_amp)

    # Diagnostic: cold cluster-assignment histogram on the global prototypes.
    proto_anchor_np = global_prototypes_np
    cold_h_chunks: list[np.ndarray] = []
    for apt in cold_apts:
        # Reuse the same loader used by evaluate_cold via a minimal local replay;
        # we ONLY need cold h_g for the histogram, so this is a separate fwd pass.
        try:
            from dataloader.umass import HouseholdDataset, load_apartment_hourly
            from config import HORIZON, TRAIN_RATIO
            series = load_apartment_hourly(apt).values.astype(np.float32)
            n = len(series); train_end = int(n * TRAIN_RATIO); seg = series[:train_end]
            m_ = float(seg.mean()); s_ = float(seg.std()) if seg.std() > 1e-8 else 1.0
            ds = HouseholdDataset(seg, m_, s_, stride=HORIZON)
            if len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
            h = _gather_h_g_from_one_loader(global_model, loader, use_amp=cfg.use_amp)
            cold_h_chunks.append(h)
        except FileNotFoundError:
            continue
    if cold_h_chunks:
        cold_h = np.concatenate(cold_h_chunks, axis=0)
        d2 = (
            np.sum(cold_h ** 2, axis=1, keepdims=True)
            - 2 * cold_h @ proto_anchor_np.T
            + np.sum(proto_anchor_np ** 2, axis=1)
        )
        cold_cluster = d2.argmin(axis=1)
        cold_usage = np.bincount(cold_cluster, minlength=cfg.K).tolist()
    else:
        cold_usage = [0] * cfg.K

    train_assign_total = (
        np.sum(np.stack(last_round_counts, axis=0), axis=0).tolist()
        if last_round_counts else [0] * cfg.K
    )
    proto_diag = {
        "K": int(cfg.K),
        "global_prototype_norms": np.linalg.norm(proto_anchor_np, axis=1).tolist(),
        "global_prototype_norm_mean": float(np.linalg.norm(proto_anchor_np, axis=1).mean()),
        "global_prototype_norm_min": float(np.linalg.norm(proto_anchor_np, axis=1).min()),
        "global_prototype_norm_max": float(np.linalg.norm(proto_anchor_np, axis=1).max()),
        "train_assignment_histogram_last_round": train_assign_total,
        "cold_assignment_histogram": cold_usage,
        "cold_n_clusters_used": int(sum(1 for c in cold_usage if c > 0)),
    }

    return {
        "algorithm": "fedproto",
        "config": cfg.__dict__,
        "history": history.as_dict(),
        "cold_metrics": cold_metrics,
        "n_train_clients": len(clients),
        "final_state_dict": global_state,
        "prototype_diagnostics": proto_diag,
        "global_prototypes": proto_anchor_np,  # numpy [K, D] — caller saves to .npz
    }
