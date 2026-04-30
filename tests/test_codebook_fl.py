"""Pytest for ``src/fl/codebook_fl.py`` — the v05 federated codebook helpers.

Three cases (plan step 5):
  a. Determinism: ``merge_local_codebooks`` is bit-identical under fixed seed.
  b. Stage-2 sanity at ``K_local = M``: utilization ≥ 0.95 on synthetic
     gaussian clusters.
  c. Tiny-client fallback: ``_local_kmeans`` with ``N_i < K_local`` does
     not raise; it sets ``K_local_i = N_i``.

All three tests use the pure-numpy inner helpers (``_local_kmeans`` and the
``federated_residual_offsets`` /  ``merge_local_codebooks`` functions); the
model-dependent ``local_codebook_step`` is exercised by the smoke run, not
unit tests, because mocking NBEATSxAux's tri-tuple forward is more brittle
than just splitting the helper.
"""

from __future__ import annotations

import sys
from pathlib import Path

# pyproject.toml sets pythonpath = ["src"] but be explicit so the tests can
# also be run via ``uv run python -m pytest`` from any cwd.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pytest

from fl.codebook_fl import (
    _local_kmeans,
    federated_residual_offsets,
    merge_local_codebooks,
)


# ----------------------------------------------------------------------------
# Synthetic data helpers
# ----------------------------------------------------------------------------


def _make_synthetic_clients(
    *,
    n_clients: int,
    n_per_client: int,
    K_local: int,
    D: int,
    seed: int,
    horizon: int = 24,
) -> list[dict]:
    """Generate ``n_clients`` packets of synthetic gaussian clusters.

    Each client draws ``K_local`` ground-truth centers in ℝ^D from a
    standard normal × 5, then samples ``n_per_client / K_local`` points
    around each center with σ=0.5. We compute (centroids, counts) via the
    real ``_local_kmeans`` so the returned packets are exactly what
    ``local_codebook_step`` would emit on real data.

    Each packet also carries dummy ``y_hat_z`` / ``y_true_z`` shaped
    ``(n_per_client, horizon)`` so ``federated_residual_offsets`` has
    something to integrate against.
    """
    rng = np.random.RandomState(seed)
    packets: list[dict] = []
    for ci in range(n_clients):
        # Distinct clients have distinct ground-truth centers so the merged
        # Stage-2 codebook has to span N_clients × K_local distinct modes.
        client_seed = seed * 1000 + ci
        sub_rng = np.random.RandomState(client_seed)
        per_cluster = n_per_client // K_local
        H_chunks = []
        for k in range(K_local):
            mu = sub_rng.randn(D).astype(np.float32) * 5.0
            chunk = mu[None, :] + sub_rng.randn(per_cluster, D).astype(np.float32) * 0.5
            H_chunks.append(chunk)
        H_i = np.concatenate(H_chunks, axis=0).astype(np.float32)
        # Shuffle so labels don't trivially recover by index order.
        perm = sub_rng.permutation(H_i.shape[0])
        H_i = H_i[perm]
        centroids, counts, inertia, K_local_i = _local_kmeans(
            H_i, K_local=K_local, seed=client_seed
        )
        # Dummy targets: y_true - y_hat = sin(2πi/n) profile so different
        # clients have non-trivial residual signatures.
        idx = np.arange(H_i.shape[0], dtype=np.float32)[:, None]
        h_arr = np.arange(horizon, dtype=np.float32)[None, :]
        y_hat_z = np.zeros((H_i.shape[0], horizon), dtype=np.float32)
        y_true_z = (
            np.sin(2 * np.pi * (idx + ci) / max(H_i.shape[0], 1)) * h_arr / horizon
        ).astype(np.float32)
        packets.append({
            "centroids": centroids,
            "counts": counts,
            "h_g": H_i,
            "y_hat_z": y_hat_z,
            "y_true_z": y_true_z,
            "K_local_i": K_local_i,
            "inertia": inertia,
        })
    return packets


# ----------------------------------------------------------------------------
# (a) Determinism of Stage 2
# ----------------------------------------------------------------------------


def test_merge_local_codebooks_deterministic():
    """Two calls with identical seed return bit-identical codebooks.

    sklearn KMeans with fixed ``random_state`` and ``n_init=10`` must be
    reproducible — if this regresses, the v05 multi-seed sweep loses its
    "same seed → same codebook" property and the per-seed PAPE numbers
    become irreproducible.
    """
    packets = _make_synthetic_clients(
        n_clients=80, n_per_client=64, K_local=4, D=8, seed=42
    )
    a = merge_local_codebooks(packets, M_global=32, seed=2024)
    b = merge_local_codebooks(packets, M_global=32, seed=2024)
    np.testing.assert_array_equal(a["codebook"], b["codebook"])
    assert a["stage2_inertia"] == b["stage2_inertia"]
    assert a["n_empty_clusters"] == b["n_empty_clusters"]
    # Sanity: a different seed should generally give a different codebook.
    c = merge_local_codebooks(packets, M_global=32, seed=2025)
    assert not np.array_equal(a["codebook"], c["codebook"]), (
        "different seeds produced identical codebook — sklearn KMeans is "
        "not seeding correctly?"
    )


# ----------------------------------------------------------------------------
# (b) Stage-2 sanity at K_local = M
# ----------------------------------------------------------------------------


def test_merge_local_codebooks_high_utilization_when_K_local_eq_M():
    """When each client emits exactly M centroids and clusters are well-separated,
    Stage 2 should keep most of the M global slots populated.

    With 80 clients × 32 well-separated synthetic centers each, the input
    pool to Stage 2 is 80 × 32 = 2560 points spread across many distinct
    modes. A KMeans++ fit at M=32 should not collapse — utilization
    should be at or near 1.0.
    """
    packets = _make_synthetic_clients(
        n_clients=80, n_per_client=128, K_local=32, D=8, seed=7
    )
    out = merge_local_codebooks(packets, M_global=32, seed=7)
    assert out["utilization"] >= 0.95, (
        f"Stage-2 utilization too low: {out['utilization']} "
        f"(n_empty={out['n_empty_clusters']}/32)"
    )
    assert out["k_min"] >= 1, (
        f"Stage-2 produced an empty cluster after fit: k_min={out['k_min']}"
    )
    # Codebook shape contract.
    assert out["codebook"].shape == (32, 8)
    assert out["codebook"].dtype == np.float32


# ----------------------------------------------------------------------------
# (c) Tiny-client fallback in _local_kmeans
# ----------------------------------------------------------------------------


def test_local_kmeans_tiny_client_fallback():
    """``_local_kmeans`` must not raise when ``N_i < K_local``.

    sklearn KMeans raises ``ValueError: n_samples < n_clusters`` directly,
    so the helper is responsible for clamping ``K_local_i = N_i``. UMass
    apts almost certainly never trigger this at K_local ≤ 8 (≈270 windows
    per apt), but the contract still has to hold for robustness.
    """
    rng = np.random.RandomState(123)
    H_tiny = rng.randn(3, 8).astype(np.float32)
    centroids, counts, inertia, K_local_i = _local_kmeans(
        H_tiny, K_local=8, seed=123
    )
    # Fallback fired:
    assert K_local_i == 3
    assert centroids.shape == (3, 8)
    assert counts.shape == (3,)
    assert int(counts.sum()) == 3
    # And the helper still returns a finite inertia.
    assert np.isfinite(inertia)


# ----------------------------------------------------------------------------
# Bonus sanity: federated_residual_offsets shape + zero-empty-cluster contract.
# ----------------------------------------------------------------------------


def test_federated_residual_offsets_shapes_and_empty_zero():
    """Offsets shape is (M, H); empty Stage-2 clusters get zero offset.

    We force an empty cluster by giving a tiny pool to the Stage-2 fit and
    checking that any cluster receiving zero h_g-routed mass on the
    aggregate has a zero row.
    """
    packets = _make_synthetic_clients(
        n_clients=4, n_per_client=32, K_local=2, D=8, seed=11
    )
    merge = merge_local_codebooks(packets, M_global=8, seed=11)
    offsets = federated_residual_offsets(packets, merge["codebook"])
    assert offsets.shape == (8, 24)
    assert offsets.dtype == np.float32
    # Build the cold-side check: which clusters received any window mass?
    M = 8
    sum_count = np.zeros(M, dtype=np.int64)
    for p in packets:
        if int(p["K_local_i"]) == 0:
            continue
        # Reuse the same routing as the helper.
        d = (
            (p["h_g"] ** 2).sum(axis=1, keepdims=True)
            - 2.0 * p["h_g"] @ merge["codebook"].T
            + (merge["codebook"] ** 2).sum(axis=1)
        )
        idx = d.argmin(axis=1)
        sum_count += np.bincount(idx, minlength=M)
    for c in range(M):
        if sum_count[c] == 0:
            np.testing.assert_array_equal(
                offsets[c], np.zeros(24, dtype=np.float32),
                err_msg=f"empty cluster {c} should have zero offset",
            )
