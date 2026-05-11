"""Pytest for v06 Phase 2 codebook stacking add-on.

Covers:
    a. ``local_codebook_step_from_splits`` shape contracts (h_g, centroids,
       counts, K_local_i) on a small synthetic 3-client pool.
    b. ``merge_local_codebooks`` returns a ``(M, D)`` codebook with non-zero
       utilization on the same packets.
    c. ``federated_residual_offsets`` returns ``(M, H)`` and the per-cluster
       average matches an analytic ground-truth on a hand-crafted toy.
    d. Centralised pooled-KMeans path produces a ``(M, D)`` codebook + ``(M, H)``
       offsets with identical shape contract.
    e. CMO correction ``y_hat_corr = y_hat + α · offset[c*]`` is element-wise
       exact (no broadcast / dtype quirks).

The tests use a dummy ``DummyAux`` torch module that returns
``(y_hat, hiddens, aux)`` tuples — same forward signature as
``NBEATSxAux`` — so we exercise the real ``_extract_h_g_from_windows``
without touching the NBEATSx backbone.
"""

from __future__ import annotations

import sys
from pathlib import Path

# pyproject.toml sets pythonpath = ["src"] but be explicit so the tests can
# also be run via ``uv run pytest`` from any cwd.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pytest
import torch
import torch.nn as nn

from fl.base import DEVICE
from fl.codebook_fl import (
    _extract_h_g_from_windows,
    _route_h_g_to_codebook,
    federated_residual_offsets,
    local_codebook_step_from_splits,
    merge_local_codebooks,
)


def _make_dummy_model() -> "DummyAux":
    """Build a DummyAux on ``DEVICE`` so the forward path matches the helper
    contract (helper always moves input to DEVICE)."""
    return DummyAux().to(DEVICE).eval()


# ----------------------------------------------------------------------------
# Dummy backbone — emits fake (y_hat, hiddens, aux) with the right shapes.
# ----------------------------------------------------------------------------


class DummyAux(nn.Module):
    """NBEATSxAux-shaped forward stub: (y_hat, {'h_generic': ...}, (amp, hr)).

    h_generic is a deterministic non-trivial mapping of the input window
    (mean / std / max-of-last-24 of x) projected up to D=64 so the
    KMeans++ has actual variance to cluster on. y_hat is set to
    ``mean(x_last_24)`` repeated over horizon — the residual tests don't
    depend on its exact shape, just that it is finite and consistent.
    """

    def __init__(self, d_h: int = 64, horizon: int = 24) -> None:
        super().__init__()
        self.d_h = d_h
        self.horizon = horizon
        # Deterministic projection seed so two calls give the same h_g.
        torch.manual_seed(0)
        self.proj = nn.Linear(3, d_h, bias=True)

    def forward(self, x: torch.Tensor):
        # x : (B, INPUT_SIZE)
        last24 = x[:, -24:]
        feats = torch.stack(
            [last24.mean(dim=1), last24.std(dim=1), last24.max(dim=1).values],
            dim=1,
        )
        h_g = self.proj(feats)                            # (B, d_h)
        y_hat = last24.mean(dim=1, keepdim=True).expand(-1, self.horizon).contiguous()
        amp = y_hat.max(dim=1, keepdim=True).values
        hr = torch.zeros((x.shape[0], self.horizon))
        return y_hat, {"h_generic": h_g}, (amp, hr)


# ----------------------------------------------------------------------------
# Synthetic split fixtures
# ----------------------------------------------------------------------------


def _make_synthetic_splits(
    *, n_clients: int, n_train: int, n_test: int, input_size: int = 96,
    horizon: int = 24, seed: int = 11,
) -> dict[str, dict]:
    """Build a small per-client split mimicking ``per_client_split.pkl``.

    Each client gets a distinct random offset so its windows are
    distinguishable in h_g space — different clients should land in
    different KMeans buckets (Stage 1 utilization > 0).
    """
    rng = np.random.RandomState(seed)
    splits: dict[str, dict] = {}
    for ci in range(n_clients):
        offset = float(ci * 5.0)
        train_x = rng.randn(n_train, input_size).astype(np.float32) + offset
        train_y = rng.randn(n_train, horizon).astype(np.float32) + offset
        test_x = rng.randn(n_test, input_size).astype(np.float32) + offset
        test_y = rng.randn(n_test, horizon).astype(np.float32) + offset
        splits[f"Apt{ci}"] = {
            "train_x": train_x, "train_y": train_y,
            "val_x": np.zeros((0, input_size), dtype=np.float32),
            "val_y": np.zeros((0, horizon), dtype=np.float32),
            "test_x": test_x, "test_y": test_y,
            "mean": float(offset), "std": 1.0,
        }
    return splits


# ----------------------------------------------------------------------------
# (a) local_codebook_step_from_splits shape contract
# ----------------------------------------------------------------------------


def test_local_codebook_step_from_splits_shapes():
    splits = _make_synthetic_splits(n_clients=3, n_train=20, n_test=8)
    model = _make_dummy_model()
    apt = "Apt0"
    pkt = local_codebook_step_from_splits(
        model, splits[apt]["train_x"], splits[apt]["train_y"],
        K_local=2, seed=42, batch_size=4, use_amp=False,
    )
    assert pkt["h_g"].shape == (20, 64)
    assert pkt["h_g"].dtype == np.float32
    assert pkt["y_hat_z"].shape == (20, 24)
    assert pkt["y_true_z"].shape == (20, 24)
    np.testing.assert_array_equal(pkt["y_true_z"], splits[apt]["train_y"])
    assert pkt["centroids"].shape == (2, 64)
    assert pkt["counts"].shape == (2,)
    assert pkt["K_local_i"] == 2
    assert int(pkt["counts"].sum()) == 20
    assert np.isfinite(pkt["inertia"])


def test_local_codebook_step_from_splits_empty_client():
    """Empty client (zero train windows) emits a zero packet, no exception."""
    model = _make_dummy_model()
    pkt = local_codebook_step_from_splits(
        model,
        np.zeros((0, 96), dtype=np.float32),
        np.zeros((0, 24), dtype=np.float32),
        K_local=2, seed=42, batch_size=4, use_amp=False,
    )
    assert pkt["K_local_i"] == 0
    assert pkt["h_g"].shape == (0, 64)
    assert pkt["centroids"].shape == (0, 64)
    assert int(pkt["counts"].sum()) == 0


# ----------------------------------------------------------------------------
# (b) merge_local_codebooks shape + utilization on Stage-1 packets
# ----------------------------------------------------------------------------


def test_merge_local_codebooks_from_splits_packets():
    splits = _make_synthetic_splits(n_clients=8, n_train=32, n_test=8, seed=7)
    model = _make_dummy_model()
    packets = []
    for apt in splits:
        pkt = local_codebook_step_from_splits(
            model, splits[apt]["train_x"], splits[apt]["train_y"],
            K_local=4, seed=7, batch_size=8, use_amp=False,
        )
        packets.append(pkt)
    merge = merge_local_codebooks(packets, M_global=8, seed=7)
    assert merge["codebook"].shape == (8, 64)
    assert merge["codebook"].dtype == np.float32
    # 8 well-separated clients × K_local=4 = 32 input centroids; M=8 should
    # at least keep some clusters non-empty.
    assert merge["utilization"] > 0.0


# ----------------------------------------------------------------------------
# (c) federated_residual_offsets — analytic correctness on toy
# ----------------------------------------------------------------------------


def test_federated_residual_offsets_analytic():
    """Hand-craft 2 packets routing to 2 clusters with known per-cluster
    residuals; verify the cluster-mean matches the analytic answer.

    Cluster 0: client A windows 0,1 with residual = +1 (over horizon=4).
    Cluster 1: client B windows 0,1 with residual = -1 (over horizon=4).
    Expected offsets[0] = +1, offsets[1] = -1 (constant over horizon).
    """
    H = 4
    D = 4
    M = 2
    # Codebook = two well-separated centroids on opposite axes.
    codebook = np.array([
        [10.0, 0.0, 0.0, 0.0],
        [-10.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    # Client A — h_g all near codebook[0].
    h_g_A = np.tile(codebook[0:1, :], (2, 1))
    y_hat_A = np.zeros((2, H), dtype=np.float32)
    y_true_A = np.ones((2, H), dtype=np.float32)
    pkt_A = {
        "centroids": h_g_A.copy(),
        "counts": np.array([2], dtype=np.int64),
        "h_g": h_g_A.astype(np.float32),
        "y_hat_z": y_hat_A,
        "y_true_z": y_true_A,
        "K_local_i": 1,
        "inertia": 0.0,
    }
    # Client B — h_g all near codebook[1].
    h_g_B = np.tile(codebook[1:2, :], (2, 1))
    y_hat_B = np.zeros((2, H), dtype=np.float32)
    y_true_B = -np.ones((2, H), dtype=np.float32)
    pkt_B = {
        "centroids": h_g_B.copy(),
        "counts": np.array([2], dtype=np.int64),
        "h_g": h_g_B.astype(np.float32),
        "y_hat_z": y_hat_B,
        "y_true_z": y_true_B,
        "K_local_i": 1,
        "inertia": 0.0,
    }
    offsets = federated_residual_offsets([pkt_A, pkt_B], codebook)
    assert offsets.shape == (M, H)
    assert offsets.dtype == np.float32
    np.testing.assert_allclose(offsets[0], np.ones(H, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(offsets[1], -np.ones(H, dtype=np.float32), atol=1e-6)


# ----------------------------------------------------------------------------
# (d) Centralised pooled KMeans path — driver-level helper.
# ----------------------------------------------------------------------------


def test_centralised_codebook_path_shapes():
    """The 08_codebook_stacking.py centralised path uses sklearn KMeans on
    the pooled h_g and the same residual aggregation. Replicate it here
    inline so we don't import the script (which would also import torch
    config side-effects)."""
    splits = _make_synthetic_splits(n_clients=4, n_train=16, n_test=4, seed=3)
    model = _make_dummy_model()
    packets = []
    for apt in splits:
        pkt = local_codebook_step_from_splits(
            model, splits[apt]["train_x"], splits[apt]["train_y"],
            K_local=2, seed=3, batch_size=8, use_amp=False,
        )
        packets.append(pkt)
    pooled = np.concatenate([p["h_g"] for p in packets], axis=0)
    M = 4
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=M, init="k-means++", n_init=10, random_state=3).fit(pooled)
    codebook = km.cluster_centers_.astype(np.float32)
    assert codebook.shape == (M, 64)
    # Pooled per-cluster residual offsets — same accumulation as the FL
    # variant, but applied to the pooled-h_g stream.
    offsets = federated_residual_offsets(packets, codebook)
    assert offsets.shape == (M, 24)
    assert offsets.dtype == np.float32


# ----------------------------------------------------------------------------
# (e) CMO correction is element-wise exact.
# ----------------------------------------------------------------------------


def test_cmo_correction_elementwise():
    """``y_hat_corr = y_hat + alpha * offset[c*]`` (no broadcast surprises)."""
    M = 3
    H = 24
    rng = np.random.RandomState(0)
    codebook = rng.randn(M, 16).astype(np.float32) * 5.0
    offsets = rng.randn(M, H).astype(np.float32)
    h_g_cold = np.stack([codebook[1], codebook[0], codebook[2], codebook[1]]).astype(np.float32)
    y_hat_base = rng.randn(4, H).astype(np.float32)

    c_idx = _route_h_g_to_codebook(h_g_cold, codebook)
    np.testing.assert_array_equal(c_idx, np.array([1, 0, 2, 1], dtype=np.int64))

    alpha = 1.0
    y_hat_corr = (y_hat_base + alpha * offsets[c_idx]).astype(np.float32)
    # Manual sanity-check on row 0 (cluster 1).
    np.testing.assert_allclose(
        y_hat_corr[0], y_hat_base[0] + offsets[1],
        atol=1e-7,
    )
    # And on row 3 (cluster 1 again — should also pick offsets[1]).
    np.testing.assert_allclose(
        y_hat_corr[3], y_hat_base[3] + offsets[1],
        atol=1e-7,
    )


# ----------------------------------------------------------------------------
# (f) _extract_h_g_from_windows row-count contract on multi-batch input
# ----------------------------------------------------------------------------


def test_extract_h_g_from_windows_handles_multi_batch():
    """If batch_size < N, the helper still returns N rows (no truncation)."""
    model = _make_dummy_model()
    n = 13
    train_x = np.random.RandomState(2).randn(n, 96).astype(np.float32)
    train_y = np.random.RandomState(3).randn(n, 24).astype(np.float32)
    h_g, y_hat, y_true = _extract_h_g_from_windows(
        model, train_x, train_y, batch_size=5, use_amp=False,
    )
    assert h_g.shape == (n, 64)
    assert y_hat.shape == (n, 24)
    assert y_true.shape == (n, 24)
    np.testing.assert_array_equal(y_true, train_y)


def test_extract_h_g_from_windows_row_mismatch_raises():
    """Asymmetric (N_x != N_y) input raises a clear ValueError."""
    model = _make_dummy_model()
    with pytest.raises(ValueError):
        _extract_h_g_from_windows(
            model,
            np.zeros((4, 96), dtype=np.float32),
            np.zeros((3, 24), dtype=np.float32),
            batch_size=2, use_amp=False,
        )
