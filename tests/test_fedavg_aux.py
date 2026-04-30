"""Pytest for ``src/fl/fedavg_aux.py`` — extracted FedAvg-NBEATSxAux loop.

Smoke contract: 2 rounds × 2 local epochs × 2 synthetic clients on CPU.

Asserts:
    1. ``final_state_dict`` keys equal ``init_backbone_aux(seed=0).state_dict().keys()``
       — i.e. the federated state can be loaded with ``strict=True`` into a
       fresh NBEATSxAux. State dict contract preservation is load-bearing
       across v02 / v04 / v05 reuse.
    2. ``history["rounds"]`` has length == requested rounds, and each
       per-round entry has matching length (rounds, main_loss, aux_loss,
       n_clients).
    3. The history main losses are finite real numbers (not NaN/Inf) — the
       FedAvg+aux loss path runs without numerical collapse on a tiny
       synthetic dataset.

No real UMass data is touched. Two fabricated ``ClientData`` objects with
synthetic series of length 200 are passed via ``clients=`` so the loop
skips ``build_clients`` (and therefore the disk read).
"""

from __future__ import annotations

import sys
from pathlib import Path

# pyproject.toml sets pythonpath = ["src"], but be explicit so the test can
# also be invoked as ``uv run pytest tests/test_fedavg_aux.py`` from any cwd.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch

from config import INPUT_SIZE, HORIZON
from dataloader.umass import HouseholdDataset
from fl.base import ClientData
from fl.fedavg_aux import fedavg_aux_round_loop, init_backbone_aux


def _make_synthetic_client(apt: str, seed: int, length: int = 240) -> ClientData:
    """Fabricate a ClientData with a synthetic sin+noise series.

    length=240 gives ~120 stride-1 train windows after the (96+24) window
    requirement, which is plenty for a 2-epoch smoke run. The series is
    deterministic per ``seed`` so the test is reproducible.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(length, dtype=np.float32)
    series = (
        np.sin(2 * np.pi * t / 24.0)        # daily pattern
        + 0.3 * rng.randn(length).astype(np.float32)
    ).astype(np.float32) + 5.0              # offset away from zero
    train_end = int(length * 0.7)
    val_end = int(length * 0.8)
    seg_train = series[:train_end]
    seg_val = series[train_end:val_end]
    m_ = float(seg_train.mean())
    s_ = float(seg_train.std()) if seg_train.std() > 1e-8 else 1.0
    train_set = HouseholdDataset(seg_train, m_, s_, stride=1)
    val_set = HouseholdDataset(seg_val, m_, s_, stride=1)
    return ClientData(
        apt=apt,
        train_set=train_set,
        val_set=val_set,
        mean=m_,
        std=s_,
        n_train_windows=len(train_set),
    )


def test_fedavg_aux_round_loop_smoke():
    """End-to-end 2-round × 2-epoch × 2-client run; CPU-friendly (<60s)."""
    clients = [
        _make_synthetic_client("AptSyntheticA", seed=42),
        _make_synthetic_client("AptSyntheticB", seed=43),
    ]
    # Sanity: each synthetic client has a non-trivial number of train windows.
    for c in clients:
        assert c.n_train_windows >= 1, f"{c.apt}: n_train_windows = {c.n_train_windows}"

    rounds = 2
    local_epochs = 2
    out = fedavg_aux_round_loop(
        train_apts=[c.apt for c in clients],   # ignored when clients= passed
        clients=clients,
        rounds=rounds,
        local_epochs=local_epochs,
        lr=1e-3,
        batch_size=16,
        weight_decay=1e-5,
        seed=0,
        use_amp=False,                          # CPU-friendly: bf16 disabled
        aux_lambda=0.3,
        hr_weight=0.1,
    )

    # (1) state-dict key contract: federated state loads strict=True into a
    # fresh NBEATSxAux. This is the load-bearing invariant for v04 09_fix_rerun
    # checkpoint reuse downstream.
    fresh = init_backbone_aux(seed=0)
    assert set(out["final_state_dict"].keys()) == set(fresh.state_dict().keys()), (
        "fedavg_aux_round_loop final_state_dict keys diverged from a fresh "
        "init — strict=True load would fail downstream."
    )

    # (2) history shape contract.
    hist = out["history"]
    for k in ("rounds", "main_loss", "aux_loss", "n_clients"):
        assert k in hist, f"history missing key: {k}"
        assert len(hist[k]) == rounds, (
            f"history[{k!r}] length {len(hist[k])} != rounds {rounds}"
        )

    # (3) numerical sanity: losses are finite, n_clients tracks all.
    for k in ("main_loss", "aux_loss"):
        for v in hist[k]:
            assert np.isfinite(v), f"history[{k}] contains non-finite: {v}"
    assert all(n == len(clients) for n in hist["n_clients"])
    assert out["n_train_clients"] == len(clients)


def test_fedavg_aux_round_loop_strict_load():
    """The CPU-trained federated state must load strict=True into a fresh model.

    This is a separate test from the key-set check above to also verify
    dtype/shape compatibility (state_dict.keys() can match while a single
    tensor disagrees on shape).
    """
    clients = [
        _make_synthetic_client("AptSyntheticC", seed=7),
        _make_synthetic_client("AptSyntheticD", seed=8),
    ]
    out = fedavg_aux_round_loop(
        train_apts=[c.apt for c in clients],
        clients=clients,
        rounds=1, local_epochs=1, lr=1e-3, batch_size=16,
        weight_decay=1e-5, seed=0, use_amp=False,
    )
    fresh = init_backbone_aux(seed=0)
    fresh.load_state_dict(out["final_state_dict"], strict=True)  # raises on mismatch
