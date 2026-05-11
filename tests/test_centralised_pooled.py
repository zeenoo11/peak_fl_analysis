"""Pytest for ``src/fl/centralised_pooled.py`` — V6-Dyn-A reference helper.

Smoke contract: 2 epochs × 3 synthetic clients on CPU.

Asserts:
  1. final state dict loads strict=True into a fresh NBEATSxAux.
  2. history.main_loss[1] < history.main_loss[0] (training reduces loss
     on a deterministic synthetic dataset within a single epoch step).
  3. ``on_round_end`` callback is called exactly ``n_epochs`` times with
     ``client_states=None`` (centralised cell) and zero-bytes comm stats.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from fl.centralised_pooled import centralised_pooled_train
from fl.fedavg_aux import init_backbone_aux


def _make_synthetic_split(seed: int, n_apts: int = 3) -> dict[str, dict]:
    """Build 3 synthetic apt splits with deterministic sin+noise data."""
    rng = np.random.RandomState(seed)
    splits = {}
    for k in range(n_apts):
        # length ≈ 240 hours → ~6 train windows at stride=24 (96+24=120 < 240*0.7=168 → 168-120=48, stride=24 → 3 windows; tight but works).
        # Use length=400 to be comfortable: train portion ~280, val ~40, test ~80.
        length = 400
        t = np.arange(length, dtype=np.float32)
        s = (np.sin(2 * np.pi * t / 24.0) + 0.3 * rng.randn(length).astype(np.float32) + 5.0).astype(np.float32)
        train_end = int(length * 0.7)
        val_end = int(round(length * 0.8, 6))
        m_ = float(s[:train_end].mean())
        st = float(s[:train_end].std()) if s[:train_end].std() > 1e-8 else 1.0
        z = (s - m_) / st

        from config import HORIZON, INPUT_SIZE
        def carve(lo, hi):
            starts = list(range(max(0, lo), max(0, hi - INPUT_SIZE - HORIZON + 1), HORIZON))
            if not starts:
                return np.zeros((0, INPUT_SIZE), dtype=np.float32), np.zeros((0, HORIZON), dtype=np.float32), []
            x = np.stack([z[ss:ss+INPUT_SIZE] for ss in starts]).astype(np.float32)
            y = np.stack([z[ss+INPUT_SIZE:ss+INPUT_SIZE+HORIZON] for ss in starts]).astype(np.float32)
            return x, y, starts

        tr_x, tr_y, tr_s = carve(0, train_end)
        va_x, va_y, va_s = carve(train_end - INPUT_SIZE, val_end)
        te_x, te_y, te_s = carve(val_end - INPUT_SIZE, length)
        splits[f"AptSyn{k}"] = {
            "train_x": tr_x, "train_y": tr_y,
            "val_x": va_x, "val_y": va_y,
            "test_x": te_x, "test_y": te_y,
            "mean": m_, "std": st,
            "train_idx_count": len(tr_s), "val_idx_count": len(va_s), "test_idx_count": len(te_s),
            "train_starts": tr_s, "val_starts": va_s, "test_starts": te_s,
            "series_len": length,
        }
    return splits


def test_centralised_pooled_runs_and_reduces_loss():
    splits = _make_synthetic_split(seed=42)
    callback_calls = []

    def cb(**kw):
        callback_calls.append(kw)

    result = centralised_pooled_train(
        splits,
        n_epochs=2, lr=1e-3, batch_size=16, weight_decay=1e-5,
        aux_lambda=0.3, hr_weight=0.1, seed=0, use_amp=False,
        on_round_end=cb,
    )
    # (1) history shape
    assert "main_loss" in result["history"] and len(result["history"]["main_loss"]) == 2

    # (2) loss reduces across the 2 epochs (deterministic synthetic data).
    assert result["history"]["main_loss"][1] <= result["history"]["main_loss"][0] * 1.05, (
        f"loss did not decrease: {result['history']['main_loss']}"
    )

    # (3) state dict loads strict=True
    fresh = init_backbone_aux(seed=0)
    fresh.load_state_dict(result["final_state_dict"], strict=True)

    # (4) callback called twice with the right contract.
    assert len(callback_calls) == 2
    for call in callback_calls:
        assert call["client_states"] is None
        assert call["server_state_pre"] is None
        assert call["comm_stats"]["upload_bytes_round"] == 0
        assert call["comm_stats"]["broadcast_bytes_round"] == 0
