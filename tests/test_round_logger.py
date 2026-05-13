"""Pytest for ``src/fl/round_logger.py``.

Tests:
1. 2-round dummy trace: log_round writes 2 jsonl lines, each parses, has
   the right top-level keys, and the second row's ``upload_bytes_cum`` is
   the running sum.
2. across-apt mean / std: with hand-crafted predictions yielding known
   per-apt PAPE values, the across-apt mean equals numpy mean and the
   ``pape_std_across_clients`` equals numpy std(ddof=1).
3. drift_l2: when ``client_states is None`` the drift is exactly 0;
   when client_states are 1-step-distant from server_state_pre, the L2
   matches the analytical value.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from fl.round_logger import RoundLogger, _drift_l2, _per_client_eval


class _ConstPredModel(nn.Module):
    """Predicts a constant horizon (z-space) regardless of input — useful for
    forcing known per-apt PAPE values in the metric tests.
    """

    def __init__(self, c: float = 0.0, horizon: int = 24) -> None:
        super().__init__()
        self.register_buffer("c", torch.tensor(float(c)))
        self.horizon = horizon
        # An unused parameter so .parameters() is non-empty (some downstream
        # asserts expect at least one param).
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        y_hat = self.c.expand(b, self.horizon).clone()
        # Mimic NBEATSx 2-tuple return (y_hat, hiddens).
        return y_hat, {}


def _make_split_block(mean: float, std: float, x: np.ndarray, y: np.ndarray) -> dict:
    return {
        "mean": float(mean), "std": float(std),
        "train_x": x[:0], "train_y": y[:0],
        "val_x": x, "val_y": y,
        "test_x": x[:0], "test_y": y[:0],
        "train_idx_count": 0, "val_idx_count": int(len(x)), "test_idx_count": 0,
        "train_starts": [], "val_starts": list(range(len(x))), "test_starts": [],
        "series_len": int(len(x) + 96 + 24),
    }


def test_log_round_writes_jsonl_with_running_cumsum(tmp_path):
    """2-round dummy trace: jsonl parses, comm cumsum increments."""
    apt = "AptFake"
    x = np.zeros((3, 96), dtype=np.float32)
    y = np.zeros((3, 24), dtype=np.float32)
    splits = {apt: _make_split_block(0.0, 1.0, x, y)}
    val_data = {apt: {"x": x, "y": y}}

    log_path = tmp_path / "round_log.jsonl"
    model = _ConstPredModel(c=0.0)
    with RoundLogger(log_path, splits=splits, val_data=val_data) as logger:
        logger.log_round(
            round_idx=1, model=model,
            comm_stats={"upload_bytes_round": 100, "broadcast_bytes_round": 50},
            wall_seconds=1.0,
        )
        logger.log_round(
            round_idx=2, model=model,
            comm_stats={"upload_bytes_round": 200, "broadcast_bytes_round": 50},
            wall_seconds=1.0,
        )

    rows = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
    assert len(rows) == 2
    for r in rows:
        for k in ("round", "epoch_equivalent", "val", "train", "comm", "drift_l2", "wall_seconds_round"):
            assert k in r
        for k in ("pape_mean", "hr@1_mean", "n_clients", "n_windows_total"):
            assert k in r["val"]
    assert rows[0]["comm"]["upload_bytes_cum"] == 100
    assert rows[1]["comm"]["upload_bytes_cum"] == 300
    assert rows[0]["comm"]["broadcast_bytes_cum"] == 50
    assert rows[1]["comm"]["broadcast_bytes_cum"] == 100


def test_across_apt_mean_std_matches_numpy(tmp_path):
    """Hand-craft 3 apts so per-apt PAPE is exactly 0/100/200 % and the
    aggregator reports mean = 100, std (ddof=1) = 100.
    """
    # Apt 1: ground truth peak = 1.0, prediction = 1.0 → PAPE = 0
    # Apt 2: ground truth peak = 1.0, prediction = 0.0 → PAPE = 100
    # Apt 3: ground truth peak = 1.0, prediction = 3.0 → PAPE = 200
    horizon = 24
    x = np.zeros((1, 96), dtype=np.float32)
    y_peak = np.zeros((1, horizon), dtype=np.float32)
    y_peak[0, 0] = 1.0  # peak amplitude = 1.0 in z-space ; with mean=0,std=1 → 1.0 in kW too

    # Three apts each with their own per-apt model. We use a single shared
    # model and three different spits' (mean, std) to indirectly drive PAPE.
    # Easier: build a constant-pred model that returns 1.0 (z-space) and
    # control kW peak via per-apt (mean, std).
    model = _ConstPredModel(c=1.0)
    splits = {
        "AptA": _make_split_block(0.0, 1.0, x, y_peak),  # pred peak=1, true peak=1 → PAPE=0
        "AptB": _make_split_block(0.0, 0.0,  # std=0 → falls back to 1.0 in HouseholdDataset, but here we keep std=0 raw and just use it
                                  x, y_peak * 2.0),
        # Easier: just prepare three apts with explicit y_peak values to drive 0/100/200 directly.
    }
    # Re-do cleanly with three apts whose y_peak gives 0, 100, 200 PAPE under c=1.
    splits = {
        "AptA": _make_split_block(0.0, 1.0, x, y_peak.copy()),                  # true peak 1, pred 1 → 0
        "AptB": _make_split_block(0.0, 1.0, x, np.zeros_like(y_peak) + 1e-3),   # true peak ~ 0 → filtered (valid = |peak|>1e-5 fails) — must avoid
    }
    # Replace with: peak=1 (true) vs pred 1 → 0; peak=2 (true) vs pred 1 → 50; peak=0.5 vs pred 1 → 100
    yA = np.zeros((1, horizon), dtype=np.float32); yA[0, 0] = 1.0
    yB = np.zeros((1, horizon), dtype=np.float32); yB[0, 0] = 2.0
    yC = np.zeros((1, horizon), dtype=np.float32); yC[0, 0] = 0.5
    splits = {
        "AptA": _make_split_block(0.0, 1.0, x, yA),
        "AptB": _make_split_block(0.0, 1.0, x, yB),
        "AptC": _make_split_block(0.0, 1.0, x, yC),
    }
    val_data = {a: {"x": splits[a]["val_x"], "y": splits[a]["val_y"]} for a in splits}

    log_path = tmp_path / "round_log.jsonl"
    with RoundLogger(log_path, splits=splits, val_data=val_data) as logger:
        row = logger.log_round(round_idx=1, model=model)

    # PAPE per apt = |pred_peak - true_peak| / true_peak * 100
    #   AptA: |1 - 1| / 1 * 100 = 0
    #   AptB: |1 - 2| / 2 * 100 = 50
    #   AptC: |1 - 0.5| / 0.5 * 100 = 100
    expected_pape = [0.0, 50.0, 100.0]
    np.testing.assert_allclose(row["val"]["pape_mean"], np.mean(expected_pape), atol=1e-6)
    np.testing.assert_allclose(
        row["val"]["pape_std_across_clients"],
        np.std(expected_pape, ddof=1),
        atol=1e-6,
    )
    assert row["val"]["n_clients"] == 3


def test_log_round_includes_test_block_when_test_data_attached(tmp_path):
    """1A (round-by-round test trajectory): when test_data is attached, every
    round row carries a `test` block alongside `val` with the same metric keys."""
    apt = "AptFake"
    x = np.zeros((3, 96), dtype=np.float32)
    y = np.zeros((3, 24), dtype=np.float32)
    splits = {apt: _make_split_block(0.0, 1.0, x, y)}
    val_data  = {apt: {"x": x, "y": y}}
    test_data = {apt: {"x": x, "y": y}}

    log_path = tmp_path / "round_log.jsonl"
    model = _ConstPredModel(c=0.0)
    with RoundLogger(log_path, splits=splits,
                     val_data=val_data, test_data=test_data) as logger:
        logger.log_round(round_idx=1, model=model)

    rows = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
    assert len(rows) == 1
    r = rows[0]
    assert "test" in r, "test block must be present when test_data is attached"
    for k in ("pape_mean", "pape_std_across_clients",
              "hr@1_mean", "hr@2_mean", "mae_mean", "mse_kw2_mean",
              "n_clients", "n_windows_total"):
        assert k in r["test"], f"missing test.{k}"


def test_log_round_omits_test_block_when_test_data_none(tmp_path):
    """1A back-compat: when test_data=None, no `test` key is written so legacy
    callers / older jsonl rows remain interpretable.
    """
    apt = "AptFake"
    x = np.zeros((3, 96), dtype=np.float32)
    y = np.zeros((3, 24), dtype=np.float32)
    splits = {apt: _make_split_block(0.0, 1.0, x, y)}
    val_data = {apt: {"x": x, "y": y}}

    log_path = tmp_path / "round_log.jsonl"
    model = _ConstPredModel(c=0.0)
    with RoundLogger(log_path, splits=splits,
                     val_data=val_data, test_data=None) as logger:
        logger.log_round(round_idx=1, model=model)

    rows = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
    assert len(rows) == 1
    assert "test" not in rows[0]


def test_log_terminal_writes_single_row_with_val_and_test(tmp_path):
    """log_terminal must append exactly one row (round = -1) carrying both val
    and test blocks when test_data is attached (plan §3 single-row spec).
    """
    apt = "AptFake"
    x = np.zeros((3, 96), dtype=np.float32)
    y = np.zeros((3, 24), dtype=np.float32)
    splits = {apt: _make_split_block(0.0, 1.0, x, y)}
    val_data  = {apt: {"x": x, "y": y}}
    test_data = {apt: {"x": x, "y": y}}

    log_path = tmp_path / "round_log.jsonl"
    model = _ConstPredModel(c=0.0)
    with RoundLogger(log_path, splits=splits,
                     val_data=val_data, test_data=test_data) as logger:
        logger.log_terminal(
            model=model,
            comm_total={"upload_bytes_round": 4096, "broadcast_bytes_round": 2048},
            wall_total=12.5,
        )

    rows = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
    assert len(rows) == 1, "log_terminal must write exactly one row"
    r = rows[0]
    assert r["round"] == -1
    assert r["epoch_equivalent"] == -1.0
    assert "val" in r and "test" in r, "both val and test blocks required"
    for k in ("pape_mean", "hr@1_mean", "n_clients", "n_windows_total"):
        assert k in r["val"]
        assert k in r["test"]
    # comm_total: *_round and *_cum populated to the same total.
    assert r["comm"]["upload_bytes_round"]    == 4096
    assert r["comm"]["upload_bytes_cum"]      == 4096
    assert r["comm"]["broadcast_bytes_round"] == 2048
    assert r["comm"]["broadcast_bytes_cum"]   == 2048
    assert r["wall_seconds_round"] == 12.5


def test_log_terminal_omits_test_block_when_test_data_none(tmp_path):
    """log_terminal back-compat: when test_data=None, the row contains only
    the val block (no `test` key) — same convention as log_round.
    """
    apt = "AptFake"
    x = np.zeros((3, 96), dtype=np.float32)
    y = np.zeros((3, 24), dtype=np.float32)
    splits = {apt: _make_split_block(0.0, 1.0, x, y)}
    val_data = {apt: {"x": x, "y": y}}

    log_path = tmp_path / "round_log.jsonl"
    model = _ConstPredModel(c=0.0)
    with RoundLogger(log_path, splits=splits,
                     val_data=val_data, test_data=None) as logger:
        logger.log_terminal(model=model)

    rows = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
    assert len(rows) == 1
    r = rows[0]
    assert r["round"] == -1
    assert "val" in r
    assert "test" not in r


def test_drift_l2_zero_when_client_states_none():
    server_state = OrderedDict([("w", torch.randn(4))])
    assert _drift_l2(server_state, None) == 0.0


def test_drift_l2_matches_analytical():
    """Two clients each 1 step away in opposite L2 directions of magnitude 2 and 3
    → mean drift = (2 + 3) / 2 = 2.5.
    """
    server_state = OrderedDict([("w", torch.tensor([0.0, 0.0]))])
    client_a = OrderedDict([("w", torch.tensor([2.0, 0.0]))])  # ||·||₂ = 2
    client_b = OrderedDict([("w", torch.tensor([0.0, 3.0]))])  # ||·||₂ = 3
    drift = _drift_l2(server_state, [client_a, client_b])
    np.testing.assert_allclose(drift, 2.5, atol=1e-6)
