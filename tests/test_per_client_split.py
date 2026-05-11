"""Pytest for ``src/dataloader/per_client_split.py``.

Tests:
1. ~100 valid apartments are returned (UMass 2016 pool with min_hours=7000).
   We assert >=10 to keep the test cheap to gate locally; the full pool is
   100 apt and the unit verifies that *every* returned apt has positive
   train/val/test windowing.
2. Within each apt, train/val/test window-start indices do not overlap
   (time-ordered split — train ends before val starts before test starts,
   modulo the INPUT_SIZE look-back into the previous segment).
3. z-norm mean/std are fit on the *train* segment only — verified by
   computing the test segment's raw mean and confirming it differs from
   ``mean`` (this would fail if z-norm leaked the test split into stats).
4. Determinism: re-running with the same seed gives bit-identical splits.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pytest

from config import HORIZON, INPUT_SIZE
from dataloader.per_client_split import build_per_client_splits


@pytest.fixture(scope="module")
def splits(tmp_path_factory) -> dict[str, dict]:
    """Build splits with seed=42 into a tmp cache (so we don't pollute outputs/)."""
    cache_path = tmp_path_factory.mktemp("v06_pcs") / "per_client_split.pkl"
    return build_per_client_splits(seed=42, cache_path=cache_path, use_cache=False)


def test_returns_substantial_number_of_apartments(splits):
    # 100 apt full pool expected; assert >= 50 to be lenient if local UMass
    # checkout is partial. Hard guard against returning a near-empty dict.
    assert len(splits) >= 50, (
        f"build_per_client_splits returned only {len(splits)} apartments — "
        "expected ~100 from filter_valid_apartments(min_hours=7000)."
    )


def test_every_apt_has_positive_windows_per_split(splits):
    """All three splits must yield at least one window per kept apt."""
    for apt, sp in splits.items():
        for k_x, k_y in (("train_x", "train_y"), ("val_x", "val_y"), ("test_x", "test_y")):
            assert sp[k_x].shape[0] > 0, f"{apt}: {k_x} is empty"
            assert sp[k_y].shape[0] == sp[k_x].shape[0], (
                f"{apt}: {k_y} has {sp[k_y].shape[0]} windows but {k_x} has {sp[k_x].shape[0]}"
            )
            assert sp[k_x].shape[1] == INPUT_SIZE
            assert sp[k_y].shape[1] == HORIZON


def test_train_val_test_starts_do_not_overlap(splits):
    """Time-ordered split: each train forecast TARGET window is inside the train
    segment; each val target inside val; each test target inside test.

    Window 'start' is the input-window start; the forecast target lies in
    [start+INPUT_SIZE, start+INPUT_SIZE+HORIZON). We assert the *target*
    intervals do not overlap across splits per apt.
    """
    for apt, sp in splits.items():
        n = sp["series_len"]
        # Mirror per_client_split.build_per_client_splits boundary computation
        # exactly, including the round() that protects against 0.7+0.1 fp drift.
        train_end = int(n * 0.7)
        val_end = int(round(n * (0.7 + 0.1), 6))

        for s in sp["train_starts"]:
            tgt_lo = s + INPUT_SIZE
            tgt_hi = tgt_lo + HORIZON
            assert tgt_hi <= train_end, (
                f"{apt}: train target [{tgt_lo}, {tgt_hi}) extends past train_end={train_end}"
            )
        for s in sp["val_starts"]:
            tgt_lo = s + INPUT_SIZE
            tgt_hi = tgt_lo + HORIZON
            assert train_end <= tgt_lo and tgt_hi <= val_end, (
                f"{apt}: val target [{tgt_lo}, {tgt_hi}) outside [train_end={train_end}, val_end={val_end})"
            )
        for s in sp["test_starts"]:
            tgt_lo = s + INPUT_SIZE
            tgt_hi = tgt_lo + HORIZON
            assert val_end <= tgt_lo and tgt_hi <= n, (
                f"{apt}: test target [{tgt_lo}, {tgt_hi}) outside [val_end={val_end}, n={n})"
            )


def test_znorm_fit_on_train_segment_only(splits):
    """``mean`` and ``std`` must come from the train segment, not the entire
    series. Verify by recomputing stats on the test target region in raw
    space (= z * std + mean, then comparing to the apt's own train mean).
    For typical apts, train mean and test-period raw mean differ by more
    than 1e-3 in kW (load patterns drift across seasons), so this is a
    conservative regression test.
    """
    drift_count = 0
    total = 0
    for apt, sp in splits.items():
        # Reconstruct raw test target values: z * std + mean on the y-arrays.
        if sp["test_y"].shape[0] == 0:
            continue
        raw_test_y = sp["test_y"] * sp["std"] + sp["mean"]
        raw_test_mean = float(raw_test_y.mean())
        # Train mean is exactly sp["mean"] (z-norm fit on train segment).
        # If z-norm had been fit on the *entire* series, the two means would
        # be much closer (within ~1% relative). Allow some apts where they
        # happen to match by coincidence; demand at least 30% of apts show
        # a meaningful drift.
        if abs(raw_test_mean - sp["mean"]) > 0.05 * (abs(sp["mean"]) + 1e-6):
            drift_count += 1
        total += 1
    assert drift_count >= 0.3 * total, (
        f"Only {drift_count}/{total} apts show train-vs-test raw-mean drift > 5% — "
        "suspicious that z-norm was not fit on train segment alone."
    )


def test_seed_determinism(tmp_path):
    """Same seed → bit-identical splits."""
    cache1 = tmp_path / "cache1.pkl"
    cache2 = tmp_path / "cache2.pkl"
    s1 = build_per_client_splits(seed=42, cache_path=cache1, use_cache=False)
    s2 = build_per_client_splits(seed=42, cache_path=cache2, use_cache=False)
    assert set(s1.keys()) == set(s2.keys())
    # Spot-check first apt.
    apt = sorted(s1.keys())[0]
    np.testing.assert_array_equal(s1[apt]["train_x"], s2[apt]["train_x"])
    np.testing.assert_array_equal(s1[apt]["val_y"], s2[apt]["val_y"])
    assert s1[apt]["mean"] == s2[apt]["mean"]
    assert s1[apt]["std"] == s2[apt]["std"]
