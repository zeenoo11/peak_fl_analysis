"""Per-client 70/10/20 internal sliding-window split for v06 round dynamics.

(한글 요약)
v06 (`plans/v06-01_round_dynamics.md` §"클라이언트 모집단") 전용 dataloader.
v01-v05 의 cold-split 구조 (train apts vs cold apts) 를 *사용하지 않는다* —
모든 100 가구가 federated 학습에 참여하고, 각 가구 내부에서 시간 순으로
train(70%) / val(10%) / test(20%) 윈도우를 자른다.

Public surface
--------------
- ``build_per_client_splits(seed, year, train_ratio, val_ratio, cache_path)``
    - returns ``dict[apt_name -> dict]`` with windowed splits.
    - per-apt z-norm computed on the train segment only (CLAUDE.md unchanged).
    - sliding window ``INPUT_SIZE=96, HORIZON=24, stride=24`` for *all three
      splits* — non-overlapping windows, matches v04/v05 cold-eval stride
      convention. train stride=24 (not 1) keeps the train pool size in line
      with the val/test pool (same window discretisation across splits) and
      drops train pool to ~199 windows/apt = a single B=512 mini-batch per
      apt per epoch (matches plan §3 wall-clock discussion).
- The cache (``per_client_split.pkl``) stores the *windowed* numpy arrays
  (not the raw series) under ``outputs/v06_round_dynamics/seed{S}/``;
  re-running with the same seed loads from disk.

Determinism
-----------
``filter_valid_apartments(min_hours=7000)`` returns the 100-apt UMass 2016
pool deterministically (sorted Apt names). Within an apt, slicing is purely
positional → bit-equivalent across runs for a fixed seed. The ``seed``
argument currently only affects the cache file path (no random shuffle),
but is kept in the signature so future random extensions (e.g. resampled
windows) plug in without breaking the API.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from config import HORIZON, INPUT_SIZE, OUTPUT_DIR, TRAIN_RATIO, VAL_RATIO
from dataloader.umass import (
    filter_valid_apartments,
    list_available_apartments,
    load_apartment_hourly,
)


def build_per_client_splits(
    seed: int,
    year: str = "2016",
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    cache_path: Optional[Path] = None,
    use_cache: bool = True,
) -> dict[str, dict]:
    """Build the per-apartment 70/10/20 windowed split for v06.

    Parameters
    ----------
    seed         : seed (only affects cache filename in this module).
    year         : data year string (default '2016').
    train_ratio  : segment fraction for train (default ``TRAIN_RATIO=0.7``).
    val_ratio    : segment fraction for val   (default ``VAL_RATIO=0.1``).
                   test fraction = 1 - train_ratio - val_ratio (default 0.2).
    cache_path   : where to write/read the pickled split. Defaults to
                   ``OUTPUT_DIR / 'v06_round_dynamics' / f'seed{seed}' /
                   'per_client_split.pkl'``.
    use_cache    : if True, load existing cache when present.

    Returns
    -------
    dict mapping ``apt_name -> {
        'train_x', 'train_y',                   # float32 (Ntr, 96), (Ntr, 24)
        'val_x',   'val_y',                     # float32 (Nva, 96), (Nva, 24)
        'test_x',  'test_y',                    # float32 (Nte, 96), (Nte, 24)
        'mean', 'std',                          # float (z-norm fit on train segment)
        'train_idx_count', 'val_idx_count',     # ints (windows per split)
        'test_idx_count',
        'train_starts', 'val_starts', 'test_starts',  # window-start indices into raw series
        'series_len',                           # int (raw hourly series length)
    }``

    Apartments whose CSV is missing are silently skipped (matches
    ``fl/base.build_clients`` semantics). Apartments with too few hours to
    produce at least one window per split are also dropped.
    """
    if cache_path is None:
        cache_path = (
            OUTPUT_DIR / "v06_round_dynamics" / f"seed{seed}" / "per_client_split.pkl"
        )
    cache_path = Path(cache_path)
    if use_cache and cache_path.exists():
        with cache_path.open("rb") as fh:
            return pickle.load(fh)

    # 100 apt pool, deterministic & sorted (list_available_apartments + filter
    # are themselves deterministic — no seed needed).
    all_apts = list_available_apartments(year=year)
    apts = filter_valid_apartments(all_apts, year=year, min_hours=7000)

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError(
            f"build_per_client_splits: train_ratio({train_ratio})+val_ratio({val_ratio}) "
            f"must leave a positive test fraction; got {test_ratio}"
        )

    splits: dict[str, dict] = {}
    for apt in apts:
        try:
            series = load_apartment_hourly(apt, year=year).values.astype(np.float32)
        except FileNotFoundError:
            continue
        n = len(series)
        if n < INPUT_SIZE + HORIZON + 10:
            continue
        train_end = int(n * train_ratio)
        # Round-via-int after explicit add to avoid 0.7 + 0.1 = 0.79999... drift
        # versus the 0.8 written in tests/docs.
        val_end = int(round(n * (train_ratio + val_ratio), 6))

        seg_train = series[:train_end]
        # z-norm fit on TRAIN portion only (CLAUDE.md invariant).
        m_ = float(seg_train.mean())
        s_ = float(seg_train.std()) if seg_train.std() > 1e-8 else 1.0

        # Apply train-fit z-norm to the entire series; then carve splits.
        z = (series - m_) / s_

        # Windowing: stride = HORIZON (non-overlapping). Same stride for all
        # splits. We carve sub-series with INPUT_SIZE-window of context spilling
        # back into the previous split (train/val boundary): for val we need
        # the (INPUT_SIZE) hours before val_start as input context, otherwise
        # val/test windows would lose look-back. So:
        #   train windows : starts in [0, train_end - INPUT_SIZE - HORIZON]
        #   val windows   : starts in [train_end - INPUT_SIZE, val_end - INPUT_SIZE - HORIZON]
        #   test windows  : starts in [val_end - INPUT_SIZE, n - INPUT_SIZE - HORIZON]
        # Equivalent to the v01/v02 convention "warm-start z-norm + stride
        # boundary" where val/test segments include the look-back from the
        # immediately preceding portion. Train forecasts a target inside the
        # train segment only.

        def _carve(start_lo: int, end_hi: int, stride: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
            """Slide windows where the start index satisfies start in [start_lo, end_hi - INPUT_SIZE - HORIZON]."""
            seg_starts = list(
                range(max(0, start_lo), max(0, end_hi - INPUT_SIZE - HORIZON + 1), stride)
            )
            if not seg_starts:
                return (
                    np.zeros((0, INPUT_SIZE), dtype=np.float32),
                    np.zeros((0, HORIZON), dtype=np.float32),
                    [],
                )
            x = np.stack([z[s : s + INPUT_SIZE] for s in seg_starts]).astype(np.float32)
            y = np.stack(
                [z[s + INPUT_SIZE : s + INPUT_SIZE + HORIZON] for s in seg_starts]
            ).astype(np.float32)
            return x, y, seg_starts

        tr_x, tr_y, tr_starts = _carve(0, train_end, HORIZON)
        # Val/test keep the look-back into the previous segment.
        va_x, va_y, va_starts = _carve(train_end - INPUT_SIZE, val_end, HORIZON)
        te_x, te_y, te_starts = _carve(val_end - INPUT_SIZE, n, HORIZON)

        if len(tr_x) == 0 or len(va_x) == 0 or len(te_x) == 0:
            # Skip apts that can't yield at least one window in every split.
            continue

        splits[apt] = {
            "train_x": tr_x, "train_y": tr_y,
            "val_x":   va_x, "val_y":   va_y,
            "test_x":  te_x, "test_y":  te_y,
            "mean": m_, "std": s_,
            "train_idx_count": int(len(tr_starts)),
            "val_idx_count":   int(len(va_starts)),
            "test_idx_count":  int(len(te_starts)),
            "train_starts": tr_starts,
            "val_starts":   va_starts,
            "test_starts":  te_starts,
            "series_len": int(n),
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(splits, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return splits
