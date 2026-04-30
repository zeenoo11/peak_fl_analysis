"""Recompute MSE/RMSE for the 3 FM zero-shot baselines.

Why
---
FM baselines (Chronos-Bolt small, Chronos-T5 tiny, TimesFM) have no
saved state dict — they are zero-shot and the model is reloaded from
HuggingFace each call. The existing v04 03_fm_zero_shot.py forecasts
in raw-kW space (no z-norm), so adding MSE/RMSE is a simple addition
on top of the same forecast loop.

This script mirrors 03_fm_zero_shot.py's path (zero-shot inference on
the v02 cold pool, sliding windows stride=24, kW directly) but:
    - iterates over all (model, seed) cells in one process so each
      forecaster is loaded once and reused across seeds,
    - computes MSE / RMSE alongside PAPE / HR / MAE,
    - dumps a unified JSON for downstream merge into the main
      mse_recompute_summary.json.

Output:
    outputs/v05_fedcb_codebook/mse_recompute_fm_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from config import HORIZON, INPUT_SIZE, OUTPUT_DIR, TRAIN_RATIO
from dataloader.splits import load_v02_split
from dataloader.umass import load_apartment_hourly
from utils.metrics import compute_hr, compute_mae, compute_pape

V05_OUT = OUTPUT_DIR / "v05_fedcb_codebook"

FM_FACTORY = {
    "fm_chronos_bolt_small": ("ChronosForecaster", {"model_id": "amazon/chronos-bolt-small"}),
    "fm_chronos_t5_tiny":    ("ChronosForecaster", {"model_id": "amazon/chronos-t5-tiny"}),
    "fm_timesfm":            ("TimesFMForecaster", {}),
}

SEEDS = [42, 123, 7]


def _build_forecaster(model_key: str):
    cls_name, kwargs = FM_FACTORY[model_key]
    if cls_name == "ChronosForecaster":
        from fm import ChronosForecaster
        return ChronosForecaster(**kwargs)
    if cls_name == "TimesFMForecaster":
        from fm import TimesFMForecaster
        return TimesFMForecaster(**kwargs)
    raise ValueError(f"unknown forecaster class: {cls_name}")


def _slide_windows(seg: np.ndarray, L: int = INPUT_SIZE, H: int = HORIZON, stride: int = HORIZON):
    total = L + H
    if len(seg) < total:
        return np.zeros((0, L), dtype=np.float32), np.zeros((0, H), dtype=np.float32)
    starts = np.arange(0, len(seg) - total + 1, stride)
    X = np.stack([seg[s:s + L] for s in starts]).astype(np.float32)
    Y = np.stack([seg[s + L:s + L + H] for s in starts]).astype(np.float32)
    return X, Y


def _compute_all(Y_kw: np.ndarray, Yp_kw: np.ndarray):
    err = Y_kw - Yp_kw
    mse = float((err ** 2).mean())
    rmse = float(np.sqrt(mse))
    return {
        "pape": float(compute_pape(Y_kw, Yp_kw)),
        "hr@1": float(compute_hr(Y_kw, Yp_kw, tol=1)),
        "hr@2": float(compute_hr(Y_kw, Yp_kw, tol=2)),
        "mae": float(compute_mae(Y_kw, Yp_kw)),
        "mse": mse,
        "rmse": rmse,
        "n_windows": int(Y_kw.shape[0]),
    }


def run_seed(forecaster, cold_apts, batch_size: int):
    true_chunks, pred_chunks = [], []
    n_apts_seen = 0
    for apt in cold_apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            continue
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        seg = series[:train_end]
        X, Y = _slide_windows(seg)
        if len(X) == 0:
            continue
        n_apts_seen += 1
        Y_pred = []
        for i in range(0, len(X), batch_size):
            yp = forecaster.forecast(X[i:i + batch_size])
            Y_pred.append(yp)
        Y_pred = np.concatenate(Y_pred, axis=0).astype(np.float32)
        true_chunks.append(Y)
        pred_chunks.append(Y_pred)
    if not true_chunks:
        return None
    Y_kw = np.concatenate(true_chunks, axis=0)
    Yp_kw = np.concatenate(pred_chunks, axis=0)
    return _compute_all(Y_kw, Yp_kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=V05_OUT / "mse_recompute_fm_summary.json")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    cold_cache = {s: load_v02_split(s)["cold"] for s in SEEDS}

    results = {}
    for model_key in FM_FACTORY:
        print(f"=== {model_key} ===")
        t_load = time.time()
        try:
            forecaster = _build_forecaster(model_key)
        except Exception as e:
            print(f"  FAILED to build {model_key}: {e}")
            continue
        print(f"  forecaster loaded ({time.time() - t_load:.1f}s)")
        per_seed = {}
        for s in SEEDS:
            t0 = time.time()
            res = run_seed(forecaster, cold_cache[s], args.batch_size)
            if res is None:
                print(f"  seed {s}: SKIP (no cold windows)")
                continue
            print(f"  seed {s}: PAPE={res['pape']:6.3f}  MAE={res['mae']:.4f}  "
                  f"MSE={res['mse']:.4f}  RMSE={res['rmse']:.4f}  ({time.time() - t0:.1f}s)")
            per_seed[str(s)] = res
        if per_seed:
            agg = {}
            for k in ("pape", "hr@1", "hr@2", "mae", "mse", "rmse"):
                vals = [per_seed[str(s)][k] for s in SEEDS if str(s) in per_seed]
                agg[k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "values": vals,
                }
            results[model_key] = {"per_seed": per_seed, "agg": agg, "n_seeds": len(per_seed)}
        # Free the forecaster's GPU memory before loading the next.
        del forecaster
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"methods": results, "seeds": SEEDS, "stride": HORIZON}, fh, indent=2)
    print(f"\n[done] saved -> {args.out}")


if __name__ == "__main__":
    main()
