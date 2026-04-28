"""v04 FM zero-shot inference (no UMass training).

Per-seed × per-FM model invocation. Runs the FM on every cold apt's
sliding windows (warm-start z-norm = identity for FM since FM does its
own internal normalisation; we feed raw kW directly), denormalises if
needed, scores with PAPE / HR@1 / HR@2 / MAE.

The "seed" arg here only selects which 80:20 split's cold apts we
score on — the FM model itself is deterministic (zero-shot, no
training noise), so the 3-seed sweep just changes the apt list.

CLI:

    uv run python experiments/v04_full_baseline_comparison/03_fm_zero_shot.py \\
        --seed 42 --model chronos_bolt_small
    uv run python experiments/v04_full_baseline_comparison/03_fm_zero_shot.py \\
        --seed 42 --model chronos_t5_tiny
    uv run python experiments/v04_full_baseline_comparison/03_fm_zero_shot.py \\
        --seed 42 --model timesfm
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import HORIZON, INPUT_SIZE, OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO
from dataloader.splits import load_v02_split
from dataloader.umass import load_apartment_hourly
from utils.metrics import compute_hr, compute_mae, compute_pape

V04_OUT_ROOT = OUTPUT_DIR / "v04_full_baseline_comparison"

FM_FACTORY = {
    "chronos_bolt_small": ("ChronosForecaster", {"model_id": "amazon/chronos-bolt-small"}),
    "chronos_t5_tiny":    ("ChronosForecaster", {"model_id": "amazon/chronos-t5-tiny"}),
    "timesfm":            ("TimesFMForecaster", {}),
}


def _gpu_snapshot() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        used, free, total, util = (int(s.strip()) for s in out.strip().split(","))
        return {"used_MiB": used, "free_MiB": free, "total_MiB": total, "util_pct": util}
    except Exception:
        return {"cpu_only": not torch.cuda.is_available()}


def _build_forecaster(model_key: str):
    cls_name, kwargs = FM_FACTORY[model_key]
    if cls_name == "ChronosForecaster":
        from fm import ChronosForecaster
        return ChronosForecaster(**kwargs)
    elif cls_name == "TimesFMForecaster":
        from fm import TimesFMForecaster
        return TimesFMForecaster(**kwargs)
    else:
        raise ValueError(f"unknown forecaster class: {cls_name}")


def _slide_windows(seg: np.ndarray, L: int = INPUT_SIZE, H: int = HORIZON, stride: int = HORIZON):
    """Build (X, Y) sliding windows over a 1-D series, raw kW (no z-norm)."""
    total = L + H
    if len(seg) < total:
        return np.zeros((0, L), dtype=np.float32), np.zeros((0, H), dtype=np.float32)
    starts = np.arange(0, len(seg) - total + 1, stride)
    X = np.stack([seg[s:s+L] for s in starts]).astype(np.float32)
    Y = np.stack([seg[s+L:s+L+H] for s in starts]).astype(np.float32)
    return X, Y


def main() -> None:
    ap = argparse.ArgumentParser(description="v04 FM zero-shot cold inference (no UMass training).")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--model", required=True, choices=list(FM_FACTORY.keys()))
    ap.add_argument("--batch_size", type=int, default=64,
                    help="Per-call batch for the FM .forecast() — 64 is safe under 16 GB.")
    args = ap.parse_args()

    sp = load_v02_split(args.seed)
    cold_apts = sp["cold"]
    out_dir = V04_OUT_ROOT / f"seed{args.seed}" / f"fm_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v04 FM] seed={args.seed}  model={args.model}  cold={len(cold_apts)} apts")
    gpu_start = _gpu_snapshot()
    print(f"[v04 FM] GPU @start: {gpu_start}")

    forecaster = _build_forecaster(args.model)
    print(f"[v04 FM] forecaster ready")
    gpu_after_load = _gpu_snapshot()
    print(f"[v04 FM] GPU after load: {gpu_after_load}")

    t0 = time.time()
    true_chunks, pred_chunks = [], []
    n_apts_seen = 0
    for apt in cold_apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            print(f"  [skip] {apt}: missing")
            continue
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        seg = series[:train_end]
        X, Y = _slide_windows(seg)
        if len(X) == 0:
            continue
        n_apts_seen += 1
        # Run FM in batches to stay within VRAM bounds.
        Y_pred = []
        for i in range(0, len(X), args.batch_size):
            yp = forecaster.forecast(X[i:i+args.batch_size])
            Y_pred.append(yp)
        Y_pred = np.concatenate(Y_pred, axis=0).astype(np.float32)
        true_chunks.append(Y)
        pred_chunks.append(Y_pred)

    elapsed = time.time() - t0
    gpu_end = _gpu_snapshot()
    print(f"[v04 FM] GPU @end: {gpu_end}")

    if not true_chunks:
        print("[v04 FM] no cold windows produced")
        cold_metrics = {"pape": float("nan"), "hr@1": float("nan"), "hr@2": float("nan"),
                        "mae": float("nan"), "n_cold_windows": 0, "n_cold_apts": 0}
    else:
        Y = np.concatenate(true_chunks, axis=0)
        Yp = np.concatenate(pred_chunks, axis=0)
        cold_metrics = {
            "pape": float(compute_pape(Y, Yp)),
            "hr@1": float(compute_hr(Y, Yp, tol=1)),
            "hr@2": float(compute_hr(Y, Yp, tol=2)),
            "mae": float(compute_mae(Y, Yp)),
            "n_cold_windows": int(Y.shape[0]),
            "n_cold_apts": n_apts_seen,
        }

    print(f"[v04 FM] cold: PAPE={cold_metrics['pape']:.2f}  HR@1={cold_metrics['hr@1']:.1f}  "
          f"HR@2={cold_metrics['hr@2']:.1f}  ({cold_metrics['n_cold_windows']} windows)")
    print(f"[v04 FM] elapsed: {elapsed:.0f}s")

    with open(out_dir / "result.json", "w") as fh:
        json.dump({
            "algorithm": f"fm_{args.model}",
            "model": args.model,
            "model_factory": FM_FACTORY[args.model],
            "seed": int(args.seed),
            "cold_metrics": cold_metrics,
            "elapsed_seconds": elapsed,
            "gpu_at_start": gpu_start,
            "gpu_after_load": gpu_after_load,
            "gpu_at_end": gpu_end,
        }, fh, indent=2)
    print(f"[v04 FM] saved -> {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
