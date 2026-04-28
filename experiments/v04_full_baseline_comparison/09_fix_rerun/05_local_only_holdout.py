"""Local-only NBEATSx with proper held-out evaluation (PR #1 fix follow-up).

Why
---
The parent v04 folder's Local-only baseline (``src/fl/local_only.py``)
trains on each cold apt's first-70% segment (``series[:train_end]``)
**and evaluates on the same segment**. PR #1 caught that the previous
docstring misleadingly claimed it evaluated on a held-out segment; the
docstring was corrected (commit ``4fdab45``) to honestly say "self-train
+ self-eval, overfit upper bound on its own data". The implementation
itself was kept aligned with the other v04 baselines (which evaluate on
the cold apt's first-70%, which is **unseen** for the FL/NF/FM
baselines).

This script gives the **fair generalisation** number Local-only needs to
take seriously as a competing baseline:

    Train on : cold_apt.series[:train_end]                 # first 70%
    Eval  on : cold_apt.series[train_end:val_end]          # next 20% (val portion of v01 7:1:2)

with ``val_end = int(n * (TRAIN_RATIO + VAL_RATIO))`` = ``int(n * 0.8)``.
Sliding stride=24 windows; z-norm uses train-segment statistics
(``warm-start`` — same convention every other cold-eval helper uses).

To preserve comparability with the parent folder's Local-only row, the
script reports BOTH:

- ``self_eval``    — the existing self-train + self-eval metric
                     on ``series[:train_end]`` (overfit upper bound,
                     should match the parent folder's local_only row).
- ``holdout_eval`` — the new fair-generalisation metric on
                     ``series[train_end:val_end]``.

Reuse
-----
``src/fl/local_only.py:_train_one_cold_apt`` already implements the
self-train step (and is the same one used by the parent folder); we
import it directly so the *training* step stays bit-identical. We add a
local ``_eval_one_cold_apt_holdout`` that slices the held-out segment.
The parent folder's ``_eval_one_cold_apt`` is also imported and reused
for the ``self_eval`` block.

CLI
---
    uv run python experiments/v04_full_baseline_comparison/09_fix_rerun/05_local_only_holdout.py --seed 42

Output
------
    outputs/v04_full_baseline_comparison/09_fix_rerun/seed{S}/local_only_holdout/
        result.json
            ├── algorithm: "local_only_holdout"
            ├── seed
            ├── self_eval:    {pape, hr@1, hr@2, mae, n_cold_windows, n_cold_apts}
            ├── holdout_eval: {pape, hr@1, hr@2, mae, n_cold_windows, n_cold_apts}
            ├── per_apt_metrics:
            │       [{apt, self: {...}, holdout: {...}, n_train_windows, n_holdout_windows}, ...]
            ├── n_train_windows (sum), n_holdout_windows (sum)
            ├── elapsed_seconds, gpu_at_start/end

Wall-clock per seed: ~15 min on a 5070 Ti
    (one fresh NBEATSx self-train per cold apt × 20 cold apts; same as the
     parent local_only cell — the held-out eval is two extra forward passes
     per apt, negligible.)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import HORIZON, OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO
from dataloader.splits import load_v02_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from fl.base import DEVICE
from fl.local_only import (
    LocalOnlyConfig,
    _eval_one_cold_apt,    # parent's self-eval helper (reused as-is)
    _train_one_cold_apt,   # parent's self-train helper (reused as-is)
)
from utils.metrics import compute_hr, compute_mae, compute_pape

V04_FIX_OUT_ROOT = OUTPUT_DIR / "v04_full_baseline_comparison" / "09_fix_rerun"


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


def _eval_one_cold_apt_holdout(
    model: torch.nn.Module,
    apt: str,
    mean: float,
    std: float,
    *,
    batch: int = 256,
    stride: int = HORIZON,
) -> dict:
    """Evaluate the per-apt self-trained model on the **held-out** segment.

    Held-out segment = ``series[train_end:val_end]`` where
    ``val_end = int(n * (TRAIN_RATIO + VAL_RATIO))``. This is the val
    portion of the v01 7:1:2 split. z-norm uses the train-segment
    statistics (``mean``, ``std`` passed in from ``_train_one_cold_apt``)
    — warm-start, same convention every cold-eval helper uses.
    """
    series = load_apartment_hourly(apt).values.astype(np.float32)
    n = len(series)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    seg = series[train_end:val_end]
    ds = HouseholdDataset(seg, mean, std, stride=stride)
    if len(ds) == 0:
        return {"pape": float("nan"), "hr@1": float("nan"), "hr@2": float("nan"),
                "mae": float("nan"), "n_windows": 0}
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    model.eval()
    true_z, pred_z = [], []
    with torch.no_grad():
        for x, y in loader:
            y_hat, _ = model(x.to(DEVICE))
            true_z.append(y.numpy())
            pred_z.append(y_hat.cpu().numpy())
    t_z = np.concatenate(true_z, axis=0)
    p_z = np.concatenate(pred_z, axis=0)
    t_kw = t_z * std + mean
    p_kw = p_z * std + mean
    return {
        "pape": float(compute_pape(t_kw, p_kw)),
        "hr@1": float(compute_hr(t_kw, p_kw, tol=1)),
        "hr@2": float(compute_hr(t_kw, p_kw, tol=2)),
        "mae": float(compute_mae(t_kw, p_kw)),
        "n_windows": int(t_z.shape[0]),
    }


def _wmean(pairs: list[tuple[float, int]]) -> float:
    if not pairs or sum(n for _, n in pairs) == 0:
        return float("nan")
    return float(sum(v * n for v, n in pairs) / sum(n for _, n in pairs))


def main() -> None:
    ap = argparse.ArgumentParser(description="v04 09_fix_rerun: Local-only with held-out evaluation.")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--rounds", type=int, default=20,
                    help="Total epochs of self-train per cold apt (LocalOnlyConfig.rounds reinterpretation).")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg = LocalOnlyConfig(
        rounds=args.rounds,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        seed=args.seed,
        use_amp=not args.no_amp,
    )

    sp = load_v02_split(args.seed)
    cold_apts = sp["cold"]   # train apts unused — Local-only definition
    out_dir = V04_FIX_OUT_ROOT / f"seed{args.seed}" / "local_only_holdout"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v04 LO-holdout] seed={args.seed}  rounds={args.rounds}  batch={args.batch_size}  amp={cfg.use_amp}")
    gpu_start = _gpu_snapshot()
    print(f"[v04 LO-holdout] GPU @start: {gpu_start}")

    t0 = time.time()
    per_apt: list[dict] = []
    pooled_self_pape, pooled_self_hr1, pooled_self_hr2, pooled_self_mae = [], [], [], []
    pooled_ho_pape, pooled_ho_hr1, pooled_ho_hr2, pooled_ho_mae = [], [], [], []
    n_train_total, n_ho_total = 0, 0

    for apt in cold_apts:
        out = _train_one_cold_apt(apt, cfg)
        if out is None:
            print(f"  [skip] {apt}: missing or empty train set")
            continue
        model, log = out

        self_m = _eval_one_cold_apt(model, apt, log["mean"], log["std"])
        ho_m = _eval_one_cold_apt_holdout(
            model, apt, log["mean"], log["std"],
            batch=args.batch_size, stride=HORIZON,
        )

        per_apt.append({
            "apt": apt,
            "n_train_windows": int(log["n_train_windows"]),
            "n_self_windows":  int(self_m["n_windows"]),
            "n_holdout_windows": int(ho_m["n_windows"]),
            "self":    {k: self_m[k] for k in ("pape", "hr@1", "hr@2", "mae", "n_windows")},
            "holdout": {k: ho_m[k]   for k in ("pape", "hr@1", "hr@2", "mae", "n_windows")},
        })
        pooled_self_pape.append((self_m["pape"], self_m["n_windows"]))
        pooled_self_hr1.append((self_m["hr@1"], self_m["n_windows"]))
        pooled_self_hr2.append((self_m["hr@2"], self_m["n_windows"]))
        pooled_self_mae.append((self_m["mae"],  self_m["n_windows"]))
        pooled_ho_pape.append((ho_m["pape"], ho_m["n_windows"]))
        pooled_ho_hr1.append((ho_m["hr@1"], ho_m["n_windows"]))
        pooled_ho_hr2.append((ho_m["hr@2"], ho_m["n_windows"]))
        pooled_ho_mae.append((ho_m["mae"],  ho_m["n_windows"]))
        n_train_total += int(log["n_train_windows"])
        n_ho_total += int(ho_m["n_windows"])
        print(f"  {apt}: self_PAPE={self_m['pape']:.2f}  holdout_PAPE={ho_m['pape']:.2f}  "
              f"(n_train={log['n_train_windows']}  n_ho={ho_m['n_windows']})")

    elapsed = time.time() - t0
    gpu_end = _gpu_snapshot()

    self_eval = {
        "pape": _wmean(pooled_self_pape),
        "hr@1": _wmean(pooled_self_hr1),
        "hr@2": _wmean(pooled_self_hr2),
        "mae":  _wmean(pooled_self_mae),
        "n_cold_windows": sum(n for _, n in pooled_self_pape),
        "n_cold_apts": len(per_apt),
    }
    holdout_eval = {
        "pape": _wmean(pooled_ho_pape),
        "hr@1": _wmean(pooled_ho_hr1),
        "hr@2": _wmean(pooled_ho_hr2),
        "mae":  _wmean(pooled_ho_mae),
        "n_cold_windows": sum(n for _, n in pooled_ho_pape),
        "n_cold_apts": len(per_apt),
    }

    print(f"[v04 LO-holdout] aggregated:")
    print(f"  self_eval    PAPE={self_eval['pape']:.2f}  HR@1={self_eval['hr@1']:.1f}  "
          f"HR@2={self_eval['hr@2']:.1f}  MAE={self_eval['mae']:.4f}  "
          f"(n_windows={self_eval['n_cold_windows']})")
    print(f"  holdout_eval PAPE={holdout_eval['pape']:.2f}  HR@1={holdout_eval['hr@1']:.1f}  "
          f"HR@2={holdout_eval['hr@2']:.1f}  MAE={holdout_eval['mae']:.4f}  "
          f"(n_windows={holdout_eval['n_cold_windows']})")
    print(f"[v04 LO-holdout] elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    out = {
        "algorithm": "local_only_holdout",
        "seed": int(args.seed),
        "config": {
            "rounds": args.rounds, "lr": args.lr, "batch_size": args.batch_size,
            "weight_decay": args.weight_decay, "use_amp": cfg.use_amp,
        },
        "self_eval": self_eval,
        "holdout_eval": holdout_eval,
        "per_apt_metrics": per_apt,
        "n_train_windows": int(n_train_total),
        "n_holdout_windows": int(n_ho_total),
        "elapsed_seconds": elapsed,
        "gpu_at_start": gpu_start,
        "gpu_at_end": gpu_end,
        "comment": (
            "Local-only NBEATSx with proper held-out evaluation. self_eval is "
            "the SAME metric the parent folder's local_only cell reports "
            "(self-train + self-eval on series[:train_end] — overfit upper "
            "bound). holdout_eval is the new fair-generalisation metric on "
            "series[train_end:val_end] (the val portion of the v01 7:1:2 "
            "split). z-norm uses the apt's train-segment statistics for both "
            "evaluations (warm-start). Local-only's row in the unified pFL "
            "paper should switch from self_eval to holdout_eval as the "
            "headline number; the gap between the two also quantifies the "
            "self-train overfitting bias the parent folder's footnote calls out."
        ),
    }

    with open(out_dir / "result.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[v04 LO-holdout] saved -> {out_dir}")


if __name__ == "__main__":
    main()


# Expected output (seed=42, GTX 5070 Ti):
#   - 20 cold apts × ~30s self-train each ≈ 10-12 min wall-clock
#     (parent local_only cell took ~12 min on seed=42).
#   - self_eval is expected to match the parent folder's local_only cell
#     within ±0.1 kW PAPE (the only difference is amp_ctx is identical and
#     no other code path varies).
#   - holdout_eval is expected to be MEASURABLY worse than self_eval —
#     this is the whole point of the redesign. Order-of-magnitude guess:
#     +5 to +15 kW PAPE, depending on how much the per-apt model overfits
#     the train segment in 20 epochs of self-train.
