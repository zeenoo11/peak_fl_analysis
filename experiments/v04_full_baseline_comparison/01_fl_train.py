"""v04 FL training — one seed × one algorithm per invocation.

Per-seed argparse pattern (matches v01-v03 conventions):

    uv run python experiments/v04_full_baseline_comparison/01_fl_train.py \\
        --seed 42 --algorithm fedavg

    uv run python experiments/v04_full_baseline_comparison/01_fl_train.py \\
        --seed 42 --algorithm fedprox    # default mu=0.01
    uv run python experiments/v04_full_baseline_comparison/01_fl_train.py \\
        --seed 42 --algorithm fedrep     # default head_epochs=1
    uv run python experiments/v04_full_baseline_comparison/01_fl_train.py \\
        --seed 42 --algorithm ditto      # default lam=0.1
    uv run python experiments/v04_full_baseline_comparison/01_fl_train.py \\
        --seed 42 --algorithm local_only

Outputs:

    outputs/v04_full_baseline_comparison/seed{S}/{algorithm}/result.json
    outputs/v04_full_baseline_comparison/seed{S}/{algorithm}/final_state_dict.pt
        (FL algorithms only — local_only writes per_apt_metrics in result.json instead).

VRAM / system snapshot is logged at start + finish via nvidia-smi.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from config import OUTPUT_DIR, RANDOM_SEED
from dataloader.splits import load_v02_split
from fl import (
    DittoConfig,
    FedProxConfig,
    FedRepConfig,
    FLConfig,
    LocalOnlyConfig,
    train_ditto,
    train_fedavg,
    train_fedprox,
    train_fedrep,
    train_local_only,
)

V04_OUT_ROOT = OUTPUT_DIR / "v04_full_baseline_comparison"

ALGORITHMS = {
    "fedavg":     (FLConfig,        train_fedavg),
    "fedprox":    (FedProxConfig,   train_fedprox),
    "fedrep":     (FedRepConfig,    train_fedrep),
    "ditto":      (DittoConfig,     train_ditto),
    "local_only": (LocalOnlyConfig, train_local_only),
}


def _gpu_snapshot() -> dict:
    """nvidia-smi snapshot; falls back to torch.cuda if smi missing."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        used, free, total, util = (int(s.strip()) for s in out.strip().split(","))
        return {"used_MiB": used, "free_MiB": free, "total_MiB": total, "util_pct": util}
    except Exception:
        if torch.cuda.is_available():
            return {
                "used_MiB": int(torch.cuda.memory_allocated() / 1024**2),
                "reserved_MiB": int(torch.cuda.memory_reserved() / 1024**2),
            }
        return {"cpu_only": True}


def main() -> None:
    ap = argparse.ArgumentParser(description="v04 FL training (one seed × one algorithm).")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--algorithm", required=True, choices=list(ALGORITHMS.keys()))
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--no_amp", action="store_true")
    # Algorithm-specific extras (use the right one for your --algorithm).
    ap.add_argument("--mu", type=float, default=0.01, help="FedProx proximal-term weight.")
    ap.add_argument("--head_epochs", type=int, default=1, help="FedRep head epochs (rep_epochs = local_epochs - head_epochs).")
    ap.add_argument("--lam", type=float, default=0.1, help="Ditto personal-model regulariser.")
    args = ap.parse_args()

    cfg_cls, train_fn = ALGORITHMS[args.algorithm]
    kwargs = dict(
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        use_amp=not args.no_amp,
    )
    if args.algorithm == "fedprox":
        kwargs["mu"] = args.mu
    elif args.algorithm == "fedrep":
        kwargs["head_epochs"] = args.head_epochs
    elif args.algorithm == "ditto":
        kwargs["lam"] = args.lam
    cfg = cfg_cls(**kwargs)

    sp = load_v02_split(args.seed)
    train_apts, cold_apts = sp["train"], sp["cold"]

    seed_root = V04_OUT_ROOT / f"seed{args.seed}"
    out_dir = seed_root / args.algorithm
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v04 FL] seed={args.seed}  algorithm={args.algorithm}")
    print(f"[v04 FL] config: {cfg}")
    gpu_start = _gpu_snapshot()
    print(f"[v04 FL] GPU @start: {gpu_start}")

    t0 = time.time()
    out = train_fn(train_apts, cold_apts, cfg)
    elapsed = time.time() - t0

    gpu_end = _gpu_snapshot()
    print(f"[v04 FL] GPU @end:   {gpu_end}")
    cm = out["cold_metrics"]
    print(
        f"[v04 FL] cold: PAPE={cm.get('pape', float('nan')):.2f}  "
        f"HR@1={cm.get('hr@1', float('nan')):.1f}  HR@2={cm.get('hr@2', float('nan')):.1f}"
    )
    print(f"[v04 FL] elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Strip the state-dict tensors out before JSON dump — they go to a separate .pt file.
    final_sd = out.pop("final_state_dict", None)
    extra_sd_keys = [k for k in list(out.keys()) if k.endswith("_state_dict")]
    extras_sd = {k: out.pop(k) for k in extra_sd_keys}
    out["seed"] = int(args.seed)
    out["elapsed_seconds"] = elapsed
    out["gpu_at_start"] = gpu_start
    out["gpu_at_end"] = gpu_end
    out["config"] = asdict(cfg)

    with open(out_dir / "result.json", "w") as fh:
        json.dump(out, fh, indent=2)
    if final_sd is not None:
        torch.save(final_sd, out_dir / "final_state_dict.pt")
    for k, sd in extras_sd.items():
        if sd is not None:
            torch.save(sd, out_dir / f"{k}.pt")
    print(f"[v04 FL] saved -> {out_dir}")


if __name__ == "__main__":
    main()
