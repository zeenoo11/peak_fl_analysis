"""Priority 2C — FedProto baseline (Tan et al., AAAI 2022) adapted to forecasting.

Why
---
The unified pFL paper claims a cluster-prototype recipe (Peak-VQ + W5
hybrid) outperforms federated baselines, but does not include a *peer
prototype-based pFL* baseline — leaving the headline open to the
question "is the gap really about peak-shape templating, or is any
prototype-aligned representation enough?". FedProto is the closest
published prototype-based pFL competitor: it federates per-class
prototypes alongside the representation network and adds an MSE-on-
latents alignment term to the local loss. Adding the FedProto row makes
the cluster-prototype-vs-prototype comparison ceteris-paribus.

Forecasting adaptation (documented at length in src/fl/fedproto.py)
------------------------------------------------------------------
There are no class labels in load forecasting; "per-class" prototypes
become "per-cluster" prototypes (KMeans on h_g latents, K=32 to peer
v01-v02's M=32 codebook). Backbone weights are still federated by
FedAvg; per-cluster prototypes are aggregated by count-weighted mean
across clients each round; per-batch latents are pulled toward the
nearest **round-start** global prototype via λ_proto · MSE.

Hyperparameters
---------------
- ``K = 32``           — peers v01/v02's M=32 codebook size for
                         apples-to-apples representation-richness
                         comparison.
- ``lambda_proto = 0.1`` — keeps the MAE objective dominant; FedProto
                         paper §5 used λ ∈ [0.01, 1] for image tasks.
                         A smaller λ would make FedProto degenerate
                         to FedAvg; a larger λ risks suppressing the
                         forecasting objective.
- Other FL hyperparameters match the parent folder's defaults
  (rounds=20, local_epochs=2, lr=1e-3, batch_size=512, bf16 autocast).

Cold inference (no W5 correction)
---------------------------------
FedProto's contribution is the prototype-aligned representation, not a
peak-shape template. For each cold apt we forward through the federated
backbone and report PAPE / HR@1 / HR@2 / MAE on the raw forecast — the
same protocol as ``v04 01_fl_train.py``'s evaluate_cold flow. As a
diagnostic we ALSO record the cold-window cluster assignment histogram
(1-NN on the global prototypes) so a downstream analysis can compare
cold vs train cluster usage.

CLI
---
    uv run python experiments/v04_full_baseline_comparison/09_fix_rerun/04_fedproto.py --seed 42

Output
------
    outputs/v04_full_baseline_comparison/09_fix_rerun/seed{S}/fedproto/
        result.json
            ├── algorithm: "fedproto"
            ├── config: {rounds, local_epochs, lr, batch_size, K, lambda_proto, ...}
            ├── history: per-round main_loss + proto_loss
            ├── cold_metrics: {pape, hr@1, hr@2, mae, n_cold_windows, n_cold_apts}
            ├── prototype_diagnostics: {global_prototype_norms,
            │                          train_assignment_histogram_last_round,
            │                          cold_assignment_histogram,
            │                          cold_n_clusters_used}
        prototypes.npz             — final global prototypes [K, D] (numpy)
        final_state_dict.pt        — federated NBEATSx backbone

Wall-clock per seed: ~25 min on a 5070 Ti
    (FedAvg ~12 min × overhead 1.5-2× from per-client KMeans + extra
     proto-loss forward; well within v04's per-cell budget.)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch

from config import OUTPUT_DIR, RANDOM_SEED
from dataloader.splits import load_v02_split
from fl.fedproto import FedProtoConfig, train_fedproto

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


def main() -> None:
    ap = argparse.ArgumentParser(description="v04 09_fix_rerun: FedProto (Priority 2C).")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--K", type=int, default=32, help="Number of prototypes (peers v01/v02 M=32).")
    ap.add_argument("--lambda_proto", type=float, default=0.1,
                    help="Prototype-alignment loss weight; default 0.1 keeps MAE dominant.")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg = FedProtoConfig(
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        seed=args.seed,
        use_amp=not args.no_amp,
        K=args.K,
        lambda_proto=args.lambda_proto,
    )

    sp = load_v02_split(args.seed)
    train_apts, cold_apts = sp["train"], sp["cold"]
    out_dir = V04_FIX_OUT_ROOT / f"seed{args.seed}" / "fedproto"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v04 2C] seed={args.seed}  K={args.K}  lambda_proto={args.lambda_proto}")
    print(f"[v04 2C] config: {cfg}")
    gpu_start = _gpu_snapshot()
    print(f"[v04 2C] GPU @start: {gpu_start}")

    t0 = time.time()
    out = train_fedproto(train_apts, cold_apts, cfg)
    elapsed = time.time() - t0
    gpu_end = _gpu_snapshot()
    print(f"[v04 2C] GPU @end: {gpu_end}")

    cm = out["cold_metrics"]
    print(f"[v04 2C] cold: PAPE={cm.get('pape', float('nan')):.2f}  "
          f"HR@1={cm.get('hr@1', float('nan')):.1f}  "
          f"HR@2={cm.get('hr@2', float('nan')):.1f}")
    print(f"[v04 2C] elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Pop the heavy artefacts before JSON dump.
    final_sd = out.pop("final_state_dict", None)
    global_protos = out.pop("global_prototypes", None)

    out["seed"] = int(args.seed)
    out["elapsed_seconds"] = elapsed
    out["gpu_at_start"] = gpu_start
    out["gpu_at_end"] = gpu_end
    out["config"] = asdict(cfg)
    out["comment"] = (
        "FedProto (Tan et al., AAAI 2022) adapted to load forecasting via "
        "per-cluster prototypes (KMeans, K=32) instead of per-class. Backbone "
        "is FedAvg-aggregated NBEATSx; prototypes are per-cluster count-weighted "
        "mean across clients each round; local loss adds lambda_proto·MSE between "
        "h_g and the nearest round-start global prototype. Cold inference uses "
        "the federated backbone with no W5 correction (FedProto's contribution is "
        "the prototype-aligned representation). cold_assignment_histogram and "
        "train_assignment_histogram_last_round are recorded so an aggregator can "
        "compare cluster usage between train and cold pools."
    )

    with open(out_dir / "result.json", "w") as fh:
        json.dump(out, fh, indent=2)
    if final_sd is not None:
        torch.save(final_sd, out_dir / "final_state_dict.pt")
    if global_protos is not None:
        np.savez(out_dir / "prototypes.npz", global_prototypes=global_protos.astype(np.float32))
    print(f"[v04 2C] saved -> {out_dir}")


if __name__ == "__main__":
    main()


# Expected output (seed=42, GTX 5070 Ti):
#   - 20 rounds × 80 clients; each client does (a) local SGD with proto-loss,
#     (b) post-train h_g forward for KMeans warm-start. Proto KMeans is small
#     (K=32 on a few hundred latents per client); the dominant cost is the
#     extra forward pass for centroid computation.
#   - Total ~22-30 min per seed depending on GPU contention.
#   - Expected cold PAPE: ~50-58 kW (in the FL-baseline cluster from the
#     parent v04 results — 56-57 kW for FedAvg/FedProx/FedRep/Ditto).
#     If FedProto materially beats this band it would be evidence that
#     prototype alignment alone helps cold-side; if it lands inside the
#     band, the pFL paper's "cluster-prototype + W5 hybrid" recipe carries
#     more weight than FedProto's prototype-only recipe.
