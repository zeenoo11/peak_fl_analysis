"""v04 G7 — Communication-cost accounting (seed-independent).

Computes bytes-per-round and total bytes for each FL baseline + the
v02 1-shot codebook upload. Defends v02's "1 boundary cross"
efficiency claim against iterative FL.

We count **upload bytes only** (client → server). Download (server →
client) is symmetric for FedAvg and friends but our baselines are
upload-dominated for the comparison narrative ("how much does each
client communicate").

Bytes model
-----------
- One float = 4 bytes (fp32 weights, the canonical FL model exchange).
- One MinimalNBEATSx forward-only state dict has ``P_backbone`` floats.
- FedAvg / FedProx: each round, every client uploads the **full
  backbone**: bytes_per_round = 80 × P_backbone × 4.
- FedRep: clients upload the **encoder only**, head stays local:
  bytes_per_round = 80 × P_encoder × 4.
- Ditto: FedAvg-style global update (full backbone) per round; the
  per-client personal model is local-only.
- v01-v03 method (Peak-VQ): one-shot upload of {h_g}_{j} aggregated
  for the central KMeans, plus the codebook + offsets broadcast back.
  bytes_total = N_train_windows × D_h_g × 4 (upload) +
                M × D_h_g × 4 + M × H × 4 (broadcast).
  Boundary crosses: 1 (vs 20 rounds for FL).

Output is **seed-independent** in the architectural sense — it depends
only on parameter counts and the protocol. We use the numbers from the
seed=42 run for concrete codebook size if available; otherwise compute
from the canonical M=32, D=64, H=24 constants.

CLI:

    uv run python experiments/v04_full_baseline_comparison/06_communication.py

Output:
    outputs/v04_full_baseline_comparison/communication_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from config import HORIZON, OUTPUT_DIR
from models.nbeatsx import MinimalNBEATSx

V04_OUT_ROOT = OUTPUT_DIR / "v04_full_baseline_comparison"

BYTES_PER_FP32 = 4

# Canonical v01-v03 hyperparameters (referenced in the bytes model below).
N_TRAIN_APTS = 80
N_TRAIN_WINDOWS_PER_APT = 12_020 // 50  # ~240 from v01 §3.3 numbers, tightened later
N_TRAIN_WINDOWS_TOTAL = 19_250          # matched against v02 03_fit_codebook output
D_H_G = 64                              # NBEATSx generic-stack hidden width
M_CODEBOOK = 32                         # KMeans clusters
H_HORIZON = HORIZON                     # 24


def _count_params(model: torch.nn.Module) -> dict:
    """Total + per-name groupings used by FL algorithms (encoder vs head)."""
    total = 0
    encoder = 0   # everything that is not stack_*.proj.*
    head = 0      # stack_*.proj.*  (FedRep's head)
    for n, p in model.named_parameters():
        n_p = p.numel()
        total += n_p
        if ".proj." in n:
            head += n_p
        else:
            encoder += n_p
    return {"total": total, "encoder": encoder, "head": head}


def main() -> None:
    ap = argparse.ArgumentParser(description="v04 G7 communication-cost accounting.")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--n_clients", type=int, default=N_TRAIN_APTS)
    args = ap.parse_args()

    # Build a backbone once just to count its params; no GPU needed.
    bb = MinimalNBEATSx()
    pcounts = _count_params(bb)
    print(f"[v04 G7] backbone param counts: {pcounts}")

    # Per-client bytes per round (upload).
    full_bytes_per_round_per_client = pcounts["total"] * BYTES_PER_FP32
    enc_bytes_per_round_per_client = pcounts["encoder"] * BYTES_PER_FP32
    head_bytes_per_round_per_client = pcounts["head"] * BYTES_PER_FP32

    # Aggregate per-round upload across all participating clients.
    rounds = args.rounds
    n_clients = args.n_clients

    methods: list[dict] = []

    # FedAvg / FedProx / Ditto: full-backbone upload.
    for algo in ["fedavg", "fedprox", "ditto"]:
        bytes_per_round = n_clients * full_bytes_per_round_per_client
        methods.append({
            "method": algo,
            "what_uploaded": "full backbone weights",
            "params_per_client_upload": pcounts["total"],
            "bytes_per_round_per_client": full_bytes_per_round_per_client,
            "bytes_per_round_total": bytes_per_round,
            "n_rounds": rounds,
            "total_bytes": bytes_per_round * rounds,
            "boundary_crosses": rounds,  # one cross per round (upload)
            "notes": ("Ditto's per-client personal model adds local compute but no upload "
                      "(personal model never leaves the client)."
                      if algo == "ditto" else None),
        })

    # FedRep: encoder-only upload, head local.
    methods.append({
        "method": "fedrep",
        "what_uploaded": "encoder only (per-client head stays local)",
        "params_per_client_upload": pcounts["encoder"],
        "bytes_per_round_per_client": enc_bytes_per_round_per_client,
        "bytes_per_round_total": n_clients * enc_bytes_per_round_per_client,
        "n_rounds": rounds,
        "total_bytes": n_clients * enc_bytes_per_round_per_client * rounds,
        "boundary_crosses": rounds,
        "notes": f"head ({pcounts['head']} params) is per-client; never crosses the boundary.",
    })

    # Local-only: no FL communication at all.
    methods.append({
        "method": "local_only",
        "what_uploaded": "nothing",
        "params_per_client_upload": 0,
        "bytes_per_round_per_client": 0,
        "bytes_per_round_total": 0,
        "n_rounds": 0,
        "total_bytes": 0,
        "boundary_crosses": 0,
        "notes": "no federation; each cold gucha trains locally on its own data.",
    })

    # v01-v03 method (Peak-VQ): one-shot codebook artefact.
    # Upload: N_train_windows × D_h_g (each train apt sends its h_g latents to the server once).
    # Broadcast: M × D_h_g (centroids) + M × H (offsets).
    pv_upload = N_TRAIN_WINDOWS_TOTAL * D_H_G * BYTES_PER_FP32
    pv_broadcast = M_CODEBOOK * D_H_G * BYTES_PER_FP32 + M_CODEBOOK * H_HORIZON * BYTES_PER_FP32
    pv_total = pv_upload + pv_broadcast
    methods.append({
        "method": "peak_vq_v01_v03",
        "what_uploaded": (
            "one-shot aggregate of train-side h_g latents (N x D); "
            "server broadcasts back centroids (M x D) + offsets (M x H)."
        ),
        "n_train_windows_total": N_TRAIN_WINDOWS_TOTAL,
        "h_g_dim": D_H_G,
        "M_codebook": M_CODEBOOK,
        "horizon": H_HORIZON,
        "bytes_upload": pv_upload,
        "bytes_broadcast": pv_broadcast,
        "bytes_per_round_total": pv_total,    # interpreted as "single-round" cost
        "n_rounds": 1,
        "total_bytes": pv_total,
        "boundary_crosses": 1,
        "notes": ("post-hoc 1-shot KMeans; 20× fewer boundary crosses than iterative FL "
                  "at rounds=20."),
    })

    summary = {
        "n_clients": n_clients,
        "n_rounds_iterative_FL": rounds,
        "bytes_per_fp32": BYTES_PER_FP32,
        "backbone_param_counts": pcounts,
        "methods": methods,
    }
    out_path = V04_OUT_ROOT / "communication_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[v04 G7] saved -> {out_path}")
    print()
    print(f"  {'method':<22} {'bytes/round':>14} {'rounds':>7} {'total bytes':>16} {'crosses':>9}")
    print(f"  {'-'*22} {'-'*14} {'-'*7} {'-'*16} {'-'*9}")
    for m in methods:
        print(f"  {m['method']:<22} {m['bytes_per_round_total']:>14,} "
              f"{m['n_rounds']:>7} {m['total_bytes']:>16,} {m['boundary_crosses']:>9}")


if __name__ == "__main__":
    main()
