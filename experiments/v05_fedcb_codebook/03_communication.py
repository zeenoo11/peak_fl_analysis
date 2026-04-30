"""v05 communication accounting (V5-FedCB-4, seed-independent).

(한글 요약)
v05 federated codebook 파이프라인의 통신 비용을 paper.md §4.7 형식과 호환되는
JSON으로 기록한다. ``K_local ∈ {2, 4, 8}`` 각각에 대해

    local_upload_per_client     = K_local × (D + 1) × 4 bytes   (centroids + counts)
    broadcast                   = M × D × 4 bytes               (codebook 한 번 broadcast)
    residual_aggregation_per_cl = M × (H + 1) × 4 bytes         (residual 합 + 카운트)
    bytes_per_round_total       = N_clients × (local + residual) + broadcast
    boundary_crosses            = 2  (local centroids 업로드 1회 + residual 업로드 1회;
                                     broadcast은 server→client 방향이라 별도로
                                     세지 않는 게 paper.md §4.7의 컨벤션이다.)

이 스크립트는 backbone forward / data를 건드리지 않으므로 seed-independent.

CLI:
    uv run python experiments/v05_fedcb_codebook/03_communication.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import HORIZON, OUTPUT_DIR

V05_OUT_ROOT = OUTPUT_DIR / "v05_fedcb_codebook"

# Plan §"Hard-coded shapes" + paper.md §4.7 conventions.
N_CLIENTS = 80           # UMass v02 80:20 train cohort
D_LATENT = 64            # h_generic dim
M_GLOBAL = 32            # global codebook size
BYTES_PER_FP32 = 4
K_LOCAL_SWEEP = [2, 4, 8]


def _entry(K_local: int) -> dict:
    """One row of the communication table for a given ``K_local``."""
    local_upload_per_client = K_local * (D_LATENT + 1) * BYTES_PER_FP32
    broadcast = M_GLOBAL * D_LATENT * BYTES_PER_FP32
    residual_per_client = M_GLOBAL * (HORIZON + 1) * BYTES_PER_FP32
    bytes_per_round_total = (
        N_CLIENTS * local_upload_per_client
        + broadcast
        + N_CLIENTS * residual_per_client
    )
    return {
        "K_local": int(K_local),
        "what_uploaded": (
            f"Stage 1: per-client {K_local} centroids (64-d) + counts; "
            f"Stage 3: per-client {M_GLOBAL} residual sums (24-d) + counts. "
            f"Server broadcasts the {M_GLOBAL}-cluster codebook once."
        ),
        "n_clients": int(N_CLIENTS),
        "h_g_dim": int(D_LATENT),
        "M_codebook": int(M_GLOBAL),
        "horizon": int(HORIZON),
        "bytes_per_fp32": int(BYTES_PER_FP32),
        "local_upload_per_client": int(local_upload_per_client),
        "broadcast": int(broadcast),
        "residual_per_client": int(residual_per_client),
        "bytes_per_round_per_client": int(local_upload_per_client + residual_per_client),
        "bytes_per_round_total": int(bytes_per_round_total),
        "n_rounds": 1,                        # single-shot
        "total_bytes": int(bytes_per_round_total),
        "boundary_crosses": 2,                # Stage-1 upload + Stage-3 upload
        "notes": (
            "Hierarchical 2-stage single-shot federated KMeans + federated "
            "residual aggregation; boundary crosses count client → server "
            "uploads only (server → client broadcast is excluded, matching "
            "v04 communication_summary.json convention)."
        ),
    }


def main() -> None:
    rows = [_entry(K) for K in K_LOCAL_SWEEP]

    # Cross-method context — copy the relevant rows from the v04 paper §4.7
    # numerically so the v05 summary is self-contained for paper writing.
    # These are the verbatim Table 4 numbers in paper.md §4.7.
    context = {
        "fedavg_fedprox_ditto": {
            "bytes_per_round_per_client": 262_736,
            "bytes_per_round_total": 21_018_880,
            "n_rounds": 20,
            "total_bytes": 420_377_600,
            "boundary_crosses": 20,
            "notes": "Iterative FL of full backbone weights (paper.md §4.7).",
        },
        "fedrep": {
            "bytes_per_round_per_client": 224_256,
            "bytes_per_round_total": 17_940_480,
            "n_rounds": 20,
            "total_bytes": 358_809_600,
            "boundary_crosses": 20,
            "notes": "Iterative FL of encoder only; head is per-client (paper.md §4.7).",
        },
        "v01_v03_centralised_codebook_one_shot": {
            "bytes_upload": 4_928_000,
            "bytes_broadcast": 11_264,
            "bytes_per_round_total": 4_939_264,
            "n_rounds": 1,
            "total_bytes": 4_939_264,
            "boundary_crosses": 1,
            "notes": (
                "Centralised codebook anchor: one-shot upload of the 19,250 train-"
                "window h_g pool (~4.93 MB), centroids + offsets broadcast back. "
                "v05 federated alternative replaces this single boundary cross with "
                "two per-client crosses (Stage 1 + Stage 3) but ships dramatically "
                "less raw signal."
            ),
        },
    }

    summary = {
        "n_clients": int(N_CLIENTS),
        "h_g_dim": int(D_LATENT),
        "M_codebook": int(M_GLOBAL),
        "horizon": int(HORIZON),
        "bytes_per_fp32": int(BYTES_PER_FP32),
        "K_local_sweep": rows,
        "context": context,
        "comment": (
            "v05 V5-FedCB-4: seed-independent communication accounting. The "
            "K_local sweep rows are the per-K_local cost; the 'context' block "
            "carries paper.md §4.7 Table 4 numbers verbatim so the v05 paper "
            "patch (plan step 6) can quote them without recomputation."
        ),
    }

    V05_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = V05_OUT_ROOT / "communication_summary.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[v05 communication] written -> {out_path}")
    print()
    print(f"{'K_local':>8} {'per-client/round':>20} {'total/round':>16} "
          f"{'rounds':>7} {'total bytes':>16}")
    print("-" * 75)
    for row in rows:
        print(f"{row['K_local']:>8} {row['bytes_per_round_per_client']:>20,} "
              f"{row['bytes_per_round_total']:>16,} {row['n_rounds']:>7} "
              f"{row['total_bytes']:>16,}")


if __name__ == "__main__":
    main()
