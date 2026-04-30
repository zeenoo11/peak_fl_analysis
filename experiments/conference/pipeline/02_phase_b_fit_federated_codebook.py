"""Phase B — federated 2-stage hierarchical KMeans + federated residual offsets.

(한글 요약)
KIIE conference 발표 (``papers/conference_draft/presentation.md``)의 §3.3
"Federated Codebook Construction" 본문 (Stage 1 / Stage 2 / Stage 3) 그대로
반영하는 per-seed 드라이버. Phase A에서 federated 학습된 NBEATSxAux 백본을 freeze
한 채로:
    Stage 1: 각 가구가 자기 학습 윈도우에서 ``h_generic``을 추출 후 local
             KMeans++로 ``K_local`` (default 4) 개 centroid를 만든다 — raw h_g는
             가구 내에 머문다.
    Stage 2: 서버가 80가구의 ``80 × K_local`` local centroid를 모아
             sample-count-weighted KMeans++로 다시 클러스터링 → 32-entry global
             codebook을 broadcast.
    Stage 3: 각 가구가 자기 윈도우들을 global codebook으로 routing 후 cluster별
             forecast residual의 partial sum과 count를 업로드 → 서버에서 cluster-
             aggregated mean residual ``o_c``를 계산.

본 스크립트는 *연합* 경로의 Phase B만 담당한다. ``α``는 Phase C의 인자이므로 여기
없다. 결과 ``codebook.npz`` (codebook + offsets fp32) 와 진단/메타 ``codebook_meta.json``
이 produced되어 Phase C가 consume.

**Federated contract is enforced by import structure**: 본 파일은 ``src/fl/codebook_fl``
의 *forwarding* helper (``local_codebook_step``, ``merge_local_codebooks``,
``federated_residual_offsets``)만 import하고, 어떤 centralised pooling helper
(예: v04 09_fix_rerun의 ``gather_train_segment_aux``)도 호출하지 않는다 — 이것이
"Phase B는 fully-federated"라는 §3.3 주장을 코드 차원에서 보증한다.

Per-seed argparse — 멀티시드 sweep ({42, 123, 7})은 외부 launcher가 ``--seed S``로
세 번 호출 (memory: feedback_argparse_per_seed). MLflow 사용 안 함.
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

from config import HORIZON, OUTPUT_DIR, RANDOM_SEED
from dataloader.splits import load_v02_split

# IMPORTANT: federation contract — only forwarding helpers are imported.
# Do NOT import gather_train_segment_aux or any other centralised pooling
# routine here; if a future contributor adds one, the "Phase B is federated"
# claim from presentation.md §3.3 silently breaks.
from fl.base import DEVICE, build_clients
from fl.codebook_fl import (
    federated_residual_offsets,
    local_codebook_step,
    merge_local_codebooks,
)
from models.nbeatsx_aux import NBEATSxAux

CONFERENCE_OUT_ROOT = OUTPUT_DIR / "conference" / "pipeline"


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


def _communication_bytes(
    *,
    n_train_clients: int,
    K_local: int,
    M_global: int,
    horizon: int,
) -> dict[str, int]:
    """Communication accounting (bytes) — re-uses the formula from
    ``experiments/v05_fedcb_codebook/01_fedcb_codebook.py``.

    ``local_upload_per_client = K_local × (D + 1) × 4``  (centroids + counts)
    ``broadcast              = M × D × 4``                (codebook broadcast)
    ``residual_per_client    = M × (H + 1) × 4``          (residuals + counts)
    """
    D = 64  # h_generic dim
    local_upload_per_client = int(K_local * (D + 1) * 4)
    broadcast = int(M_global * D * 4)
    residual_per_client = int(M_global * (horizon + 1) * 4)
    total_round = int(
        n_train_clients * local_upload_per_client
        + broadcast
        + n_train_clients * residual_per_client
    )
    return {
        "local_upload_per_client": local_upload_per_client,
        "broadcast": broadcast,
        "residual_per_client": residual_per_client,
        "total_round": total_round,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Conference Phase B — federated 2-stage hierarchical KMeans + "
            "federated residual offsets, on top of a Phase-A FedAvg-NBEATSxAux "
            "backbone. Loads the Phase-A artefact for the same seed; halts if "
            "missing (does NOT silently retrain)."
        )
    )
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--K_local", type=int, default=4,
                    help="Stage-1 cluster count per client (V5-FedCB-1 default = 4).")
    ap.add_argument("--M", type=int, default=32, help="Global codebook size.")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=HORIZON,
                    help="Codebook fit stride; v01/v02 default = 24.")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    sp = load_v02_split(args.seed)
    train_apts = sp["train"]

    phase_a_dir = CONFERENCE_OUT_ROOT / f"seed{args.seed}" / "phase_a"
    backbone_ckpt = phase_a_dir / "final_state_dict.pt"
    out_dir = CONFERENCE_OUT_ROOT / f"seed{args.seed}" / "phase_b"

    if not backbone_ckpt.exists():
        raise FileNotFoundError(
            f"Phase B requires a Phase A backbone artefact at {backbone_ckpt}. "
            f"Run experiments/conference/pipeline/01_phase_a_train_backbone.py "
            f"--seed {args.seed} first (does NOT silently re-train)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[conference phase_b] seed={args.seed}  K_local={args.K_local}  M={args.M}  "
        f"batch={args.batch_size}  stride={args.stride}"
    )
    print(f"[conference phase_b] backbone_source={backbone_ckpt}")
    gpu_start = _gpu_snapshot()
    print(f"[conference phase_b] GPU @start: {gpu_start}")

    t0 = time.time()

    # Load the frozen FedAvg-NBEATSxAux backbone from Phase A.
    model = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
    model.load_state_dict(
        torch.load(backbone_ckpt, map_location="cpu", weights_only=False),
        strict=True,
    )

    # Build train clients (one ClientData per train apt).
    clients = build_clients(train_apts)
    if not clients:
        raise RuntimeError(
            "Phase B: no train clients (all apts missing? check data/raw/Umass)."
        )

    # Stage 1 — local KMeans on each client's h_g (raw h_g never leaves client).
    packets = []
    n_train_windows_total = 0
    for ci, client in enumerate(clients):
        pkt = local_codebook_step(
            model, client,
            K_local=args.K_local, seed=args.seed,
            batch_size=args.batch_size, stride=args.stride,
        )
        packets.append(pkt)
        n_train_windows_total += int(pkt["h_g"].shape[0])
        if (ci + 1) % 20 == 0 or (ci + 1) == len(clients):
            print(
                f"[conference phase_b] Stage 1: {ci + 1}/{len(clients)} clients, "
                f"K_local_i={pkt['K_local_i']}, N_i={pkt['h_g'].shape[0]}"
            )

    # Stage 2 — server merge (sample-weight = local cluster counts).
    merge = merge_local_codebooks(packets, M_global=args.M, seed=args.seed)
    print(
        f"[conference phase_b] Stage 2: util={merge['utilization']:.3f}  "
        f"ppl={merge['perplexity']:.2f}  k_min={merge['k_min']}  "
        f"k_max={merge['k_max']}  n_empty={merge['n_empty_clusters']}  "
        f"stage2_inertia={merge['stage2_inertia']:.1f}"
    )

    # Stage 3 — federated residual offsets.
    offsets = federated_residual_offsets(packets, merge["codebook"])

    elapsed = time.time() - t0
    print(f"[conference phase_b] Phase B done in {elapsed:.1f}s")

    # Communication accounting (per the plan / §4.7).
    n_train_clients = len(clients)
    comm = _communication_bytes(
        n_train_clients=n_train_clients,
        K_local=int(args.K_local),
        M_global=int(args.M),
        horizon=HORIZON,
    )

    # Persist codebook artefact (consumed by Phase C).
    np.savez(
        out_dir / "codebook.npz",
        codebook=merge["codebook"].astype(np.float32),
        offsets=offsets.astype(np.float32),
    )

    meta = {
        "seed": int(args.seed),
        "K_local": int(args.K_local),
        "M_global": int(args.M),
        "vq_diagnostics": {
            "utilization": float(merge["utilization"]),
            "perplexity": float(merge["perplexity"]),
            "k_min": int(merge["k_min"]),
            "k_max": int(merge["k_max"]),
            "n_empty_clusters": int(merge["n_empty_clusters"]),
            "stage1_mean_inertia": float(merge["stage1_mean_inertia"]),
            "stage2_inertia": float(merge["stage2_inertia"]),
        },
        "n_train_clients": int(n_train_clients),
        "n_train_windows_total": int(n_train_windows_total),
        "K_local_i_per_client": [int(p["K_local_i"]) for p in packets],
        "communication_bytes": comm,
        "elapsed_seconds": float(elapsed),
        "backbone_source": str(backbone_ckpt),
        "config": {
            "K_local": int(args.K_local),
            "M_global": int(args.M),
            "batch_size": int(args.batch_size),
            "stride": int(args.stride),
            "use_amp": bool(DEVICE.type == "cuda"),
        },
        "gpu_at_start": gpu_start,
        "gpu_at_end": _gpu_snapshot(),
        "comment": (
            "Conference Phase B: hierarchical 2-stage federated KMeans + "
            "federated residual offsets. Stage 1 / Stage 2 / Stage 3 follow "
            "presentation.md §3.3 'Federated Codebook Construction'. Raw h_g "
            "and raw forecast residuals never leave the client; only "
            "(centroid, count) and (residual partial sum, count) tuples are "
            "uploaded."
        ),
    }
    with open(out_dir / "codebook_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[conference phase_b] saved -> {out_dir}")


if __name__ == "__main__":
    main()
