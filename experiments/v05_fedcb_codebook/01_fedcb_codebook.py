"""v05 FedCB — federated codebook + CMO-only correction (per-seed driver).

(한글 요약)
v05는 v04 09_fix_rerun의 ``Stacked-Aux`` 파이프라인 (FedAvg-NBEATSxAux 백본 +
중앙화된 KMeans++ 코드북 + W5 hybrid 보정) 중 ``Phase B`` 부분을 *연합형*으로
바꾸고 ``Phase C`` 보정을 ``CMO-only`` (cluster-mean offset 만, Gaussian
template α_w1 항을 제거)로 단순화한 변형이다.

이 스크립트는 한 seed에 대해 **연합 코드북** 경로만 실행한다 (V5-FedCB-1 / 2a /
2b / 3). V5-FedCB-0 (Gate 1 anchor)는 v02 §B.3에 이미 발표된 CMO row를 그대로
참조하므로 (R0 routing, T2 backbone, α=1.5) 별도 재실행이 필요 없으며,
``02_aggregate.py``가 ``outputs/v02_fl_8020_ratio/seed{S}/W_component_results.json``
의 ``PAPE-aggressive.cells.V0``을 직접 읽어 V5-FedCB-0 row를 구성한다 — 이
파일은 그 anchor를 *재현*하지 않는다.

연합 경로는 09_fix_rerun의 FedAvg-NBEATSxAux 최종 가중치 (백본 + aux head 모두
federated)를 로드하고, ``src/fl/codebook_fl.py``의 3-단계 helper로 *계층적*
단일-shot 연합 KMeans 코드북을 만든다. 그 위에 cluster-mean offset 만으로 cold
보정 (Phase C, ``--alpha`` 로 강도 조절).

Per-seed argparse — 멀티시드 sweep ({42, 123, 7})은 외부 launcher가 ``--seed S``로
세 번 호출 (memory: feedback_argparse_per_seed). 결과는 ``result.json`` +
``codebook.npz``로 저장. MLflow 사용 안 함 (이 repo의 컨벤션은 print + JSON).
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

from config import HORIZON, OUTPUT_DIR, RANDOM_SEED
from dataloader.splits import load_v02_split
from eval.cold_helpers import gather_cold, metrics_z_to_kw, route_R1
from fl.base import DEVICE, build_clients
from fl.codebook_fl import (
    federated_residual_offsets,
    local_codebook_step,
    merge_local_codebooks,
)
from models.nbeatsx_aux import NBEATSxAux

V04_FIX_ROOT = OUTPUT_DIR / "v04_full_baseline_comparison" / "09_fix_rerun"
V05_OUT_ROOT = OUTPUT_DIR / "v05_fedcb_codebook"


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


# ----------------------------------------------------------------------------
# Federated codebook fit (V5-FedCB-1, 2a, 2b, 3).
# ----------------------------------------------------------------------------


def _fit_federated(
    train_apts: list[str],
    model: NBEATSxAux,
    *,
    K_local: int,
    M_global: int,
    seed: int,
    batch_size: int,
    stride: int,
) -> dict:
    """Hierarchical 2-stage federated KMeans + federated residual aggregation.

    Returns ``{codebook, offsets, vq_diagnostics, n_train_windows_total,
    n_train_clients}``.
    """
    clients = build_clients(train_apts)
    if not clients:
        raise RuntimeError("v05 federated: no train clients (all apts missing?)")

    # Stage 1 — local KMeans on each client's h_g (raw h_g never leaves client).
    packets = []
    n_train_windows_total = 0
    for ci, client in enumerate(clients):
        pkt = local_codebook_step(
            model, client, K_local=K_local, seed=seed,
            batch_size=batch_size, stride=stride,
        )
        packets.append(pkt)
        n_train_windows_total += int(pkt["h_g"].shape[0])
        if (ci + 1) % 20 == 0 or (ci + 1) == len(clients):
            print(f"[v05 federated] Stage 1: {ci + 1}/{len(clients)} clients, "
                  f"K_local_i={pkt['K_local_i']}, N_i={pkt['h_g'].shape[0]}")

    # Stage 2 — server merge (sample-weight = local cluster counts).
    merge = merge_local_codebooks(packets, M_global=M_global, seed=seed)
    print(f"[v05 federated] Stage 2: util={merge['utilization']:.3f}  "
          f"ppl={merge['perplexity']:.2f}  k_min={merge['k_min']}  k_max={merge['k_max']}  "
          f"n_empty={merge['n_empty_clusters']}  stage2_inertia={merge['stage2_inertia']:.1f}")

    # Stage 3 — federated residual offsets.
    offsets = federated_residual_offsets(packets, merge["codebook"])

    return {
        "codebook": merge["codebook"],
        "offsets": offsets,
        "n_train_windows_total": int(n_train_windows_total),
        "n_train_clients": len(clients),
        "K_local_i_per_client": [int(p["K_local_i"]) for p in packets],
        "vq_diagnostics": {
            "utilization": float(merge["utilization"]),
            "perplexity": float(merge["perplexity"]),
            "k_min": int(merge["k_min"]),
            "k_max": int(merge["k_max"]),
            "n_empty_clusters": int(merge["n_empty_clusters"]),
            "stage1_mean_inertia": float(merge["stage1_mean_inertia"]),
            "stage2_inertia": float(merge["stage2_inertia"]),
        },
    }


# ----------------------------------------------------------------------------
# Cold inference (CMO-only).
# ----------------------------------------------------------------------------


def _cold_eval_cmo(
    cold_apts: list[str],
    model: NBEATSxAux,
    codebook: np.ndarray,
    offsets: np.ndarray,
    *,
    alpha: float,
    batch_size: int,
    stride: int,
) -> dict:
    """Forward cold apts through the frozen backbone, route via h_g 1-NN, apply CMO.

    Returns ``(fl_only_metrics, with_codebook_metrics, n_cold_windows, n_cold_apts,
    cold_cluster_diagnostics)``. The aux head's ``pred_amp`` / ``pred_hr`` are
    intentionally ignored — v05 drops the Gaussian template (CMO-only).
    """
    co = gather_cold(
        cold_apts, model, batch=batch_size, stride=stride, verbose_skips=False
    )
    fl_only = metrics_z_to_kw(co["y_true_z"], co["y_hat_z"], co["mean"], co["std"])
    fl_only["n_cold_windows"] = int(co["y_true_z"].shape[0])
    fl_only["n_cold_apts"] = int(len(np.unique(co["apt"])))

    cold_cluster = route_R1(co["h_g"], codebook)            # (N,)
    cluster_offset = offsets[cold_cluster]                  # (N, 24)
    corrected_z = (co["y_hat_z"] + alpha * cluster_offset).astype(np.float32)
    cb = metrics_z_to_kw(co["y_true_z"], corrected_z, co["mean"], co["std"])

    # Routing diagnostics — which Stage-2 clusters cold windows landed in.
    M = int(codebook.shape[0])
    cold_bincount = np.bincount(cold_cluster, minlength=M).astype(int)
    cold_diag = {
        "n_clusters_used_on_cold": int((cold_bincount > 0).sum()),
        "cold_cluster_max_share": float(cold_bincount.max() / max(cold_bincount.sum(), 1)),
        "cold_cluster_top_k_share": float(
            np.sort(cold_bincount)[::-1][:5].sum() / max(cold_bincount.sum(), 1)
        ),
    }

    return {
        "fl_only": fl_only,
        "with_codebook_cmo": {
            "alpha": float(alpha),
            "metrics": cb,
        },
        "cold_diag": cold_diag,
    }


# ----------------------------------------------------------------------------
# Main entry point.
# ----------------------------------------------------------------------------


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    """Resolve (backbone_ckpt, output_dir, algorithm_label).

    The algorithm label is the cell name used downstream by 02_aggregate.py.
    Backbone is always the FedAvg-NBEATSxAux final state from v04 09_fix_rerun
    (federated backbone + aux head); v05 never trains a new backbone.
    """
    ckpt = (
        V04_FIX_ROOT / f"seed{args.seed}" / "fedavg_nbeatsx_aux"
        / "final_state_dict.pt"
    )
    # V5-FedCB-3 (α sweep) namespaces α into the cell name.
    if abs(args.alpha - 1.0) < 1e-12:
        cell = f"fedcb_K{args.K_local}"
        algo = f"fedcb_K{args.K_local}"
    else:
        cell = f"fedcb_K{args.K_local}_alpha{args.alpha}"
        algo = f"fedcb_K{args.K_local}_alpha{args.alpha}"
    out_dir = V05_OUT_ROOT / f"seed{args.seed}" / cell
    return ckpt, out_dir, algo


def _communication_bytes(
    *,
    n_train_clients: int,
    K_local: int,
    M_global: int,
    horizon: int,
) -> dict[str, int]:
    """Communication accounting (bytes), matching plan §4.7 formulas.

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
        description="v05 FedCB — federated codebook + CMO-only correction (per-seed). "
                    "V5-FedCB-0 (Gate 1 anchor) is data-only and pulled from v02 by "
                    "02_aggregate.py; this driver only handles the federated cells."
    )
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument(
        "--K_local", type=int, default=4,
        help="Stage-1 cluster count per client.",
    )
    ap.add_argument(
        "--alpha", type=float, default=1.0,
        help="CMO correction strength (Phase C). α=1.0 is the default V5-FedCB-1; "
             "the V5-FedCB-3 α sweep uses 0.5 / 1.5 / 2.0.",
    )
    ap.add_argument("--M", type=int, default=32, help="Global codebook size.")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=HORIZON,
                    help="Codebook fit / cold inference stride; v01/v02 default = 24.")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    sp = load_v02_split(args.seed)
    train_apts, cold_apts = sp["train"], sp["cold"]

    ckpt, out_dir, algo_label = _resolve_paths(args)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"v05 backbone artefact missing: {ckpt}. "
            f"Run the upstream training script for seed={args.seed} first."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v05 federated] seed={args.seed}  algo={algo_label}  "
          f"K_local={args.K_local}  M={args.M}  alpha={args.alpha}  "
          f"batch={args.batch_size}")
    print(f"[v05 federated] backbone_source={ckpt}")
    gpu_start = _gpu_snapshot()
    print(f"[v05 federated] GPU @start: {gpu_start}")

    # Load backbone (FedAvg-NBEATSxAux from v04 09_fix_rerun).
    model = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
    model.load_state_dict(
        torch.load(ckpt, map_location="cpu", weights_only=False), strict=True
    )

    # Phase B' — federated codebook fit.
    t_b = time.time()
    fit = _fit_federated(
        train_apts, model,
        K_local=args.K_local, M_global=args.M, seed=args.seed,
        batch_size=args.batch_size, stride=args.stride,
    )
    n_train_clients = int(fit["n_train_clients"])
    K_local_per_client = fit["K_local_i_per_client"]
    elapsed_b = time.time() - t_b
    print(f"[v05 federated] Phase B' done in {elapsed_b:.1f}s")

    # Phase C — cold inference (CMO-only).
    t_c = time.time()
    cold = _cold_eval_cmo(
        cold_apts, model,
        codebook=fit["codebook"], offsets=fit["offsets"],
        alpha=float(args.alpha),
        batch_size=args.batch_size, stride=args.stride,
    )
    elapsed_c = time.time() - t_c
    print(f"[v05 federated] fl_only  : PAPE={cold['fl_only']['pape']:.2f}  "
          f"HR@1={cold['fl_only']['hr@1']:.1f}  HR@2={cold['fl_only']['hr@2']:.1f}")
    print(f"[v05 federated] CMO α={args.alpha}: PAPE="
          f"{cold['with_codebook_cmo']['metrics']['pape']:.2f}  "
          f"HR@1={cold['with_codebook_cmo']['metrics']['hr@1']:.1f}  "
          f"HR@2={cold['with_codebook_cmo']['metrics']['hr@2']:.1f}")
    print(f"[v05 federated] Phase C done in {elapsed_c:.1f}s")

    elapsed_total = elapsed_b + elapsed_c

    # Communication accounting (per the plan).
    comm = _communication_bytes(
        n_train_clients=n_train_clients,
        K_local=int(args.K_local),
        M_global=int(args.M),
        horizon=HORIZON,
    )

    # Result JSON — schema mirrors plan §Outputs.
    result = {
        "algorithm": algo_label,
        "mode": "federated",
        "seed": int(args.seed),
        "backbone_source": str(ckpt),
        "K_local": int(args.K_local),
        "M_global": int(args.M),
        "alpha": float(args.alpha),
        "config": {
            "mode": "federated",
            "K_local": int(args.K_local),
            "M_global": int(args.M),
            "alpha": float(args.alpha),
            "batch_size": int(args.batch_size),
            "stride": int(args.stride),
            "use_amp": bool(DEVICE.type == "cuda"),
        },
        "fl_only": cold["fl_only"],
        "with_codebook_cmo": cold["with_codebook_cmo"],
        "vq_diagnostics": fit["vq_diagnostics"],
        "communication_bytes": comm,
        "n_train_clients": int(n_train_clients),
        "n_train_windows_total": int(fit["n_train_windows_total"]),
        "n_cold_windows": int(cold["fl_only"]["n_cold_windows"]),
        "n_cold_apts": int(cold["fl_only"]["n_cold_apts"]),
        "cold_routing_diagnostics": cold["cold_diag"],
        "elapsed_seconds": {
            "phase_b": float(elapsed_b),
            "phase_c": float(elapsed_c),
            "total": float(elapsed_total),
        },
        "gpu_at_start": gpu_start,
        "gpu_at_end": _gpu_snapshot(),
        "K_local_i_per_client": K_local_per_client,
    }

    with open(out_dir / "result.json", "w") as fh:
        json.dump(result, fh, indent=2)

    np.savez(
        out_dir / "codebook.npz",
        codebook=fit["codebook"].astype(np.float32),
        offsets=fit["offsets"].astype(np.float32),
    )
    print(f"[v05 federated] saved -> {out_dir}")
    print(f"[v05 federated] total elapsed: {elapsed_total:.0f}s")


if __name__ == "__main__":
    main()
