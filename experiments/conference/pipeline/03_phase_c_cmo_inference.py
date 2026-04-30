"""Phase C — cold inference with CMO-only correction (per-seed driver).

(한글 요약)
KIIE conference 발표 (``papers/conference_draft/presentation.md``)의 §3.3 마지막
"Cluster-wise Forecast Correction" 단락과 §"Codebook Correction Module 효과 측정"
표 (lines 211-218)의 *Backbone + Codebook Correction Module* 행에 해당하는
스크립트. Phase A의 federated 백본과 Phase B의 federated codebook을 모두 freeze한
채 cold 가구의 forecast를 cluster-mean offset (CMO)로만 보정한다 — Gaussian
template α_w1 항은 dropped (CMO-only, presentation.md §3.3 *(α=1.0 default)*).

Compute & save:
    - ``fl_only`` 행 (= Backbone, no correction): cold 백본 출력 그대로.
    - ``with_codebook_cmo`` 행 (= Backbone + Codebook): ``y_hat_z + α · o[c*]``.
    - ``cold_arrays.npz``: y_true_z / y_hat_z / corrected_z / mean / std / apt /
      cold_cluster — ablation 스크립트가 후속해서 동일 cold 윈도우 위에서 PAPE /
      HR@k / kW²-MSE를 *재계산*할 수 있게 raw 배열을 저장 (v05의
      ``04_recompute_mse.py``와 같은 패턴).

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
from eval.cold_helpers import gather_cold, metrics_z_to_kw, route_R1
from fl.base import DEVICE
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Conference Phase C — cold inference with CMO-only correction "
            "(α default = 1.0). Loads Phase-A backbone + Phase-B codebook for "
            "the same seed; halts if either is missing (does NOT silently re-fit)."
        )
    )
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="CMO correction strength (presentation.md §3.3 default = 1.0).")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=HORIZON,
                    help="Cold inference stride; v01/v02 default = 24.")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    sp = load_v02_split(args.seed)
    cold_apts = sp["cold"]

    phase_a_dir = CONFERENCE_OUT_ROOT / f"seed{args.seed}" / "phase_a"
    phase_b_dir = CONFERENCE_OUT_ROOT / f"seed{args.seed}" / "phase_b"
    backbone_ckpt = phase_a_dir / "final_state_dict.pt"
    codebook_npz = phase_b_dir / "codebook.npz"
    out_dir = CONFERENCE_OUT_ROOT / f"seed{args.seed}" / "phase_c"

    if not backbone_ckpt.exists():
        raise FileNotFoundError(
            f"Phase C requires a Phase A backbone artefact at {backbone_ckpt}. "
            f"Run experiments/conference/pipeline/01_phase_a_train_backbone.py "
            f"--seed {args.seed} first."
        )
    if not codebook_npz.exists():
        raise FileNotFoundError(
            f"Phase C requires a Phase B codebook artefact at {codebook_npz}. "
            f"Run experiments/conference/pipeline/02_phase_b_fit_federated_codebook.py "
            f"--seed {args.seed} first."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Pull metadata from Phase B for the result.json provenance.
    with open(phase_b_dir / "codebook_meta.json") as fh:
        b_meta = json.load(fh)
    K_local = int(b_meta["K_local"])
    M_global = int(b_meta["M_global"])

    print(
        f"[conference phase_c] seed={args.seed}  alpha={args.alpha}  "
        f"K_local={K_local}  M={M_global}  batch={args.batch_size}  "
        f"stride={args.stride}"
    )
    print(f"[conference phase_c] backbone_source={backbone_ckpt}")
    print(f"[conference phase_c] codebook_source={codebook_npz}")
    gpu_start = _gpu_snapshot()
    print(f"[conference phase_c] GPU @start: {gpu_start}")

    t0 = time.time()

    # Load backbone + codebook.
    model = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
    model.load_state_dict(
        torch.load(backbone_ckpt, map_location="cpu", weights_only=False),
        strict=True,
    )
    cb = np.load(codebook_npz)
    codebook = cb["codebook"].astype(np.float32)
    offsets = cb["offsets"].astype(np.float32)

    # Cold forward pass — forwards every cold apt's train segment, returns the
    # full set of arrays we need (h_g, y_hat_z, y_true_z, mean, std, apt).
    co = gather_cold(
        cold_apts, model,
        batch=args.batch_size, stride=args.stride, verbose_skips=False,
    )

    # fl_only — federated backbone alone, no correction.
    fl_only = metrics_z_to_kw(co["y_true_z"], co["y_hat_z"], co["mean"], co["std"])
    fl_only["n_cold_windows"] = int(co["y_true_z"].shape[0])
    fl_only["n_cold_apts"] = int(len(np.unique(co["apt"])))

    # Route cold windows via h_g 1-NN against the federated codebook (R1).
    cold_cluster = route_R1(co["h_g"], codebook)               # (N,)
    cluster_offset = offsets[cold_cluster]                      # (N, 24)
    corrected_z = (co["y_hat_z"] + float(args.alpha) * cluster_offset).astype(np.float32)
    cb_metrics = metrics_z_to_kw(co["y_true_z"], corrected_z, co["mean"], co["std"])

    # Cold routing diagnostics — same shape as the v05 driver.
    cold_bincount = np.bincount(cold_cluster, minlength=M_global).astype(int)
    cold_diag = {
        "n_clusters_used_on_cold": int((cold_bincount > 0).sum()),
        "cold_cluster_max_share": float(
            cold_bincount.max() / max(cold_bincount.sum(), 1)
        ),
        "cold_cluster_top_k_share": float(
            np.sort(cold_bincount)[::-1][:5].sum() / max(cold_bincount.sum(), 1)
        ),
    }

    print(
        f"[conference phase_c] fl_only          : PAPE={fl_only['pape']:.2f}  "
        f"HR@1={fl_only['hr@1']:.1f}  HR@2={fl_only['hr@2']:.1f}"
    )
    print(
        f"[conference phase_c] with_codebook_cmo: PAPE={cb_metrics['pape']:.2f}  "
        f"HR@1={cb_metrics['hr@1']:.1f}  HR@2={cb_metrics['hr@2']:.1f}  "
        f"(α={args.alpha})"
    )

    elapsed = time.time() - t0
    print(f"[conference phase_c] Phase C done in {elapsed:.1f}s")

    # Save raw cold arrays so the ablation script (and any future MSE / kW²
    # recompute) can re-derive metrics without re-running Phase C — same
    # pattern as experiments/v05_fedcb_codebook/04_recompute_mse.py.
    np.savez_compressed(
        out_dir / "cold_arrays.npz",
        y_true_z=co["y_true_z"].astype(np.float32),
        y_hat_z=co["y_hat_z"].astype(np.float32),
        corrected_z=corrected_z,
        mean=co["mean"].astype(np.float32),
        std=co["std"].astype(np.float32),
        apt=co["apt"].astype(str),
        cold_cluster=cold_cluster.astype(np.int64),
    )

    result = {
        "algorithm": f"fedcb_K{K_local}",
        "mode": "federated",
        "seed": int(args.seed),
        "alpha": float(args.alpha),
        "K_local": int(K_local),
        "M_global": int(M_global),
        "backbone_source": str(backbone_ckpt),
        "codebook_source": str(codebook_npz),
        "fl_only": fl_only,
        "with_codebook_cmo": {
            "alpha": float(args.alpha),
            "metrics": cb_metrics,
        },
        "cold_routing_diagnostics": cold_diag,
        "n_cold_windows": int(fl_only["n_cold_windows"]),
        "n_cold_apts": int(fl_only["n_cold_apts"]),
        "elapsed_seconds": float(elapsed),
        "gpu_at_start": gpu_start,
        "gpu_at_end": _gpu_snapshot(),
        "comment": (
            "Conference Phase C: CMO-only correction (no Gaussian template). "
            "Cold windows routed via h_g 1-NN against the federated codebook; "
            "y_hat_z corrected by alpha · cluster_mean_residual_offset. The two "
            "result blocks (fl_only / with_codebook_cmo) directly populate the "
            "two rows of the §'Codebook Correction Module 효과 측정' ablation "
            "table in presentation.md (lines 211-218)."
        ),
    }
    with open(out_dir / "result.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[conference phase_c] saved -> {out_dir}")


if __name__ == "__main__":
    main()
