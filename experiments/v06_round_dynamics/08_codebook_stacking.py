"""V6 Phase 2 — codebook terminal-stacking add-on (per-seed × per-cell driver).

(한글 요약)
v06 Phase 1 (round-level FL training dynamics) 가 30종 cell × 3 seeds = 90
backbone artefact 를 이미 ``outputs/v06_round_dynamics/seed{S}/{cell}/final_state_dict.pt``
로 남겼다. 이 스크립트는 그 backbone 들을 **freeze 한 채** v01-v05 의 Peak-VQ
codebook 을 *post-hoc stacking* 해서 per-client TEST split (20%) 에 대한
codebook lift 를 측정한다.

Federation contract (v05 FedCB 와 일치):
    - Centralised cell (V6-Dyn-A_centralised*) → pooled KMeans (모든 가구 train h_g
      합쳐 단일 KMeans++).
    - FL cell (V6-Dyn-B-*)                     → 2-stage hierarchical *federated*
      KMeans (Stage 1 local KMeans → Stage 2 server merge → Stage 3 federated
      residual offsets). raw h_g 가 가구 밖으로 나가지 않는다.

Correction = CMO-only (Cluster-Mean Offset, Gaussian template α_w1 dropped):
    ŷ_corr = ŷ_base + α_v0 · o_{c*}    (α_v0 default = 1.0)

Output (per seed × per cell):
    ``outputs/v06_round_dynamics/seed{S}/{cell}/codebook_lift.json``

Per-seed argparse — multi-seed × multi-cell sweep 은 사용자 launcher 가 30 cells
× 3 seeds = 90 회 호출 (memory: feedback_argparse_per_seed). 이 스크립트는 한
번에 single ``--seed S`` × single ``--cell C`` 만 처리한다.

CLAUDE.md 준수:
    - codebook hyperparameters M=32, K_local=2, stride=24 (v01-v05/FedCB 일치).
    - 7-axis metric 정의 변경 금지 (utils.metrics 그대로 사용).
    - operating point α_v0=1.0 default — cold-side α 재튜닝 금지.
    - frozen backbone, ``strict=True`` checkpoint load.
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
import torch
from sklearn.cluster import KMeans

from config import HORIZON, OUTPUT_DIR, RANDOM_SEED
from dataloader.per_client_split import build_per_client_splits
from fl.base import DEVICE
from fl.codebook_fl import (
    _route_h_g_to_codebook,
    federated_residual_offsets,
    local_codebook_step_from_splits,
    merge_local_codebooks,
)
from fl.fedavg_aux import init_backbone_aux
from utils.metrics import compute_hr, compute_mae, compute_mse, compute_pape


# Cell-name allowlist:
# - v06 default (λ=0.3) + MAEonly (λ=0)        → "" / "-MAEonly"
# - v07-A new lambdas (λ ∈ {0.05, 0.1, 0.2})   → "-aux{V}"
# - v07-A high-λ extension (λ ∈ {0.5, 0.7, 1}) → "-aux{V}"
# 6 algorithms × 8 lambda-suffixes = 48 valid cells.
_FL_BASES = ["FedAvg", "FedProx", "FedRep", "Ditto", "FedProto"]
_LAMBDA_SUFFIXES = (
    "", "-MAEonly", "-aux0.05", "-aux0.1", "-aux0.2",
    "-aux0.5", "-aux0.7", "-aux1",
)
_CENTRAL_CELLS = [f"V6-Dyn-A_centralised{suf}" for suf in _LAMBDA_SUFFIXES]
_FL_CELLS = [f"V6-Dyn-B-{b}{suf}" for b in _FL_BASES for suf in _LAMBDA_SUFFIXES]
_VALID_CELLS = _CENTRAL_CELLS + _FL_CELLS


def _is_centralised(cell: str) -> bool:
    return cell.startswith("V6-Dyn-A_centralised")


# ============================================================================
# h_g extraction over all clients (used for both protocols)
# ============================================================================


def _build_train_packets(
    model: torch.nn.Module,
    splits: dict[str, dict],
    *,
    K_local: int,
    seed: int,
    batch_size: int,
    use_amp: bool,
) -> tuple[list[dict], list[str]]:
    """Run Stage-1 ``local_codebook_step_from_splits`` on every apt's train
    windows. Returns ``(packets, apt_order)`` aligned by index.

    Empty packets (apts with zero train windows after the carve) are still
    appended so downstream diagnostics can report ``K_local_i_per_client``
    truthfully; ``merge_local_codebooks`` / ``federated_residual_offsets``
    already skip empty packets internally.
    """
    packets: list[dict] = []
    apt_order: list[str] = []
    for apt, sp in splits.items():
        pkt = local_codebook_step_from_splits(
            model,
            sp["train_x"], sp["train_y"],
            K_local=K_local, seed=seed,
            batch_size=batch_size, use_amp=use_amp,
        )
        packets.append(pkt)
        apt_order.append(apt)
    return packets, apt_order


# ============================================================================
# Centralised codebook (pooled KMeans on all clients' train h_g)
# ============================================================================


def _fit_codebook_centralised(
    packets: list[dict], M: int, seed: int
) -> tuple[np.ndarray, dict]:
    """Pool every client's ``h_g`` (raw windows) and run a single KMeans++
    of size ``M``. Mirrors the conference Phase B *centralised* path that
    v06 Phase 2 uses for the V6-Dyn-A cell only.

    Returns (codebook, diag) where diag has the same diagnostic schema as
    ``merge_local_codebooks`` so the result.json structure is consistent.
    """
    H = np.concatenate(
        [p["h_g"] for p in packets if p["h_g"].shape[0] > 0],
        axis=0,
    ).astype(np.float32)
    if H.shape[0] < M:
        raise ValueError(
            f"_fit_codebook_centralised: only {H.shape[0]} pooled h_g rows "
            f"vs M={M}; reduce M."
        )
    km = KMeans(
        n_clusters=int(M), init="k-means++", n_init=10, random_state=int(seed)
    ).fit(H)
    codebook = km.cluster_centers_.astype(np.float32)
    labels = km.labels_.astype(np.int64)
    counts = np.bincount(labels, minlength=int(M)).astype(np.int64)
    nonzero = counts[counts > 0]
    util = float((counts > 0).sum()) / float(M)
    total = float(counts.sum())
    if total > 0:
        probs = counts / total
        nz = probs > 0
        entropy = float(-(probs[nz] * np.log(probs[nz])).sum())
        perplexity = float(np.exp(entropy))
    else:
        perplexity = 0.0
    diag = {
        "utilization": util,
        "perplexity": perplexity,
        "k_min": int(nonzero.min()) if nonzero.size else 0,
        "k_max": int(counts.max()),
        "n_empty_clusters": int((counts == 0).sum()),
        "stage1_mean_inertia": 0.0,   # not applicable for centralised
        "stage2_inertia": float(km.inertia_),
    }
    return codebook, diag


def _residual_offsets_centralised(
    packets: list[dict], codebook: np.ndarray
) -> np.ndarray:
    """Pooled cluster-mean residual: ``o_c = mean_{n: c*[n]=c} (y_true_z - y_hat_z)``
    over ALL train windows pooled across clients. Same final answer as
    ``federated_residual_offsets`` would give if all clients were merged
    pre-aggregation, but computed centrally for the V6-Dyn-A protocol.
    """
    M = int(codebook.shape[0])
    H = HORIZON
    sum_resid = np.zeros((M, H), dtype=np.float64)
    sum_count = np.zeros((M,), dtype=np.int64)
    for p in packets:
        if p["h_g"].shape[0] == 0:
            continue
        idx = _route_h_g_to_codebook(p["h_g"], codebook)
        resid = (p["y_true_z"] - p["y_hat_z"]).astype(np.float64)
        np.add.at(sum_resid, idx, resid)
        sum_count += np.bincount(idx, minlength=M)
    offsets = np.zeros((M, H), dtype=np.float32)
    nz = sum_count > 0
    offsets[nz] = (sum_resid[nz] / sum_count[nz, None]).astype(np.float32)
    return offsets


# ============================================================================
# Test-split forward + per-apt metrics
# ============================================================================


def _forward_test_h_g(
    model: torch.nn.Module,
    test_x: np.ndarray,
    *,
    batch_size: int,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward one apt's test windows through the frozen backbone.

    Returns (h_g_cold, y_hat_base_z). Symmetric with the train-side
    ``_extract_h_g_from_windows`` but does not need ``y_true_z`` (caller
    already has ``test_y`` from the splits dict).
    """
    n = int(test_x.shape[0])
    if n == 0:
        return (
            np.zeros((0, 64), dtype=np.float32),
            np.zeros((0, HORIZON), dtype=np.float32),
        )
    from fl.codebook_fl import _NullCtx  # local import keeps file structure
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_amp and DEVICE.type == "cuda")
        else _NullCtx()
    )
    model.eval()
    h_chunks, yhat_chunks = [], []
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(test_x[i : i + batch_size]).to(DEVICE, non_blocking=True)
        with torch.no_grad(), amp_ctx:
            ret = model(xb)
        y_hat = ret[0]
        hiddens = ret[1]
        h_chunks.append(hiddens["h_generic"].float().cpu().numpy())
        yhat_chunks.append(y_hat.float().cpu().numpy())
    return (
        np.concatenate(h_chunks, axis=0).astype(np.float32),
        np.concatenate(yhat_chunks, axis=0).astype(np.float32),
    )


def _per_apt_metrics_kw(
    y_true_kw: np.ndarray, y_pred_kw: np.ndarray
) -> dict[str, float]:
    """Per-apt 5-axis metrics in kW (PAPE / HR@1 / HR@2 / MAE / MSE(kW²))."""
    return {
        "pape":     float(compute_pape(y_true_kw, y_pred_kw)),
        "hr@1":     float(compute_hr(y_true_kw, y_pred_kw, tol=1)),
        "hr@2":     float(compute_hr(y_true_kw, y_pred_kw, tol=2)),
        "mae":      float(compute_mae(y_true_kw, y_pred_kw)),
        "mse_kw2":  float(compute_mse(y_true_kw, y_pred_kw)),
    }


def _aggregate_across_clients(per_apt: list[dict[str, float]]) -> dict[str, float]:
    """RoundLogger-style across-apt aggregation: simple mean / sample-std
    (ddof=1) over apt count. Each apt counts once."""
    if not per_apt:
        return {
            "pape_mean": float("nan"), "pape_std_across_clients": float("nan"),
            "hr@1_mean": float("nan"), "hr@2_mean": float("nan"),
            "mae_mean": float("nan"), "mse_kw2_mean": float("nan"),
            "n_clients": 0,
        }
    pape = np.asarray([m["pape"]    for m in per_apt], dtype=np.float64)
    hr1  = np.asarray([m["hr@1"]    for m in per_apt], dtype=np.float64)
    hr2  = np.asarray([m["hr@2"]    for m in per_apt], dtype=np.float64)
    mae  = np.asarray([m["mae"]     for m in per_apt], dtype=np.float64)
    mse  = np.asarray([m["mse_kw2"] for m in per_apt], dtype=np.float64)
    return {
        "pape_mean":               float(np.mean(pape)),
        "pape_std_across_clients": float(np.std(pape, ddof=1)) if pape.size > 1 else 0.0,
        "hr@1_mean":               float(np.mean(hr1)),
        "hr@2_mean":               float(np.mean(hr2)),
        "mae_mean":                float(np.mean(mae)),
        "mse_kw2_mean":            float(np.mean(mse)),
        "n_clients":               int(pape.size),
    }


# ============================================================================
# Main per-seed × per-cell driver
# ============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "v06 Phase 2 — post-hoc Peak-VQ codebook stacking on a v06 Phase 1 "
            "backbone. CMO-only correction (no Gaussian template). Single seed × "
            "single cell per invocation; outer launcher loops over 30 cells × {42,123,7}."
        )
    )
    ap.add_argument("--seed", type=int, default=RANDOM_SEED,
                    help="Single seed (memory: feedback_argparse_per_seed).")
    ap.add_argument("--cell", type=str, required=True, choices=_VALID_CELLS,
                    help="One of the 30 v06 Phase-1 cells (the backbone source).")
    ap.add_argument("--M", type=int, default=32,
                    help="Codebook size (CLAUDE.md fixed = 32).")
    ap.add_argument("--K_local", type=int, default=2,
                    help="Stage-1 client cluster count for FL protocol "
                         "(CLAUDE.md fixed = 2; ignored for centralised).")
    ap.add_argument("--alpha_v0", type=float, default=1.0,
                    help="CMO correction strength (default = 1.0 — operating "
                         "point carry-over; do NOT re-tune on test).")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--no_amp", action="store_true",
                    help="Disable bf16 autocast (auto-disabled on CPU).")
    ap.add_argument("--ablation_suffix", type=str, default="",
                    help="Optional suffix appended to the output filename "
                         "(e.g. '_alpha1.5' or '_K4') so an ablation does not "
                         "overwrite the default codebook_lift.json. Empty "
                         "string = the canonical default file path.")
    ap.add_argument("--output_namespace", type=str, default="v06_round_dynamics",
                    help="Top-level output sub-directory under outputs/ (default: v06_round_dynamics; "
                         "v07 launcher overrides to v07_loss_budget_sweeps to read v07 backbones).")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_amp = not args.no_amp

    cell_dir = OUTPUT_DIR / args.output_namespace / f"seed{args.seed}" / args.cell
    backbone_ckpt = cell_dir / "final_state_dict.pt"
    if not backbone_ckpt.exists():
        raise FileNotFoundError(
            f"v06 Phase 2 requires a Phase 1 backbone artefact at {backbone_ckpt}. "
            f"Run experiments/v06_round_dynamics/01_centralised.py or 02_fl_dynamics.py "
            f"for seed={args.seed}, cell={args.cell} first (does NOT silently retrain)."
        )

    protocol = "centralised" if _is_centralised(args.cell) else "fl"

    print(f"[{args.cell} / phase2] seed={args.seed}  protocol={protocol}  "
          f"M={args.M}  K_local={args.K_local}  alpha_v0={args.alpha_v0}  amp={use_amp}")
    print(f"[{args.cell} / phase2] backbone_source={backbone_ckpt}")

    t0 = time.time()

    # 1) Load frozen backbone (NBEATSxAux full state, strict=True).
    model = init_backbone_aux(seed=args.seed)
    model.load_state_dict(
        torch.load(backbone_ckpt, map_location="cpu", weights_only=False),
        strict=True,
    )
    model = model.to(DEVICE).eval()

    # 2) Per-client 70/10/20 splits (cached on disk; same cache as Phase 1).
    print(f"[{args.cell} / phase2] building per-client 70/10/20 splits ...")
    splits = build_per_client_splits(seed=args.seed)
    print(f"[{args.cell} / phase2] {len(splits)} apartments retained.")

    # 3) Stage 1 — extract h_g + (y_hat_z, y_true_z) on every apt's train
    #    windows (frozen forward; raw h_g stays in the in-memory packet but,
    #    in the FL protocol, would never leave the client over the wire).
    print(f"[{args.cell} / phase2] Stage 1: extracting h_g on train windows ...")
    packets, apt_order = _build_train_packets(
        model, splits,
        K_local=args.K_local, seed=args.seed,
        batch_size=args.batch_size, use_amp=use_amp,
    )
    n_train_windows_total = sum(int(p["h_g"].shape[0]) for p in packets)
    n_clients = len(splits)
    print(f"[{args.cell} / phase2] Stage 1 done: {n_train_windows_total} train "
          f"windows pooled (or per-client for FL).")

    # 4) Stage 2 — codebook fit (centralised pooled vs federated 2-stage).
    if protocol == "centralised":
        codebook, diag = _fit_codebook_centralised(packets, M=args.M, seed=args.seed)
        offsets = _residual_offsets_centralised(packets, codebook)
        print(f"[{args.cell} / phase2] Stage 2 (centralised pooled KMeans): "
              f"util={diag['utilization']:.3f}  ppl={diag['perplexity']:.2f}  "
              f"k_min={diag['k_min']}  k_max={diag['k_max']}  "
              f"n_empty={diag['n_empty_clusters']}  inertia={diag['stage2_inertia']:.1f}")
    else:
        merge = merge_local_codebooks(packets, M_global=args.M, seed=args.seed)
        codebook = merge["codebook"]
        offsets = federated_residual_offsets(packets, codebook)
        diag = {
            "utilization":         float(merge["utilization"]),
            "perplexity":          float(merge["perplexity"]),
            "k_min":               int(merge["k_min"]),
            "k_max":               int(merge["k_max"]),
            "n_empty_clusters":    int(merge["n_empty_clusters"]),
            "stage1_mean_inertia": float(merge["stage1_mean_inertia"]),
            "stage2_inertia":      float(merge["stage2_inertia"]),
        }
        print(f"[{args.cell} / phase2] Stage 2 (federated 2-stage): "
              f"util={diag['utilization']:.3f}  ppl={diag['perplexity']:.2f}  "
              f"k_min={diag['k_min']}  k_max={diag['k_max']}  "
              f"n_empty={diag['n_empty_clusters']}  "
              f"stage1_mean_inertia={diag['stage1_mean_inertia']:.1f}  "
              f"stage2_inertia={diag['stage2_inertia']:.1f}")

    # 5) Test evaluation — per apt, forward test_x, route, correct, denorm to kW.
    print(f"[{args.cell} / phase2] Test evaluation on {n_clients} clients ...")
    per_apt_before: list[dict[str, float]] = []
    per_apt_after:  list[dict[str, float]] = []
    n_test_windows_total = 0
    cluster_assignment_counts = np.zeros((args.M,), dtype=np.int64)
    for apt in apt_order:
        sp = splits[apt]
        x = sp["test_x"]; y = sp["test_y"]
        if x.shape[0] == 0:
            continue
        m_, s_ = float(sp["mean"]), float(sp["std"])
        h_g_cold, y_hat_base_z = _forward_test_h_g(
            model, x, batch_size=args.batch_size, use_amp=use_amp
        )
        c_idx = _route_h_g_to_codebook(h_g_cold, codebook)            # (N,)
        cluster_offset = offsets[c_idx]                                # (N, 24)
        y_hat_corr_z = (
            y_hat_base_z + float(args.alpha_v0) * cluster_offset
        ).astype(np.float32)

        y_true_kw = (y * s_ + m_).astype(np.float32)
        y_hat_base_kw = (y_hat_base_z * s_ + m_).astype(np.float32)
        y_hat_corr_kw = (y_hat_corr_z * s_ + m_).astype(np.float32)

        per_apt_before.append(_per_apt_metrics_kw(y_true_kw, y_hat_base_kw))
        per_apt_after.append(_per_apt_metrics_kw(y_true_kw, y_hat_corr_kw))
        cluster_assignment_counts += np.bincount(c_idx, minlength=args.M).astype(np.int64)
        n_test_windows_total += int(x.shape[0])

    test_before = _aggregate_across_clients(per_apt_before)
    test_after  = _aggregate_across_clients(per_apt_after)
    lift = {
        "pape_delta":     float(test_after["pape_mean"]    - test_before["pape_mean"]),
        "hr@1_delta":     float(test_after["hr@1_mean"]    - test_before["hr@1_mean"]),
        "hr@2_delta":     float(test_after["hr@2_mean"]    - test_before["hr@2_mean"]),
        "mae_delta":      float(test_after["mae_mean"]     - test_before["mae_mean"]),
        "mse_kw2_delta":  float(test_after["mse_kw2_mean"] - test_before["mse_kw2_mean"]),
    }

    # Cluster diagnostics on the test-side routing.
    test_used = int((cluster_assignment_counts > 0).sum())
    cluster_total = int(cluster_assignment_counts.sum())
    test_top_share = float(
        cluster_assignment_counts.max() / max(cluster_total, 1)
    )
    test_top5_share = float(
        np.sort(cluster_assignment_counts)[::-1][:5].sum() / max(cluster_total, 1)
    )

    elapsed = time.time() - t0

    print(f"[{args.cell} / phase2] BEFORE  PAPE={test_before['pape_mean']:.2f}±"
          f"{test_before['pape_std_across_clients']:.2f}  "
          f"HR@1={test_before['hr@1_mean']:.2f}  "
          f"HR@2={test_before['hr@2_mean']:.2f}  "
          f"MAE={test_before['mae_mean']:.4f}  "
          f"MSE={test_before['mse_kw2_mean']:.4f}")
    print(f"[{args.cell} / phase2] AFTER   PAPE={test_after['pape_mean']:.2f}±"
          f"{test_after['pape_std_across_clients']:.2f}  "
          f"HR@1={test_after['hr@1_mean']:.2f}  "
          f"HR@2={test_after['hr@2_mean']:.2f}  "
          f"MAE={test_after['mae_mean']:.4f}  "
          f"MSE={test_after['mse_kw2_mean']:.4f}")
    print(f"[{args.cell} / phase2] LIFT    ΔPAPE={lift['pape_delta']:+.2f}  "
          f"ΔHR@1={lift['hr@1_delta']:+.2f}  "
          f"ΔHR@2={lift['hr@2_delta']:+.2f}  "
          f"ΔMAE={lift['mae_delta']:+.4f}  "
          f"ΔMSE={lift['mse_kw2_delta']:+.4f}")
    print(f"[{args.cell} / phase2] elapsed={elapsed:.1f}s")

    # 6) Persist result.
    out_path = cell_dir / f"codebook_lift{args.ablation_suffix}.json"
    payload = {
        "cell": args.cell,
        "seed": int(args.seed),
        "protocol": protocol,
        "M": int(args.M),
        "K_local": int(args.K_local),
        "alpha_v0": float(args.alpha_v0),
        "n_clients": int(n_clients),
        "n_train_windows_total": int(n_train_windows_total),
        "n_test_windows_total":  int(n_test_windows_total),
        "K_local_i_per_client":  [int(p["K_local_i"]) for p in packets],
        "codebook_diag": {
            "utilization":         diag["utilization"],
            "perplexity":          diag["perplexity"],
            "k_min":               int(diag["k_min"]),
            "k_max":               int(diag["k_max"]),
            "n_empty_clusters":    int(diag["n_empty_clusters"]),
            "stage1_mean_inertia": diag["stage1_mean_inertia"],
            "stage2_inertia":      diag["stage2_inertia"],
        },
        "test_routing_diag": {
            "n_clusters_used_on_test": test_used,
            "test_cluster_max_share":  test_top_share,
            "test_cluster_top5_share": test_top5_share,
        },
        "test_before": test_before,
        "test_after":  test_after,
        "lift":        lift,
        "backbone_source": str(backbone_ckpt),
        "use_amp": bool(use_amp),
        "elapsed_seconds": float(elapsed),
        "comment": (
            "v06 Phase 2: post-hoc Peak-VQ codebook stacking on a v06 Phase 1 "
            "backbone. Centralised cells use pooled KMeans; FL cells use 2-stage "
            "hierarchical federated KMeans (src/fl/codebook_fl.py). Correction "
            "is CMO-only (Gaussian template α_w1 = 0). Test split = each apt's "
            "20% test windows from per_client_split.pkl — the natural unseen "
            "future windows for the v06 protocol."
        ),
    }
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[{args.cell} / phase2] saved -> {out_path}")


if __name__ == "__main__":
    main()
