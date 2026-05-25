"""V9-RoundCB — federated post-hoc codebook fit at the END of EVERY round.

(한글 요약)
``experiments/v06_round_dynamics/08_codebook_stacking.py`` 는 frozen 된 final
backbone 위에 codebook 을 *1회* fit 한다. 본 driver 는 backbone 이 아직 학습
중인 v06 Phase 1 style FedAvg 루프 위에서 codebook 을 R 회 — 매 라운드 종료
시점마다 — fit 한다. backbone forward 는 **codebook 을 절대 통과시키지 않으며**
(commit_loss / VQ replace 없음), pure 분석 목적이다: backbone 학습 진행에 따라
post-hoc codebook + 그 lift 가 어떻게 진화하는가.

- Backbone : ``src/fl/round_aux.py:run_fl_aux`` (algorithm 기본 ``fedavg``).
              loss = MAE + 0.3 · peak_aux(hr_weight=0.1), AdamW lr=1e-3 wd=1e-5,
              batch=512.
- Codebook : ``src/fl/codebook_fl.py`` 의 ``local_codebook_step_from_splits``
              → ``merge_local_codebooks`` → ``federated_residual_offsets`` 를
              그대로 재사용. 매 라운드 callback 안에서 train 윈도우 forward → 2-stage
              federated KMeans → 가구별 test forward → CMO 보정 → before/after
              7-axis metric 산출 → JSONL 1행.

Cell name : ``V9-RoundCB-{Algo}`` (Algo = FedAvg / FedProx / ...).

Output (``outputs/v09_round_vq_codebook/seed{S}/V9-RoundCB-{Algo}/``)
    ├── codebook_log.jsonl       — 1 row per codebook fit (codebook_period-aware)
    ├── codebook_history.pt      — {rounds, codebook(R',M,D), offsets(R',M,H)}
    ├── final_state_dict.pt
    └── result.json              — terminal summary + lift_trajectory

Per-seed argparse — multi-seed sweep 은 executor 의 몫.
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
from fl.round_aux import run_fl_aux
from utils.metrics import compute_hr, compute_mae, compute_mse, compute_pape


_ALGO_PRETTY = {
    "fedavg":   "FedAvg",
    "fedprox":  "FedProx",
    "fedrep":   "FedRep",
    "ditto":    "Ditto",
    "fedproto": "FedProto",
}


# ============================================================================
# Helpers copied verbatim from
# experiments/v06_round_dynamics/08_codebook_stacking.py — kept inline so that
# this driver has no cross-experiment sys.path dependency.
# ============================================================================


def _forward_test_h_g(
    model: torch.nn.Module,
    test_x: np.ndarray,
    *,
    batch_size: int,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward one apt's test windows through the (currently-trained) backbone.

    Returns (h_g_cold, y_hat_base_z). Verbatim from
    ``08_codebook_stacking.py:_forward_test_h_g``.
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
    """Per-apt 5-axis metrics in kW (PAPE / HR@1 / HR@2 / MAE / MSE(kW²)).

    Verbatim from ``08_codebook_stacking.py``.
    """
    return {
        "pape":     float(compute_pape(y_true_kw, y_pred_kw)),
        "hr@1":     float(compute_hr(y_true_kw, y_pred_kw, tol=1)),
        "hr@2":     float(compute_hr(y_true_kw, y_pred_kw, tol=2)),
        "mae":      float(compute_mae(y_true_kw, y_pred_kw)),
        "mse_kw2":  float(compute_mse(y_true_kw, y_pred_kw)),
    }


def _aggregate_across_clients(per_apt: list[dict[str, float]]) -> dict[str, float]:
    """RoundLogger-style across-apt aggregation.

    Verbatim from ``08_codebook_stacking.py``.
    """
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
# Round-end callback : fit codebook + evaluate test lift
# ============================================================================


def _make_round_end_callback(
    splits: dict[str, dict],
    *,
    M: int,
    K_local: int,
    alpha_v0: float,
    batch_size: int,
    use_amp: bool,
    seed: int,
    codebook_period: int,
    codebook_log_path: Path,
    codebook_history: dict,
    cell_name: str,
):
    """Build the ``on_round_end`` callback that ``run_fl_aux`` will invoke.

    Signature (from ``src/fl/round_aux.py``)::

        on_round_end(round_idx, model, server_state_pre, client_states,
                     comm_stats, wall_seconds, train_stats, epoch_equivalent)

    When ``round_idx % codebook_period == 0`` we:
      1. Stage-1 KMeans on every apt's train windows (frozen current backbone).
      2. Stage-2 weighted KMeans on stacked centroids (federated merge).
      3. Per-cluster mean residuals (federated offsets).
      4. Forward each apt's test windows, route by 1-NN, correct with CMO offsets,
         denorm to kW, 5-axis metrics, across-apt mean.
      5. Append 1 JSONL row + snapshot codebook/offsets in-memory.

    Backbone forward never touches the codebook — this is post-hoc analysis on
    a still-training backbone.
    """
    apt_order = list(splits.keys())

    def _callback(
        round_idx: int,
        model,
        server_state_pre,
        client_states,
        comm_stats,
        wall_seconds,
        train_stats,
        epoch_equivalent,
    ) -> None:
        if (codebook_period > 0) and (round_idx % codebook_period != 0):
            return

        t_cb = time.time()

        # 1) Stage-1 local KMeans on every apt's train windows.
        packets: list[dict] = []
        for apt in apt_order:
            sp = splits[apt]
            pkt = local_codebook_step_from_splits(
                model,
                sp["train_x"], sp["train_y"],
                K_local=int(K_local), seed=int(seed),
                batch_size=int(batch_size), use_amp=bool(use_amp),
            )
            packets.append(pkt)

        # 2) Stage-2 federated merge KMeans.
        merge = merge_local_codebooks(packets, M_global=int(M), seed=int(seed))
        codebook = merge["codebook"]                       # (M, D)
        offsets = federated_residual_offsets(packets, codebook)   # (M, H)

        codebook_diag = {
            "utilization":         float(merge["utilization"]),
            "perplexity":          float(merge["perplexity"]),
            "k_min":               int(merge["k_min"]),
            "k_max":               int(merge["k_max"]),
            "n_empty_clusters":    int(merge["n_empty_clusters"]),
            "stage1_mean_inertia": float(merge["stage1_mean_inertia"]),
            "stage2_inertia":      float(merge["stage2_inertia"]),
        }

        # 3) Test eval per apt — mirror 08_codebook_stacking.py lines 394–442.
        per_apt_before: list[dict[str, float]] = []
        per_apt_after:  list[dict[str, float]] = []
        n_test_windows_total = 0
        cluster_assignment_counts = np.zeros((int(M),), dtype=np.int64)
        for apt in apt_order:
            sp = splits[apt]
            x = sp["test_x"]; y = sp["test_y"]
            if x.shape[0] == 0:
                continue
            m_, s_ = float(sp["mean"]), float(sp["std"])
            h_g_cold, y_hat_base_z = _forward_test_h_g(
                model, x, batch_size=int(batch_size), use_amp=bool(use_amp)
            )
            c_idx = _route_h_g_to_codebook(h_g_cold, codebook)
            cluster_offset = offsets[c_idx]
            y_hat_corr_z = (
                y_hat_base_z + float(alpha_v0) * cluster_offset
            ).astype(np.float32)

            y_true_kw     = (y               * s_ + m_).astype(np.float32)
            y_hat_base_kw = (y_hat_base_z    * s_ + m_).astype(np.float32)
            y_hat_corr_kw = (y_hat_corr_z    * s_ + m_).astype(np.float32)

            per_apt_before.append(_per_apt_metrics_kw(y_true_kw, y_hat_base_kw))
            per_apt_after .append(_per_apt_metrics_kw(y_true_kw, y_hat_corr_kw))
            cluster_assignment_counts += np.bincount(c_idx, minlength=int(M)).astype(np.int64)
            n_test_windows_total += int(x.shape[0])

        test_before = _aggregate_across_clients(per_apt_before)
        test_after  = _aggregate_across_clients(per_apt_after)
        lift = {
            "pape_delta":    float(test_after["pape_mean"]    - test_before["pape_mean"]),
            "hr@1_delta":    float(test_after["hr@1_mean"]    - test_before["hr@1_mean"]),
            "hr@2_delta":    float(test_after["hr@2_mean"]    - test_before["hr@2_mean"]),
            "mae_delta":     float(test_after["mae_mean"]     - test_before["mae_mean"]),
            "mse_kw2_delta": float(test_after["mse_kw2_mean"] - test_before["mse_kw2_mean"]),
        }

        cluster_total = int(cluster_assignment_counts.sum())
        top1_share = float(
            cluster_assignment_counts.max() / max(cluster_total, 1)
        )
        n_clusters_used_on_test = int((cluster_assignment_counts > 0).sum())

        # 4) Snapshot in-memory + append JSONL.
        codebook_history["rounds"].append(int(round_idx))
        codebook_history["codebook"].append(torch.from_numpy(codebook.copy()))
        codebook_history["offsets"].append(torch.from_numpy(offsets.copy()))

        row = {
            "round": int(round_idx),
            "epoch_equivalent": float(epoch_equivalent),
            "wall_seconds_round": float(wall_seconds),
            "codebook_wall_seconds": float(time.time() - t_cb),
            "n_test_windows_total": int(n_test_windows_total),
            "codebook_diag": codebook_diag,
            "cluster_assignment_top1_share": top1_share,
            "n_clusters_used_on_test": n_clusters_used_on_test,
            "test_before": test_before,
            "test_after":  test_after,
            "lift": lift,
        }
        with codebook_log_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        print(
            f"[{cell_name}] R{round_idx:02d} codebook ΔPAPE={lift['pape_delta']:+.2f}  "
            f"util={codebook_diag['utilization']:.3f}  "
            f"ppl={codebook_diag['perplexity']:.2f}  "
            f"cb_wall={row['codebook_wall_seconds']:.1f}s"
        )

    return _callback


# ============================================================================
# Main per-seed × per-algorithm driver
# ============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "V9-RoundCB — federated post-hoc codebook fit at the end of every "
            "round of a v06-style FL backbone training loop. Backbone forward "
            "NEVER touches the codebook (no commit_loss). Single seed × single "
            "algorithm per invocation."
        )
    )
    ap.add_argument("--seed", type=int, default=RANDOM_SEED,
                    help="Single seed (memory: feedback_argparse_per_seed).")
    ap.add_argument("--algorithm", default="fedavg", choices=list(_ALGO_PRETTY.keys()))

    # Backbone training (mirror experiments/v06_round_dynamics/02_fl_dynamics.py).
    ap.add_argument("--rounds", type=int, default=10,
                    help="Quick default for V9-RoundCB (v06 used 20).")
    ap.add_argument("--local_epochs", type=int, default=5,
                    help="Quick default for V9-RoundCB (v06 used 40).")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--aux_lambda", type=float, default=0.3)
    ap.add_argument("--hr_weight", type=float, default=0.1)
    ap.add_argument("--no_amp", action="store_true")

    # Algorithm-specific extras (v06 defaults; only used when --algorithm matches).
    ap.add_argument("--fedprox_mu", type=float, default=0.01)
    ap.add_argument("--fedrep_head_epochs", type=int, default=1)
    ap.add_argument("--ditto_lam", type=float, default=0.1)
    ap.add_argument("--fedproto_K", type=int, default=32)
    ap.add_argument("--fedproto_lambda", type=float, default=0.1)

    # Codebook (v01–v05/FedCB-aligned defaults).
    ap.add_argument("--M", type=int, default=32, help="Global codebook size.")
    ap.add_argument("--K_local", type=int, default=2, help="Per-client Stage-1 cluster count.")
    ap.add_argument("--alpha_v0", type=float, default=1.0,
                    help="CMO correction strength (carry-over; do NOT re-tune).")
    ap.add_argument("--codebook_period", type=int, default=1,
                    help="Fit codebook every N rounds (1 = every round, 2 = R2/R4/...).")

    ap.add_argument("--output_namespace", type=str, default="v09_round_vq_codebook")

    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    use_amp = not args.no_amp

    cell_name = f"V9-RoundCB-{_ALGO_PRETTY[args.algorithm]}"
    out_dir = OUTPUT_DIR / args.output_namespace / f"seed{args.seed}" / cell_name
    out_dir.mkdir(parents=True, exist_ok=True)
    codebook_log_path = out_dir / "codebook_log.jsonl"
    if codebook_log_path.exists():
        codebook_log_path.unlink()

    print(f"[{cell_name}] seed={args.seed}  algorithm={args.algorithm}  "
          f"rounds={args.rounds}  local_epochs={args.local_epochs}  "
          f"batch={args.batch_size}  amp={use_amp}")
    print(f"[{cell_name}] codebook: M={args.M}  K_local={args.K_local}  "
          f"alpha_v0={args.alpha_v0}  period={args.codebook_period}")
    print(f"[{cell_name}] out_dir={out_dir}")

    # 1) Per-client 70/10/20 splits (cached on disk).
    print(f"[{cell_name}] building per-client 70/10/20 splits ...")
    splits = build_per_client_splits(seed=args.seed)
    print(f"[{cell_name}] {len(splits)} apartments retained.")

    # 2) Algorithm-specific kwargs (mirror 02_fl_dynamics.py).
    algo_kwargs: dict = {}
    if args.algorithm == "fedprox":
        algo_kwargs["mu"] = float(args.fedprox_mu)
    elif args.algorithm == "fedrep":
        algo_kwargs["head_epochs"] = int(args.fedrep_head_epochs)
    elif args.algorithm == "ditto":
        algo_kwargs["lam"] = float(args.ditto_lam)
    elif args.algorithm == "fedproto":
        algo_kwargs["K"] = int(args.fedproto_K)
        algo_kwargs["lambda_proto"] = float(args.fedproto_lambda)

    # 3) Build the round-end callback that fits the codebook + evaluates lift.
    codebook_history: dict = {"rounds": [], "codebook": [], "offsets": []}
    cb = _make_round_end_callback(
        splits,
        M=int(args.M),
        K_local=int(args.K_local),
        alpha_v0=float(args.alpha_v0),
        batch_size=int(args.batch_size),
        use_amp=bool(use_amp),
        seed=int(args.seed),
        codebook_period=int(args.codebook_period),
        codebook_log_path=codebook_log_path,
        codebook_history=codebook_history,
        cell_name=cell_name,
    )

    # 4) Federated training with codebook callback.
    t0 = time.time()
    result = run_fl_aux(
        algorithm=args.algorithm, splits=splits,
        rounds=int(args.rounds), local_epochs=int(args.local_epochs),
        lr=float(args.lr), batch_size=int(args.batch_size),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed), use_amp=bool(use_amp),
        aux_lambda=float(args.aux_lambda), hr_weight=float(args.hr_weight),
        on_round_end=cb,
        **algo_kwargs,
    )
    elapsed = time.time() - t0

    # 5) Persist final state dict.
    torch.save(result["final_state_dict"], out_dir / "final_state_dict.pt")

    # 6) Persist codebook history (stack into (R', M, D) / (R', M, H) tensors).
    if codebook_history["rounds"]:
        cb_stack = torch.stack(codebook_history["codebook"], dim=0)   # (R', M, D)
        of_stack = torch.stack(codebook_history["offsets"], dim=0)    # (R', M, H)
    else:
        cb_stack = torch.zeros((0, int(args.M), 64), dtype=torch.float32)
        of_stack = torch.zeros((0, int(args.M), HORIZON), dtype=torch.float32)
    torch.save(
        {
            "rounds":   list(codebook_history["rounds"]),
            "codebook": cb_stack,
            "offsets":  of_stack,
        },
        out_dir / "codebook_history.pt",
    )

    # 7) Replay codebook_log.jsonl for the lift_trajectory + last test_before.
    lift_trajectory: list[dict] = []
    last_row: dict | None = None
    if codebook_log_path.exists():
        with codebook_log_path.open() as fh:
            for line in fh:
                row = json.loads(line)
                lift_trajectory.append({
                    "round":      int(row["round"]),
                    "pape_delta": float(row["lift"]["pape_delta"]),
                    "mae_delta":  float(row["lift"]["mae_delta"]),
                    "hr@1_delta": float(row["lift"]["hr@1_delta"]),
                })
                last_row = row

    # 8) result.json.
    result_json = {
        "cell": cell_name,
        "algorithm": f"{args.algorithm}_aux",
        "seed": int(args.seed),
        "n_clients": int(result["n_train_clients"]),
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "batch": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "aux_lambda": float(args.aux_lambda),
        "hr_weight": float(args.hr_weight),
        "M": int(args.M),
        "K_local": int(args.K_local),
        "alpha_v0": float(args.alpha_v0),
        "codebook_period": int(args.codebook_period),
        "use_amp": bool(use_amp),
        "algo_kwargs": algo_kwargs,
        "history": result["history"],
        # Final-round backbone metrics: use the LAST codebook_log row's test_before
        # (= backbone test after R_final rounds, evaluated against test split,
        # before the CMO correction is applied).
        "final_test_before": (last_row["test_before"] if last_row else None),
        "final_test_after":  (last_row["test_after"]  if last_row else None),
        "final_lift":        (last_row["lift"]        if last_row else None),
        "lift_trajectory":   lift_trajectory,
        "elapsed_seconds":   float(elapsed),
        "comment": (
            "V9-RoundCB — federated post-hoc codebook fit at the end of every "
            "round (or every codebook_period rounds) of a v06-style FL backbone "
            "training loop. Backbone forward never touches the codebook (no "
            "commit_loss). Pure analysis: how does the post-hoc codebook + its "
            "lift evolve as the backbone trains?"
        ),
    }
    with (out_dir / "result.json").open("w") as fh:
        json.dump(result_json, fh, indent=2)

    if last_row is not None:
        print(f"[{cell_name}] done.  final BEFORE PAPE={last_row['test_before']['pape_mean']:.2f}  "
              f"AFTER PAPE={last_row['test_after']['pape_mean']:.2f}  "
              f"ΔPAPE={last_row['lift']['pape_delta']:+.2f}  "
              f"elapsed={elapsed:.0f}s")
    else:
        print(f"[{cell_name}] done.  (no codebook fits — check --codebook_period)  "
              f"elapsed={elapsed:.0f}s")


if __name__ == "__main__":
    main()
