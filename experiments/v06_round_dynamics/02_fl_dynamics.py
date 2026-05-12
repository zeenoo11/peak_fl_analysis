"""V6-Dyn-B-{FedAvg, FedProx, FedRep, Ditto, FedProto} — federated training driver.

(한글 요약)
plan v06-01 §"Goals" G2 — conference 5종 FL 알고리즘의 *라운드별 trajectory*.
모든 100가구가 학습에 참여하며, 라운드 종료 직후 ``RoundLogger`` 가 100가구
의 *자기 val 윈도우* 에서 PAPE/HR/MAE/MSE(kW²) 를 across-client 평균/표준편차로
기록한다.

Backbone / loss / hyperparameter (전 5종 cell 공통, plan §2)
-----------------------------------------------------------
- NBEATSxAux(latent_source='h_generic') — 모든 5종이 동일한 backbone.
- L = MAE(ŷ, y) + 0.3 · peak_aux(ŷ, y; hr_weight=0.1) — combined loss.
- AdamW (Adam) lr=1e-3, weight_decay=1e-5.
- batch=512, rounds=20, local_epochs=40, full participation (C=1.0).
- Algorithm-specific extras: FedProx mu=0.01, FedRep head_epochs=1,
  Ditto lam=0.1, FedProto K=32 lambda_proto=0.1.

Output
------
``outputs/v06_round_dynamics/seed{S}/V6-Dyn-B-{Algo}/``
    ├── round_log.jsonl
    ├── final_state_dict.pt
    └── result.json     (conference Table-compatible schema)

Per-seed argparse — multi-seed sweep is the executor's job.
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

from config import OUTPUT_DIR, RANDOM_SEED
from dataloader.per_client_split import build_per_client_splits
from fl.fedavg_aux import init_backbone_aux
from fl.round_aux import run_fl_aux
from fl.round_logger import RoundLogger


_CELL_PREFIX = "V6-Dyn-B-"

_ALGO_PRETTY = {
    "fedavg":   "FedAvg",
    "fedprox":  "FedProx",
    "fedrep":   "FedRep",
    "ditto":    "Ditto",
    "fedproto": "FedProto",
}


def _aux_suffix(aux_lambda: float, default_lambda: float = 0.3) -> str:
    """Return the cell-name suffix for an ``aux_lambda`` value.

    - ``aux_lambda == default_lambda`` (0.3) → ``""`` (back-compat: existing
      18-run results stay at the un-suffixed cell name).
    - ``aux_lambda == 0.0``                 → ``"-MAEonly"``  (paper-friendly
      label for the MAE-only ablation namespace).
    - any other value (e.g. 0.1)            → ``"-aux{value}"`` with trailing
      zeros stripped (``-aux0.1``).

    Comparison uses an exact-zero check (``aux_lambda == 0.0``) — the user
    is expected to pass ``--aux_lambda 0`` for the ablation. Floating-point
    near-zero values are NOT collapsed to ``-MAEonly``.
    """
    if float(aux_lambda) == float(default_lambda):
        return ""
    if float(aux_lambda) == 0.0:
        return "-MAEonly"
    # Generic suffix — strip trailing zeros / a trailing dot for readability.
    raw = f"{float(aux_lambda):g}"
    return f"-aux{raw}"


def _hr_suffix(hr_weight: float, default_hr: float = 0.1) -> str:
    """v07 ablation — append ``-hr{value}`` only when hr_weight differs from
    the v06 default (0.1). default → ``""`` keeps every existing v06 cell
    directory untouched."""
    if float(hr_weight) == float(default_hr):
        return ""
    raw = f"{float(hr_weight):g}"
    return f"-hr{raw}"


def _build_cell_name(algorithm: str, aux_lambda: float, hr_weight: float = 0.1) -> str:
    """Public helper for tests + drivers: ``V6-Dyn-B-{Algo}{aux_suffix}{hr_suffix}``."""
    base = f"{_CELL_PREFIX}{_ALGO_PRETTY[algorithm]}"
    return f"{base}{_aux_suffix(aux_lambda)}{_hr_suffix(hr_weight)}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "V6-Dyn-B {FedAvg/FedProx/FedRep/Ditto/FedProto} on 100 UMass 2016 apartments. "
            "Single seed × single algorithm per invocation."
        )
    )
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--algorithm", required=True, choices=list(_ALGO_PRETTY.keys()))
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local_epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--aux_lambda", type=float, default=0.3)
    ap.add_argument("--hr_weight", type=float, default=0.1)
    ap.add_argument("--no_amp", action="store_true")
    # Algorithm-specific extras (plan §2 defaults).
    ap.add_argument("--fedprox_mu", type=float, default=0.01)
    ap.add_argument("--fedrep_head_epochs", type=int, default=1)
    ap.add_argument("--ditto_lam", type=float, default=0.1)
    ap.add_argument("--fedproto_K", type=int, default=32)
    ap.add_argument("--fedproto_lambda", type=float, default=0.1)
    ap.add_argument("--output_namespace", type=str, default="v06_round_dynamics",
                    help="Top-level output sub-directory under outputs/ (default: v06_round_dynamics; "
                         "v07 launcher overrides to v07_loss_budget_sweeps).")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    use_amp = not args.no_amp

    cell_name = _build_cell_name(args.algorithm, args.aux_lambda, args.hr_weight)
    out_dir = OUTPUT_DIR / args.output_namespace / f"seed{args.seed}" / cell_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "round_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    print(f"[{cell_name}] seed={args.seed}  rounds={args.rounds}  "
          f"local_epochs={args.local_epochs}  batch={args.batch_size}  amp={use_amp}")
    print(f"[{cell_name}] out_dir={out_dir}")

    # 1) Per-client splits.
    print(f"[{cell_name}] building per-client 70/10/20 splits ...")
    splits = build_per_client_splits(seed=args.seed)
    print(f"[{cell_name}] {len(splits)} apartments retained.")

    val_data  = {apt: {"x": sp["val_x"],  "y": sp["val_y"]}  for apt, sp in splits.items()}
    test_data = {apt: {"x": sp["test_x"], "y": sp["test_y"]} for apt, sp in splits.items()}

    logger = RoundLogger(
        log_path=log_path, splits=splits,
        val_data=val_data, test_data=test_data,
        batch_size=args.batch_size,
    )

    # 2) Algorithm-specific kwargs.
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

    # 3) Federated training with on_round_end -> logger.log_round.
    t0 = time.time()
    result = run_fl_aux(
        algorithm=args.algorithm, splits=splits,
        rounds=args.rounds, local_epochs=args.local_epochs,
        lr=args.lr, batch_size=args.batch_size, weight_decay=args.weight_decay,
        seed=args.seed, use_amp=use_amp,
        aux_lambda=args.aux_lambda, hr_weight=args.hr_weight,
        on_round_end=logger.log_round,
        **algo_kwargs,
    )
    elapsed = time.time() - t0

    # 4) Persist final state dict.
    torch.save(result["final_state_dict"], out_dir / "final_state_dict.pt")

    # 5) Terminal val + test (using a fresh model with the final state).
    fresh = init_backbone_aux(seed=args.seed)
    fresh.load_state_dict(result["final_state_dict"], strict=True)
    val_terminal_row  = logger.log_terminal(model=fresh, wall_total=elapsed, split="val")
    test_terminal_row = logger.log_terminal(model=fresh, wall_total=elapsed, split="test")
    logger.close()

    # 6) Aggregate drift_l2_mean_over_rounds from the jsonl rows.
    drifts = []
    with log_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["round"] >= 1:  # skip terminal -1
                drifts.append(float(row["drift_l2"]))
    drift_mean = float(np.mean(drifts)) if drifts else 0.0

    # Comm totals from the last in-train row.
    upload_cum = 0
    broadcast_cum = 0
    with log_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["round"] >= 1:
                upload_cum = int(row["comm"]["upload_bytes_cum"])
                broadcast_cum = int(row["comm"]["broadcast_bytes_cum"])

    # 7) result.json
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
        "use_amp": bool(use_amp),
        "C": 1.0,
        "algo_kwargs": algo_kwargs,
        "history": result["history"],
        "val_terminal":  val_terminal_row["val"],
        "test_terminal": test_terminal_row["val"],
        "comm_total_bytes": {"upload_cum": upload_cum, "broadcast_cum": broadcast_cum},
        "drift_l2_mean_over_rounds": drift_mean,
        "elapsed_seconds": float(elapsed),
    }
    with (out_dir / "result.json").open("w") as fh:
        json.dump(result_json, fh, indent=2)
    print(f"[{cell_name}] done.  val.PAPE={val_terminal_row['val']['pape_mean']:.2f}  "
          f"test.PAPE={test_terminal_row['val']['pape_mean']:.2f}  "
          f"drift_mean={drift_mean:.3f}  elapsed={elapsed:.0f}s")


if __name__ == "__main__":
    main()
