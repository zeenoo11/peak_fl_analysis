"""V6-Dyn-A — centralised pooled SGD on 114 UMass apartments (per-seed driver).

(한글 요약)
plan v06-01 §"Goals" G1 — v06 의 *상한선* (centralised pooled SGD upper bound)
및 round-logger 동작을 검증하는 Gate 1 cell. 114가구
(``filter_valid_apartments(min_hours=7000)`` 결과) train 윈도우를 단일
DataLoader 로 합쳐 NBEATSxAux 를 학습하고, 매 epoch 끝에 RoundLogger 가 동일한
``round_log.jsonl`` schema 로 across-client val 평균을 기록한다.

Hyperparameters (plan §2)
-------------------------
epochs = 40, batch = 512, lr = 1e-3, weight_decay = 1e-5,
aux_lambda = 0.3, hr_weight = 0.1, optimizer = Adam, AMP = bfloat16 on CUDA.

Output
------
``outputs/v06_round_dynamics/seed{S}/V6-Dyn-A_centralised/``
    ├── round_log.jsonl        (per-epoch + terminal-test rows)
    ├── final_state_dict.pt    (fresh-init NBEATSxAux trained on pooled data)
    └── result.json            (terminal val + test summary, conference-compatible schema)

Per-seed argparse — multi-seed sweep ({42,123,7}) is the executor's job
(memory: feedback_argparse_per_seed). Multi-seed loops MUST stay outside
this script.
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
from fl.centralised_pooled import centralised_pooled_train
from fl.fedavg_aux import init_backbone_aux
from fl.round_logger import RoundLogger


_BASE_CELL_NAME = "V6-Dyn-A_centralised"


def _aux_suffix(aux_lambda: float, default_lambda: float = 0.3) -> str:
    """Mirror of ``02_fl_dynamics._aux_suffix`` — namespaced suffix so the
    λ_aux=0 ablation does not overwrite the default-lambda directory.

    ``--aux_lambda 0.3`` (default) → ``""``    (existing dir untouched)
    ``--aux_lambda 0``             → ``"-MAEonly"``
    ``--aux_lambda 0.1``           → ``"-aux0.1"``
    """
    if float(aux_lambda) == float(default_lambda):
        return ""
    if float(aux_lambda) == 0.0:
        return "-MAEonly"
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


def _build_cell_name(aux_lambda: float, hr_weight: float = 0.1) -> str:
    return (
        f"{_BASE_CELL_NAME}"
        f"{_aux_suffix(aux_lambda)}"
        f"{_hr_suffix(hr_weight)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "V6-Dyn-A centralised pooled SGD on 114 UMass 2016 apartments. "
            "Single seed per invocation; outer launcher loops over {42,123,7}."
        )
    )
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--aux_lambda", type=float, default=0.3)
    ap.add_argument("--hr_weight", type=float, default=0.1)
    ap.add_argument("--no_amp", action="store_true",
                    help="Disable bf16 autocast (auto-disabled on CPU regardless).")
    ap.add_argument("--output_namespace", type=str, default="v06_round_dynamics",
                    help="Top-level output sub-directory under outputs/ (default: v06_round_dynamics; "
                         "v07 launcher overrides to v07_loss_budget_sweeps).")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_amp = not args.no_amp

    cell_name = _build_cell_name(args.aux_lambda, args.hr_weight)
    out_dir = OUTPUT_DIR / args.output_namespace / f"seed{args.seed}" / cell_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "round_log.jsonl"
    if log_path.exists():
        log_path.unlink()  # fresh start per invocation

    print(f"[V6-Dyn-A] seed={args.seed}  epochs={args.epochs}  batch={args.batch_size}  amp={use_amp}")
    print(f"[V6-Dyn-A] out_dir={out_dir}")

    # 1) Per-client splits (cached on disk; same cache used by 02_fl_dynamics.py).
    print("[V6-Dyn-A] building per-client 70/10/20 splits ...")
    splits = build_per_client_splits(seed=args.seed)
    print(f"[V6-Dyn-A] {len(splits)} apartments retained.")

    # 2) Eval data dicts for the logger (val + test windows).
    val_data = {apt: {"x": sp["val_x"],  "y": sp["val_y"]}  for apt, sp in splits.items()}
    test_data = {apt: {"x": sp["test_x"], "y": sp["test_y"]} for apt, sp in splits.items()}

    # 3) Round logger (per-epoch jsonl rows + terminal-test row at the end).
    logger = RoundLogger(
        log_path=log_path, splits=splits,
        val_data=val_data, test_data=test_data,
        batch_size=args.batch_size,
    )

    # 4) Train.
    t0 = time.time()
    result = centralised_pooled_train(
        splits,
        n_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        aux_lambda=args.aux_lambda, hr_weight=args.hr_weight,
        seed=args.seed, use_amp=use_amp,
        on_round_end=logger.log_round,
    )
    elapsed = time.time() - t0

    # 5) Persist final state dict (loadable with strict=True).
    torch.save(result["final_state_dict"], out_dir / "final_state_dict.pt")

    # 6) Terminal val + test rows.
    fresh = init_backbone_aux(seed=args.seed)
    fresh.load_state_dict(result["final_state_dict"], strict=True)
    terminal_row = logger.log_terminal(model=fresh, wall_total=elapsed)
    logger.close()

    # 7) result.json (conference-compatible schema).
    result_json = {
        "cell": cell_name,
        "algorithm": "centralised_pooled_sgd",
        "seed": int(args.seed),
        "n_clients": int(result["n_clients"]),
        "epochs": int(args.epochs),
        "rounds": int(args.epochs),
        "batch": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "aux_lambda": float(args.aux_lambda),
        "hr_weight": float(args.hr_weight),
        "use_amp": bool(use_amp),
        "history": result["history"],
        "val_terminal":  terminal_row["val"],
        "test_terminal": terminal_row["test"],
        "comm_total_bytes": {"upload_cum": 0, "broadcast_cum": 0},
        "drift_l2_mean_over_rounds": 0.0,
        "elapsed_seconds": float(elapsed),
    }
    with (out_dir / "result.json").open("w") as fh:
        json.dump(result_json, fh, indent=2)
    print(f"[V6-Dyn-A] done.  val.PAPE={terminal_row['val']['pape_mean']:.2f}  "
          f"test.PAPE={terminal_row['test']['pape_mean']:.2f}  elapsed={elapsed:.0f}s")


if __name__ == "__main__":
    main()
