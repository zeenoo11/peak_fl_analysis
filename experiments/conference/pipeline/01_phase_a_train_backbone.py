"""Phase A — FedAvg-NBEATSxAux backbone training (per-seed driver).

(한글 요약)
KIIE conference 발표 (``papers/conference_draft/presentation.md``)의 §3.3
"Federated Codebook Construction" 첫 단계 (Phase A) 와 §"Codebook Correction
Module 효과 측정" 표 (lines 211-218)의 *Backbone (no correction)* 행에 해당하는
스크립트. NBEATSxAux 백본 + auxiliary head를 80가구에 대해 FedAvg로 federated
학습하고, federated 백본 자체의 cold metric (``fl_only``)을 함께 저장해서 다음
ablation 단계에서 ``Backbone (no correction)`` 행의 source가 된다.

Defaults (rounds=20, local_epochs=2, lr=1e-3, batch=512, wd=1e-5, λ=0.3,
hr_weight=0.1, no_amp=False)는 ``experiments/v04_full_baseline_comparison/
09_fix_rerun/02_fedavg_nbeatsx_aux.py``와 *bit-equivalent*하게 일치하도록
맞췄다 — 동일 seed 입력에서 두 스크립트의 output이 같은 백본 weight를 만들도록
의도된 것이다 (단, 본 스크립트는 v04 internals를 inline 복사하지 않고 추출된
``src/fl/fedavg_aux.py``를 import해서 쓴다).

Per-seed argparse — 멀티시드 sweep ({42, 123, 7})은 외부 launcher가 ``--seed S``로
세 번 호출 (memory: feedback_argparse_per_seed). 결과는 ``result.json`` +
``final_state_dict.pt``로 저장. MLflow 사용 안 함 (이 repo의 컨벤션은 print + JSON).
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
from eval.cold_helpers import gather_cold, metrics_z_to_kw
from fl.base import DEVICE, apply_state_dict
from fl.fedavg_aux import fedavg_aux_round_loop
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
            "Conference Phase A — FedAvg over NBEATSxAux (backbone + aux head) "
            "for one seed. Defaults match v04 09_fix_rerun/02_fedavg_nbeatsx_aux.py "
            "for bit-equivalence under matching seed."
        )
    )
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--no_amp", action="store_true",
                    help="Disable bf16 autocast (auto-disabled on CPU regardless).")
    ap.add_argument("--aux_lambda", type=float, default=0.3,
                    help="Combined-loss weight on peak_aux (CLAUDE.md default = 0.3).")
    ap.add_argument("--hr_weight", type=float, default=0.1,
                    help="peak_aux internal hr-CE weight (CLAUDE.md default = 0.1).")
    ap.add_argument("--stride", type=int, default=HORIZON,
                    help="Cold-eval stride (default = horizon = 24, v01/v02 convention).")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    use_amp = not args.no_amp

    sp = load_v02_split(args.seed)
    train_apts, cold_apts = sp["train"], sp["cold"]

    out_dir = CONFERENCE_OUT_ROOT / f"seed{args.seed}" / "phase_a"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[conference phase_a] seed={args.seed}  rounds={args.rounds}  "
        f"local_epochs={args.local_epochs}  batch={args.batch_size}  amp={use_amp}"
    )
    gpu_start = _gpu_snapshot()
    print(f"[conference phase_a] GPU @start: {gpu_start}")

    # Phase A — federated training of NBEATSxAux.
    t0 = time.time()
    fa = fedavg_aux_round_loop(
        train_apts,
        rounds=args.rounds, local_epochs=args.local_epochs,
        lr=args.lr, batch_size=args.batch_size, weight_decay=args.weight_decay,
        seed=args.seed, use_amp=use_amp,
        aux_lambda=args.aux_lambda, hr_weight=args.hr_weight,
    )
    fl_elapsed = time.time() - t0
    print(f"[conference phase_a] FL training done in {fl_elapsed:.0f}s "
          f"({fl_elapsed/60:.1f} min)")

    # Persist the federated state dict first — this is what Phase B / Phase C
    # later load with strict=True. Save before computing fl_only so a Phase C
    # forward-pass crash does not lose the trained weights.
    torch.save(fa["final_state_dict"], out_dir / "final_state_dict.pt")

    # fl_only cold metrics — federated backbone, no codebook correction.
    # This row is what the ablation table calls "Backbone (no correction)".
    model = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
    apply_state_dict(model, fa["final_state_dict"])
    co = gather_cold(
        cold_apts, model,
        batch=args.batch_size, stride=args.stride, verbose_skips=False,
    )
    fl_only_metrics = metrics_z_to_kw(co["y_true_z"], co["y_hat_z"], co["mean"], co["std"])
    fl_only_metrics["n_cold_windows"] = int(co["y_true_z"].shape[0])
    fl_only_metrics["n_cold_apts"] = int(len(np.unique(co["apt"])))
    print(
        f"[conference phase_a] fl_only:  PAPE={fl_only_metrics['pape']:.2f}  "
        f"HR@1={fl_only_metrics['hr@1']:.1f}  HR@2={fl_only_metrics['hr@2']:.1f}"
    )

    elapsed = time.time() - t0
    gpu_end = _gpu_snapshot()

    result = {
        "algorithm": "fedavg_nbeatsx_aux",
        "seed": int(args.seed),
        "config": {
            "rounds": int(args.rounds),
            "local_epochs": int(args.local_epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "weight_decay": float(args.weight_decay),
            "use_amp": bool(use_amp),
            "aux_lambda": float(args.aux_lambda),
            "hr_weight": float(args.hr_weight),
            "stride": int(args.stride),
        },
        "history": fa["history"],
        "fl_only": fl_only_metrics,
        "n_train_clients": int(fa["n_train_clients"]),
        "n_cold_windows": int(fl_only_metrics["n_cold_windows"]),
        "n_cold_apts": int(fl_only_metrics["n_cold_apts"]),
        "elapsed_seconds": float(elapsed),
        "fl_elapsed_seconds": float(fl_elapsed),
        "gpu_at_start": gpu_start,
        "gpu_at_end": gpu_end,
        "comment": (
            "Conference Phase A: FedAvg over the full NBEATSxAux (backbone + "
            "aux head federated jointly) under combined loss "
            "L = MAE(y_hat, y) + 0.3 · peak_aux_loss(...). fl_only is the "
            "federated backbone's raw cold metric — this is the 'Backbone "
            "(no correction)' row of the ablation table in §Experiments."
        ),
    }

    with open(out_dir / "result.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[conference phase_a] saved -> {out_dir}")
    print(f"[conference phase_a] total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
