"""iter5-D: W5 grid search — squeeze the hybrid mechanism."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import OUTPUT_DIR, RANDOM_SEED
from dataloader.splits import load_v10_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
mech = import_module("09_iter4_mechanisms")

OUT = OUTPUT_DIR / "v01_peak_from_latent"
ITER5D = OUT / "iter5_D"
ITER5D.mkdir(parents=True, exist_ok=True)


def main():
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    split = load_v10_split()
    print("[setup] loading T2 features...")
    tr = mech.gather(split["train"]); co = mech.gather(split["cold"])
    print(f"[data] train={tr['lat'].shape[0]} cold={co['lat'].shape[0]}")

    # cache base for printing
    base_pape, base_hr1, base_hr2 = None, None, None

    rows = []
    sigmas = [0.5, 1.0, 1.5, 2.0, 3.0]
    alpha_v0s = [1.0, 1.5, 2.0, 2.5, 3.0]
    alpha_w1s = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]

    print(f"\n[grid] σ × α_v0 × α_w1 = {len(sigmas)}×{len(alpha_v0s)}×{len(alpha_w1s)} = {len(sigmas)*len(alpha_v0s)*len(alpha_w1s)} combos")
    print(f"{'σ':>4s}  {'α_v0':>5s}  {'α_w1':>5s}  PAPE     HR@1  HR@2  argmax_MAE")
    for sig in sigmas:
        for a_v in alpha_v0s:
            for a_w in alpha_w1s:
                m = mech.run_W5(tr, co, alpha_v0=a_v, alpha_w1=a_w, sigma=sig)
                if base_pape is None:
                    base_pape, base_hr1, base_hr2 = m["base_pape"], m["base_hr@1"], m["base_hr@2"]
                rows.append({"sigma": sig, "alpha_v0": a_v, "alpha_w1": a_w, **m})
                tag = ""
                if m["corr_pape"] < 38.0 and m["corr_hr@1"] >= 26.5:
                    tag = " ★"
                print(f"{sig:4.1f}  {a_v:5.2f}  {a_w:5.2f}  {m['corr_pape']:5.2f}    "
                      f"{m['corr_hr@1']:.1f}   {m['corr_hr@2']:.1f}  {m['corr_argmax_mae']:.2f}{tag}")

    # M ablation at best settings
    print("\n[M ablation] using α_v0=2.0 α_w1=0.5 σ=1.5")
    print(f"{'M':>4s}   PAPE     HR@1  HR@2")
    for M in [16, 32, 64, 128]:
        # need to re-call with M parameter
        _, idx_tr, cb, offsets_v0 = mech.run_V0(tr, co, M=M, alpha=2.0)
        cold_cluster = mech.cold_assign_via_key(tr["key"], co["key"], idx_tr)
        g = mech.gauss_template(co["pred_hr"], co["pred_amp"], sigma=1.5)
        corrected = co["base_z"] + 2.0 * offsets_v0[cold_cluster] + 0.5 * g
        m = mech.metrics(co["true_z"], co["base_z"], corrected, co["mean"], co["std"])
        print(f"{M:4d}   {m['corr_pape']:5.2f}    {m['corr_hr@1']:.1f}   {m['corr_hr@2']:.1f}")
        rows.append({"sigma": 1.5, "alpha_v0": 2.0, "alpha_w1": 0.5, "M": M, **m})

    # find best PAPE while HR@1 >= 26.5
    candidates = [r for r in rows if r["corr_hr@1"] >= 26.5]
    if candidates:
        best = min(candidates, key=lambda r: r["corr_pape"])
        print(f"\n[best @ HR@1≥26.5]  σ={best.get('sigma','?')}  α_v0={best.get('alpha_v0','?')}  "
              f"α_w1={best.get('alpha_w1','?')}  PAPE={best['corr_pape']:.2f}  HR@1={best['corr_hr@1']:.1f}")

    # absolute best PAPE
    best_pape = min(rows, key=lambda r: r["corr_pape"])
    print(f"[best PAPE absolute] σ={best_pape.get('sigma','?')}  α_v0={best_pape.get('alpha_v0','?')}  "
          f"α_w1={best_pape.get('alpha_w1','?')}  PAPE={best_pape['corr_pape']:.2f}  "
          f"HR@1={best_pape['corr_hr@1']:.1f}")

    print(f"\n[baseline] PAPE={base_pape:.2f}  HR@1={base_hr1:.1f}  HR@2={base_hr2:.1f}")

    with open(ITER5D / "iter5D_results.json", "w") as fh:
        json.dump({"rows": rows, "baseline": {"pape": base_pape, "hr1": base_hr1, "hr2": base_hr2}},
                  fh, indent=2)
    print(f"\n[done] wrote {ITER5D / 'iter5D_results.json'}")


if __name__ == "__main__":
    main()
