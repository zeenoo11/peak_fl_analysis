"""Quick plateau-onset analysis for v08 round-level trajectories."""
import sys
from pathlib import Path

import numpy as np

NPZ = Path(r"C:\Users\HOME\JW\Research Docs\FL_Peak_Project\outputs\v08_round_dynamics_long\trajectories.npz")
d = np.load(NPZ)

cells = [
    "V6-Dyn-B-FedAvg",
    "V6-Dyn-B-FedProx",
    "V6-Dyn-B-FedRep",
    "V6-Dyn-B-Ditto",
]


def plateau_round(x, y_per_seed, tol):
    m = np.nanmean(y_per_seed, axis=0)
    final = float(np.nanmean(m[-10:]))
    for i in range(len(m) - 5):
        if all(abs(m[i + k] - final) < tol for k in range(5)):
            return int(x[i]), final, float(m[i])
    return None, final, None


# Three tolerances
for tol in (0.3, 0.5, 0.8):
    print(f"\n=== plateau (5-round consecutive |Δ| < {tol} PAPE from final) ===")
    print(f"{'Cell':<22} {'onset':<8} {'PAPE@onset':<12} {'final PAPE':<12}")
    for cell in cells:
        x = d[f"{cell}_round_idx"][0]
        y = d[f"{cell}_test_pape_mean"]
        r, final, p_at = plateau_round(x, y, tol=tol)
        p_str = f"{p_at:.2f}" if p_at is not None else "-"
        print(f"{cell:<22} {str(r):<8} {p_str:<12} {final:.2f}")

# Round-by-round table at key rounds
print("\n=== test PAPE @ key rounds (mean across seeds) ===")
header = f"{'Round':<8}"
for cell in cells:
    header += f"{cell.replace('V6-Dyn-B-', ''):<12}"
print(header)

key_rounds = [1, 3, 5, 7, 10, 15, 20, 25, 30, 50, 75, 100, 125, 150]
for kr in key_rounds:
    line = f"{kr:<8}"
    for cell in cells:
        x = d[f"{cell}_round_idx"][0]
        y = d[f"{cell}_test_pape_mean"]
        idx = np.where(x == kr)[0]
        if len(idx) > 0:
            m = float(np.nanmean(y[:, idx[0]]))
            line += f"{m:<12.2f}"
        else:
            line += f"{'-':<12}"
    print(line)

# Min-PAPE round (the dip)
print("\n=== minimum test PAPE round (the 'dip') ===")
for cell in cells:
    x = d[f"{cell}_round_idx"][0]
    y = d[f"{cell}_test_pape_mean"]
    m = np.nanmean(y, axis=0)
    s = np.nanstd(y, axis=0, ddof=1)
    i_min = int(np.argmin(m))
    print(f"{cell:<22} min PAPE = {m[i_min]:.2f} ± {s[i_min]:.2f} at round {int(x[i_min])}")
