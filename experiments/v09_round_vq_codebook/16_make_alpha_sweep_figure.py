"""Figure: CMO correction-strength (alpha) PAPE-vs-MAE Pareto, aux=0 backbone.

Adopts the v06 F7 design (experiments/v06_round_dynamics/11_make_ablation_figures.py):
ΔMAE on x, ΔPAPE on y, one colored polyline per FL algorithm threading its
alpha operating points, distinct markers at alpha in {0.5, 1.0, 1.5, 2.0}, and
x/y error bars = 3-seed std. Everything is a delta from the uncorrected base
(alpha=0). Lower-left = better (PAPE down, MAE unchanged); the curve bends
down-right as alpha grows: peak gain bought at rising overall-error cost.

Reads the per-seed alpha_sweep.json written by 15_alpha_sweep.py.

NOTE: v09 has no aux=0 *centralised* alpha sweep (centralised cell not run yet
-- see CLAUDE.md TODO), so only the 5 FL algorithms are drawn.

Writes: papers/conference_draft/figures/fig14_alpha_sweep.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parents[2]
NS = "v09_round_vq_codebook_R20_MAEonly"
SEEDS = (42, 123, 7)
OUT = BASE / "papers/conference_draft/figures"
OUT.mkdir(parents=True, exist_ok=True)

# v06 F7 / Wong palette so this sits next to the rest of the deck.
COLORS = {
    "FedAvg":   "#0072B2",
    "FedProx":  "#D55E00",
    "FedRep":   "#009E73",
    "Ditto":    "#CC79A7",
    "FedProto": "#E69F00",
}
ALPHA_MAX = 2.0   # display range (data on disk goes to 3.0)
MARKER_ALPHAS = [0.5, 1.0, 1.5, 2.0]
MARKER_FOR_ALPHA = {0.5: "o", 1.0: "s", 1.5: "^", 2.0: "D"}
MARKER_SIZE = {0.5: 10, 1.0: 10, 1.5: 10, 2.0: 10}


def _load(algo):
    """Return (alphas, dpape[seed,a], dmae_pct[seed,a]).

    dpape  = PAPE(alpha) - PAPE(0)            in percentage points (PAPE is a %).
    dmae_pct = 100 * (MAE(alpha) - MAE(0)) / MAE(0)   relative MAE change in %.
    """
    alphas, dpape, dmae_pct = None, [], []
    for s in SEEDS:
        p = BASE / "outputs" / NS / f"seed{s}" / f"V9-RoundCB-{algo}" / "alpha_sweep.json"
        d = json.loads(p.read_text())
        if alphas is None:
            alphas = np.asarray(d["alphas"], dtype=float)
        pape = np.asarray([r["pape"] for r in d["sweep"]], dtype=float)
        mae = np.asarray([r["mae"] for r in d["sweep"]], dtype=float)
        dpape.append(pape - pape[0])
        dmae_pct.append(100.0 * (mae - mae[0]) / mae[0])
    return alphas, np.asarray(dpape), np.asarray(dmae_pct)


plt.rcParams.update({"font.size": 16})
fig, ax = plt.subplots(1, 1, figsize=(13.0, 7.5))

alphas = None
for algo, col in COLORS.items():
    alphas, dpape, dmae = _load(algo)
    mx_all, my_all = dmae.mean(0), dpape.mean(0)
    sx_all, sy_all = dmae.std(0, ddof=1), dpape.std(0, ddof=1)
    # Polyline through alpha points up to ALPHA_MAX (smooth sweep).
    disp = alphas <= ALPHA_MAX + 1e-9
    ax.plot(mx_all[disp], my_all[disp], color=col, linewidth=2.0, alpha=0.85, zorder=2)
    # Distinct markers + error bars at the 4 operating points.
    for a in MARKER_ALPHAS:
        i = int(np.argmin(np.abs(alphas - a)))
        ax.errorbar(
            mx_all[i], my_all[i], xerr=sx_all[i], yerr=sy_all[i],
            fmt=MARKER_FOR_ALPHA[a], color=col,
            markersize=MARKER_SIZE[a], markeredgecolor="black", markeredgewidth=0.7,
            elinewidth=0.8, capsize=2.5, ecolor="#666666", zorder=3,
        )

ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axvline(0.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)
ax.invert_yaxis()   # smaller dPAPE (= better) goes UP

# Default operating REGION: the spread of dMAE% over the 5 algorithms at
# alpha=1.0 (each algo's alpha=1.0 sits at a slightly different x).
i10 = int(np.argmin(np.abs(alphas - 1.0)))
x10 = np.array([_load(a)[2].mean(0)[i10] for a in COLORS])
ax.axvspan(x10.min(), x10.max(), color="#444444", alpha=0.12, zorder=0)
ax.text(x10.mean(), 0.82, r"default $\alpha$=1.0",
        transform=ax.get_xaxis_transform(),
        va="top", ha="center", fontsize=15, color="#333333")

# Sweet-spot annotations (use FedAvg curve as the anchor reference).
# y-axis is inverted: more-negative dPAPE (= better) is at the TOP.
_, dpA, dmA = _load("FedAvg")
i05 = int(np.argmin(np.abs(alphas - 0.5)))
i20 = int(np.argmin(np.abs(alphas - 2.0)))
ax.annotate(
    "MAE-near-zero-cost\n" + r"($\alpha=0.5$)",
    xy=(dmA.mean(0)[i05], dpA.mean(0)[i05]), xytext=(2.0, -3.0),
    fontsize=15, color="#222222", ha="left",
    arrowprops=dict(arrowstyle="->", color="#444444", lw=1.1),
)
ax.annotate(
    "PAPE-aggressive\n" + r"($\alpha=2.0$)",
    xy=(dmA.mean(0)[i20], dpA.mean(0)[i20]), xytext=(4.0, -10.0),
    fontsize=15, color="#222222", ha="left",
    arrowprops=dict(arrowstyle="->", color="#444444", lw=1.1),
)

ax.set_xlabel(r"$\Delta$MAE  (%)")
ax.set_ylabel(r"$\Delta$PAPE  (%p)")
ax.set_title(
    r"Codebook correction strength $\alpha$ — PAPE vs MAE Pareto"
)
ax.grid(True, alpha=0.3)

# Two legends on the right margin.
alpha_handles = [
    plt.Line2D([0], [0], marker=MARKER_FOR_ALPHA[a], linestyle="",
               color="#444444", markersize=10, markeredgecolor="black",
               markeredgewidth=0.7, label=fr"$\alpha = {a}$")
    for a in MARKER_ALPHAS
]
cell_handles = [
    plt.Line2D([0], [0], color=COLORS[c], linewidth=2.5, label=c)
    for c in COLORS
]
leg_alpha = ax.legend(
    handles=alpha_handles, loc="upper left", bbox_to_anchor=(1.00, 1.0),
    frameon=True, title="Operating point", fontsize=14, title_fontsize=15,
)
ax.add_artist(leg_alpha)
ax.legend(
    handles=cell_handles, loc="upper left", bbox_to_anchor=(1.02, 0.55),
    frameon=True, fontsize=14, title="FL algorithm", title_fontsize=15,
)

fig.tight_layout()
fig.subplots_adjust(right=0.80)
out_path = OUT / "fig14_alpha_sweep.png"
fig.savefig(out_path, dpi=160, bbox_inches="tight")
print(f"wrote {out_path}")

print("\nalpha=1.0 (default) dPAPE(%pt) / dMAE(%), 3-seed mean:")
for algo in COLORS:
    _, dpape, dmae_pct = _load(algo)
    i = int(np.argmin(np.abs(alphas - 1.0)))
    print(f"  {algo:9s} dPAPE={dpape.mean(0)[i]:+.2f}  dMAE%={dmae_pct.mean(0)[i]:+.2f}")
