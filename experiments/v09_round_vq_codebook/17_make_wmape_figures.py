"""WMAPE companion figures (fig3-2, fig14-2) for the conference deck.

WMAPE (= WAPE) = 100 * sum|y-yhat| / sum|y|, a stable mean-percentage error that,
unlike raw MAPE, does not blow up on near-zero off-peak residential load (no
per-point division). It is the %-framed counterpart of MAE.

Both figures read the per-seed alpha_sweep.json written by 15_alpha_sweep.py
(which now logs `wmape` per alpha), aux=0 backbone, 5 FL algos x 3 seeds.

  fig4-2_peak_gain_vs_wmape_cost.png -- dual-axis bars per algorithm: peak error
        change (PAPE, Δ pp, left) vs average error change (WMAPE, Δ pp, right).
        Sibling of fig4 (which used MAE relative %). Both axes in percentage
        points: peak drops 4.5~5.5 pp while average rises only 0.2~0.6 pp.
  fig14-2_alpha_sweep_WMAPE.png   -- dPAPE vs dWMAPE Pareto across alpha. Sibling
        of fig14 (dPAPE vs dMAE%). Same up-right bend: peak gain bought at a
        small, stable WMAPE cost.

Writes: papers/conference_draft/figures/fig4-2_*, fig14-2_*
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
ALGOS = ["FedAvg", "FedProx", "FedRep", "Ditto", "FedProto"]
OUT = BASE / "papers/conference_draft/figures"
OUT.mkdir(parents=True, exist_ok=True)

# fig14 palette (matches 16_make_alpha_sweep_figure.py).
COLORS_F14 = {
    "FedAvg": "#0072B2", "FedProx": "#D55E00", "FedRep": "#009E73",
    "Ditto": "#CC79A7", "FedProto": "#E69F00",
}


def _load(algo):
    """Return (alphas, pape[seed,a], wmape[seed,a])."""
    alphas, pape, wmape = None, [], []
    for s in SEEDS:
        p = BASE / "outputs" / NS / f"seed{s}" / f"V9-RoundCB-{algo}" / "alpha_sweep.json"
        d = json.loads(p.read_text())
        if alphas is None:
            alphas = np.asarray(d["alphas"], dtype=float)
        pape.append([r["pape"] for r in d["sweep"]])
        wmape.append([r["wmape"] for r in d["sweep"]])
    return alphas, np.asarray(pape), np.asarray(wmape)


# Collect PAPE / WMAPE before(alpha=0) / after(alpha=1.0) per algo.
alphas = _load("FedAvg")[0]
i0 = int(np.argmin(np.abs(alphas - 0.0)))
i1 = int(np.argmin(np.abs(alphas - 1.0)))
# Per-algorithm final-model effect (after - before), 3-seed mean +/- std.
pap_m, pap_s, wm_m, wm_s = [], [], [], []
for a in ALGOS:
    _, pape, w = _load(a)
    dp = pape[:, i1] - pape[:, i0]            # PAPE change, percentage points
    dw = w[:, i1] - w[:, i0]                  # WMAPE change, percentage points
    pap_m.append(dp.mean()); pap_s.append(dp.std(ddof=1))
    wm_m.append(dw.mean());  wm_s.append(dw.std(ddof=1))
pap_m, pap_s, wm_m, wm_s = map(np.array, (pap_m, pap_s, wm_m, wm_s))

PAPE_C, WMAPE_C = "#0072B2", "#E69F00"


# ---------------------------------------------------------------------------
# fig4-2 — Peak gain (PAPE) vs average-error cost (WMAPE), per algorithm.
#          Sibling of fig4 (which used MAE relative %). Both quantities are
#          percentage-point changes, so they share ONE y-axis -- the asymmetry
#          (large negative PAPE vs tiny positive WMAPE) is read directly off it.
# ---------------------------------------------------------------------------
x = np.arange(len(ALGOS))
w = 0.38
with plt.rc_context({"font.size": 13}):
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.axhline(0, color="#888", lw=1.0)
    b1 = ax.bar(x - w / 2, pap_m, w, yerr=pap_s, color=PAPE_C, capsize=3,
                label="Peak error (PAPE), Δ points")
    b2 = ax.bar(x + w / 2, wm_m, w, yerr=wm_s, color=WMAPE_C, capsize=3,
                label="Average error (WMAPE), Δ points")
    for i in range(len(ALGOS)):
        ax.text(x[i] - w / 2, pap_m[i] - 0.18, f"{pap_m[i]:+.1f}", ha="center",
                va="top", fontsize=9.5, color=PAPE_C)
        ax.text(x[i] + w / 2, wm_m[i] + 0.10, f"{wm_m[i]:+.2f}", ha="center",
                va="bottom", fontsize=9.5, color="#9a6a00")
    ax.set_xticks(x)
    ax.set_xticklabels(ALGOS)
    ax.set_ylabel("Error change (Δ percentage points)")
    ax.set_ylim(float((pap_m - pap_s).min()) - 1.0,
                max(1.5, float((wm_m + wm_s).max()) + 0.5))
    ax.set_title("Large Peak Gain, Negligible Average Cost\n"
                 "Final-round effect per algorithm (WMAPE)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(handles=[b1, b2], loc="lower right", frameon=True, framealpha=0.9,
              facecolor="white", edgecolor="none", fontsize=10.5)
    fig.tight_layout()
    p = OUT / "fig4-2_peak_gain_vs_wmape_cost.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print(f"saved: {p}")


# ---------------------------------------------------------------------------
# fig14-2 — dPAPE vs dWMAPE Pareto across alpha (sibling of fig14).
# ---------------------------------------------------------------------------
ALPHA_MAX = 2.0
MARKER_ALPHAS = [0.5, 1.0, 1.5, 2.0]
MARKER_FOR_ALPHA = {0.5: "o", 1.0: "s", 1.5: "^", 2.0: "D"}

plt.rcParams.update({"font.size": 16})
fig, ax = plt.subplots(1, 1, figsize=(13.0, 7.5))

for algo, col in COLORS_F14.items():
    al, pape, wmape = _load(algo)
    dpape = (pape - pape[:, [i0]]).mean(0)
    dwm = (wmape - wmape[:, [i0]]).mean(0)
    sdp = (pape - pape[:, [i0]]).std(0, ddof=1)
    sdw = (wmape - wmape[:, [i0]]).std(0, ddof=1)
    disp = al <= ALPHA_MAX + 1e-9
    ax.plot(dwm[disp], dpape[disp], color=col, linewidth=2.0, alpha=0.85, zorder=2)
    for a in MARKER_ALPHAS:
        j = int(np.argmin(np.abs(al - a)))
        ax.errorbar(dwm[j], dpape[j], xerr=sdw[j], yerr=sdp[j],
                    fmt=MARKER_FOR_ALPHA[a], color=col, markersize=10,
                    markeredgecolor="black", markeredgewidth=0.7,
                    elinewidth=0.8, capsize=2.5, ecolor="#666666", zorder=3)

ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axvline(0.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)
ax.invert_yaxis()   # smaller dPAPE (= better) goes UP

# Default operating REGION: spread of dWMAPE over the 5 algos at alpha=1.0.
x10 = np.array([(_load(a)[2][:, i1] - _load(a)[2][:, i0]).mean() for a in COLORS_F14])
ax.axvspan(x10.min(), x10.max(), color="#444444", alpha=0.12, zorder=0)
ax.text(x10.mean(), 0.82, r"default $\alpha$=1.0", transform=ax.get_xaxis_transform(),
        va="top", ha="center", fontsize=15, color="#333333")

# Sweet-spot annotations on the FedAvg curve (y inverted: better = up).
_, pA, wA = _load("FedAvg")
dpA = (pA - pA[:, [i0]]).mean(0); dwA = (wA - wA[:, [i0]]).mean(0)
i05 = int(np.argmin(np.abs(alphas - 0.5))); i20 = int(np.argmin(np.abs(alphas - 2.0)))
ax.annotate("WMAPE-near-zero-cost\n" + r"($\alpha=0.5$)",
            xy=(dwA[i05], dpA[i05]), xytext=(0.6, -3.0),
            fontsize=15, color="#222222", ha="left",
            arrowprops=dict(arrowstyle="->", color="#444444", lw=1.1))
ax.annotate("PAPE-aggressive\n" + r"($\alpha=2.0$)",
            xy=(dwA[i20], dpA[i20]), xytext=(1.2, -10.0),
            fontsize=15, color="#222222", ha="left",
            arrowprops=dict(arrowstyle="->", color="#444444", lw=1.1))

ax.set_xlabel(r"$\Delta$WMAPE  (percentage points; positive = WMAPE worse with codebook)")
ax.set_ylabel(r"$\Delta$PAPE  (%; negative = PAPE better with codebook)")
ax.set_title(r"Codebook correction strength $\alpha$ — PAPE vs WMAPE Pareto")
ax.grid(True, alpha=0.3)

alpha_handles = [
    plt.Line2D([0], [0], marker=MARKER_FOR_ALPHA[a], linestyle="",
               color="#444444", markersize=10, markeredgecolor="black",
               markeredgewidth=0.7, label=fr"$\alpha = {a}$")
    for a in MARKER_ALPHAS
]
cell_handles = [plt.Line2D([0], [0], color=COLORS_F14[c], linewidth=2.5, label=c)
                for c in COLORS_F14]
leg_alpha = ax.legend(handles=alpha_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                      frameon=True, title="Operating point", fontsize=14, title_fontsize=15)
ax.add_artist(leg_alpha)
ax.legend(handles=cell_handles, loc="upper left", bbox_to_anchor=(1.02, 0.55),
          frameon=True, fontsize=14, title="FL algorithm", title_fontsize=15)

fig.tight_layout()
fig.subplots_adjust(right=0.80)
p = OUT / "fig14-2_alpha_sweep_WMAPE.png"
fig.savefig(p, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"saved: {p}")

print("\nfinal-round effect (after a=1.0 - before a=0), 3-seed mean:")
for i, a in enumerate(ALGOS):
    print(f"  {a:9s} dPAPE={pap_m[i]:+.2f}pp   dWMAPE={wm_m[i]:+.2f}pp")
