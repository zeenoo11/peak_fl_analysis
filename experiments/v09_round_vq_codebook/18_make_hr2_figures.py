"""HR@2 variants of fig4 (peak gain vs MAE cost) and fig8 (baseline vs corrected).

Same data source and styling as 12_make_presentation_figures.py, but the peak
axis uses HR@2 (peak-timing hit rate, ±2h tolerance, %) instead of PAPE.

Key difference from the PAPE story: HR@2 is a *hit rate* (higher = better) and
the CMO offset corrects peak *magnitude*, not peak *timing*. From round 2 on the
codebook leaves HR@2 flat / slightly lower, so these figures are honest "the
benefit does NOT transfer to timing" companions, not improvement plots.

Reads : outputs/v09_round_vq_codebook_R20_MAEonly/seed{42,123,7}/V9-RoundCB-*/codebook_log.jsonl
Writes: papers/conference_draft/figures/fig{4,8}_*_hr2.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

matplotlib.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parents[2]
NS = "v09_round_vq_codebook_R20_MAEonly"
SEEDS = (42, 123, 7)
OUT = BASE / "papers/conference_draft/figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "FedAvg":   "#0072B2",
    "FedProx":  "#D55E00",
    "FedRep":   "#009E73",
    "Ditto":    "#CC79A7",
    "FedProto": "#56B4E9",
}
HR_C, MAE_C = "#0072B2", "#E69F00"


def load():
    """{algo: (rounds, hr_before[seed,R], hr_after[seed,R], mae_before, mae_after)}."""
    root = BASE / "outputs" / NS
    data = {}
    for algo in COLORS:
        bef, aft, mb, ma, rounds = [], [], [], [], None
        for s in SEEDS:
            rows = [json.loads(l) for l in
                    (root / f"seed{s}" / f"V9-RoundCB-{algo}" / "codebook_log.jsonl").open()]
            rounds = [r["round"] for r in rows]
            bef.append([r["test_before"]["hr@2_mean"] for r in rows])
            aft.append([r["test_after"]["hr@2_mean"] for r in rows])
            mb.append([r["test_before"]["mae_mean"] for r in rows])
            ma.append([r["test_after"]["mae_mean"] for r in rows])
        data[algo] = (rounds, np.array(bef), np.array(aft), np.array(mb), np.array(ma))
    return data


DATA = load()
ROUNDS = DATA["FedAvg"][0]
NR = max(ROUNDS)
XT = [t for t in (1, 5, 10, 15, 20) if t <= NR]
algos = list(COLORS)


def _align_zero(ax_main, ax_sec):
    l1, h1 = ax_main.get_ylim()
    f = (0 - l1) / (h1 - l1)
    t = max(ax_sec.get_ylim()[1], 1e-3)
    b = -t * f / (1 - f) if f < 1 else ax_sec.get_ylim()[0]
    ax_sec.set_ylim(b, t)


# ---------------------------------------------------------------------------
# Fig 4 (HR@2) — Peak-timing change vs average-error cost, final round.
#   Left  axis: ΔHR@2 (percentage points; positive = better timing).
#   Right axis: MAE change (relative %; positive = more average error).
# ---------------------------------------------------------------------------
hb = np.array([DATA[a][1] for a in algos])      # [algo, seed, R]
ha = np.array([DATA[a][2] for a in algos])
mbn = np.array([DATA[a][3] for a in algos])
man = np.array([DATA[a][4] for a in algos])
hr_eff = ha - hb                                # HR@2: percentage points
mae_eff = (man - mbn) / mbn * 100.0             # MAE: relative %

hm_f, hs_f = hr_eff[:, :, -1].mean(1), hr_eff[:, :, -1].std(1, ddof=1)
mm_f, ms_f = mae_eff[:, :, -1].mean(1), mae_eff[:, :, -1].std(1, ddof=1)
x = np.arange(len(algos))
w = 0.38

fig, ax = plt.subplots(figsize=(8.4, 5.0))
ax.axhline(0, color="#888", lw=1.0)
ax2 = ax.twinx()
b1 = ax.bar(x - w / 2, hm_f, w, yerr=hs_f, color=HR_C, capsize=3,
            label="Peak timing (HR@2) — left axis, Δ points")
b2 = ax2.bar(x + w / 2, mm_f, w, yerr=ms_f, color=MAE_C, capsize=3,
             label="Average error (MAE) — right axis, relative %")
for i in range(len(algos)):
    ax.text(i - w / 2, hm_f[i] - 0.06, f"{hm_f[i]:+.1f}", ha="center", va="top",
            fontsize=9.5, color=HR_C)
    ax2.text(i + w / 2, mm_f[i] + 0.05, f"{mm_f[i]:+.1f}%", ha="center", va="bottom",
             fontsize=9.5, color="#9a6a00")
ax.set_xticks(x)
ax.set_xticklabels(algos)
ax.set_ylabel("Peak-timing change ΔHR@2 (percentage points)", color=HR_C)
ax2.set_ylabel("Average error change (relative %)", color="#9a6a00")
ax.tick_params(axis="y", labelcolor=HR_C)
ax2.tick_params(axis="y", labelcolor="#9a6a00")
lo = min(float((hm_f - hs_f).min()) - 0.5, -0.5)
ax.set_ylim(lo, 1.0)
ax2.set_ylim(0, max(2.0, float((mm_f + ms_f).max()) + 0.4))
_align_zero(ax, ax2)
ax.set_title("Codebook Does Not Improve Peak Timing\n"
             "Final-round effect per algorithm (HR@2 is higher-is-better)",
             fontsize=12, fontweight="bold")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(handles=[b1, b2], loc="lower right", frameon=True, framealpha=0.9,
          facecolor="white", edgecolor="none", fontsize=10.5)
fig.tight_layout()
p4 = OUT / "fig4_peak_gain_vs_mae_cost_hr2.png"
fig.savefig(p4, dpi=160)
plt.close(fig)
print(f"saved: {p4}")


# ---------------------------------------------------------------------------
# Fig 8 (HR@2) — federated baseline (solid) vs codebook-corrected (dashed).
#   HR@2 is higher-is-better; from round 2 on the dashed (corrected) curves sit
#   below the solid baselines — the codebook does not lift peak timing.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 5.6))
for algo, color in COLORS.items():
    _, bef, aft, _, _ = DATA[algo]
    ax.plot(ROUNDS, bef.mean(0), "-o", color=color, lw=1.8, markersize=3.5)
    ax.plot(ROUNDS, aft.mean(0), "--o", color=color, lw=2.2, markersize=4.0)
ax.set_xlabel("Communication round")
ax.set_ylabel("Peak-timing Hit Rate HR@2 (%)   —   higher is better")
ax.set_title("Codebook Helps Peak Size, Not Peak Timing\n"
             "HR@2: baseline (solid) vs corrected (dashed)",
             fontsize=12, fontweight="bold")
ax.set_xticks(XT)
ax.grid(True, alpha=0.3)

handles = [
    Line2D([0], [0], color="#444", lw=2.0, ls="-", label="Federated baseline (no codebook)"),
    Line2D([0], [0], color="#444", lw=2.4, ls="--", label="global codebook correction"),
] + [Line2D([0], [0], color=c, ls="none", marker="o", markersize=7, label=a)
     for a, c in COLORS.items()]
ax.legend(handles=handles, loc="lower right", frameon=True,
          framealpha=0.95, facecolor="white", edgecolor="#ccc", fontsize=10)
fig.tight_layout()
p8 = OUT / "fig8_baseline_vs_corrected_hr2.png"
fig.savefig(p8, dpi=160)
plt.close(fig)
print(f"saved: {p8}")


print("\n=== final-round HR@2 summary (MAE-only, 3-seed) ===")
for i, a in enumerate(algos):
    print(f"{a:9s} before {hb[i, :, -1].mean():5.1f}  after {ha[i, :, -1].mean():5.1f}  "
          f"ΔHR@2 {hm_f[i]:+.2f}")
