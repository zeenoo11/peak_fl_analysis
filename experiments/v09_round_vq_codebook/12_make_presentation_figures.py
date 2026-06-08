"""Presentation figures for papers/conference_draft/presentation_final.md.

Re-renders the v09 RoundCB result figures with audience-facing English titles
(no internal jargon: no "MAE-only", "aux=0", "v09", "test_before/after", "E=5").
All numbers come from the MAE-only (lambda_aux=0) run so the codebook effect is
shown in isolation, but that experimental detail is kept OUT of the titles.

Reads : outputs/v09_round_vq_codebook_R20_MAEonly/seed{42,123,7}/V9-RoundCB-*/codebook_log.jsonl
Writes: papers/conference_draft/figures/fig{1..4}_*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parents[2]
NS = "v09_round_vq_codebook_R20_MAEonly"
SEEDS = (42, 123, 7)
OUT = BASE / "papers/conference_draft/figures"
OUT.mkdir(parents=True, exist_ok=True)

# Wong color-blind-friendly palette (consistent with the report figures).
COLORS = {
    "FedAvg":   "#0072B2",
    "FedProx":  "#D55E00",
    "FedRep":   "#009E73",
    "Ditto":    "#CC79A7",
    "FedProto": "#56B4E9",
}
PAPE_C, MAE_C = "#0072B2", "#E69F00"


def load():
    """{algo: (rounds, before[seed, R], after[seed, R], mae_before, mae_after)}."""
    root = BASE / "outputs" / NS
    data = {}
    for algo in COLORS:
        bef, aft, mb, ma, rounds = [], [], [], [], None
        for s in SEEDS:
            rows = [json.loads(l) for l in
                    (root / f"seed{s}" / f"V9-RoundCB-{algo}" / "codebook_log.jsonl").open()]
            rounds = [r["round"] for r in rows]
            bef.append([r["test_before"]["pape_mean"] for r in rows])
            aft.append([r["test_after"]["pape_mean"] for r in rows])
            mb.append([r["test_before"]["mae_mean"] for r in rows])
            ma.append([r["test_after"]["mae_mean"] for r in rows])
        data[algo] = (rounds, np.array(bef), np.array(aft), np.array(mb), np.array(ma))
    return data


DATA = load()
ROUNDS = DATA["FedAvg"][0]
NR = max(ROUNDS)
XT = [t for t in (1, 5, 10, 15, 20) if t <= NR]


def _legend(ax, **kw):
    kw.setdefault("fontsize", 11)
    ax.legend(frameon=True, framealpha=0.9, facecolor="white",
              edgecolor="none", **kw)


# ---------------------------------------------------------------------------
# Fig 1 — Federated baseline (no codebook): per-round peak error, 5 algorithms.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.0, 5.0))
for algo, color in COLORS.items():
    _, bef, _, _, _ = DATA[algo]
    m, sd = bef.mean(0), bef.std(0, ddof=1)
    ax.plot(ROUNDS, m, "-o", color=color, lw=2.0, markersize=4.5, label=algo)
    ax.fill_between(ROUNDS, m - sd, m + sd, color=color, alpha=0.15)
ax.set_xlabel("Communication round")
ax.set_ylabel("Peak Absolute Percentage Error (%)")
ax.set_title("Federated Baseline: Peak Error per Round\n"
             "Five algorithms converge within ~1 point",
             fontsize=12, fontweight="bold")
ax.set_xticks(XT)
ax.grid(True, alpha=0.3)
_legend(ax, loc="upper right")
fig.tight_layout()
p1 = OUT / "fig1_fl_baseline_pape.png"
fig.savefig(p1, dpi=160)
plt.close(fig)
print(f"saved: {p1}")


# ---------------------------------------------------------------------------
# Fig 2 — With global codebook correction: per-round peak error, 5 algorithms.
# ---------------------------------------------------------------------------
half = NR // 2
conv = np.concatenate([
    DATA[a][1][:, [i for i, r in enumerate(ROUNDS) if r > half]].ravel()
    for a in COLORS
])
conv_mean = float(conv.mean())

fig, ax = plt.subplots(figsize=(8.0, 5.0))
ax.axhline(conv_mean, color="red", ls="--", lw=1.6,
           label=f"Federated baseline (~{conv_mean:.0f}%)")
for algo, color in COLORS.items():
    _, _, aft, _, _ = DATA[algo]
    ax.plot(ROUNDS, aft.mean(0), "-o", color=color, lw=2.0, markersize=4.5, label=algo)
after_min = min(DATA[a][2].mean(0).min() for a in COLORS)
ax.set_ylim(after_min - 0.5, conv_mean + 1.5)
ax.set_xlabel("Communication round")
ax.set_ylabel("Peak Absolute Percentage Error (%)")
ax.set_title("Codebook Correction: Peak Error per Round\n"
             "All algorithms drop below baseline",
             fontsize=12, fontweight="bold")
ax.set_xticks(XT)
ax.grid(True, alpha=0.3)
_legend(ax, loc="upper right", fontsize=10)
fig.tight_layout()
p2 = OUT / "fig2_codebook_corrected_pape.png"
fig.savefig(p2, dpi=160)
plt.close(fig)
print(f"saved: {p2}")


# ---------------------------------------------------------------------------
# Fig 3 — Codebook lift: final-round peak error, before vs after, per algorithm.
# ---------------------------------------------------------------------------
algos = list(COLORS)
mb = np.array([DATA[a][1][:, -1].mean() for a in algos])
sb = np.array([DATA[a][1][:, -1].std(ddof=1) for a in algos])
ma = np.array([DATA[a][2][:, -1].mean() for a in algos])
sa = np.array([DATA[a][2][:, -1].std(ddof=1) for a in algos])
lift = ma - mb

x = np.arange(len(algos))
width = 0.36
y_lo = float(np.floor(np.min(ma - sa) - 2.0))
y_hi = float(np.ceil(np.max(mb + sb) + 4.0))
off = (y_hi - y_lo) * 0.04

with plt.rc_context({"font.size": 13}):
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.set_ylim(y_lo, y_hi)
    for i, a in enumerate(algos):
        col = COLORS[a]
        ax.bar(x[i] - width / 2, mb[i], width=width, yerr=sb[i],
               color=col, alpha=0.35, edgecolor="black", linewidth=0.6,
               error_kw={"elinewidth": 1.0, "capsize": 3.0, "ecolor": "#555555"},
               label="Federated baseline" if i == 0 else None)
        ax.bar(x[i] + width / 2, ma[i], width=width, yerr=sa[i],
               color=col, alpha=0.95, edgecolor="black", linewidth=0.6,
               error_kw={"elinewidth": 1.0, "capsize": 3.0, "ecolor": "#222222"},
               label="+ global codebook correction" if i == 0 else None)
        top = max(mb[i] + sb[i], ma[i] + sa[i])
        ax.text(x[i], top + off, f"Δ −{abs(lift[i]):.1f}",
                ha="center", va="bottom", fontsize=11, color="black",
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=11)
    ax.set_ylabel("Peak Absolute Percentage Error (%)")
    ax.set_title("Codebook Lift on Peak Error\n"
                 "Final round: before vs after",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.12),
              frameon=False, fontsize=11, ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    p3 = OUT / "fig3_codebook_lift.png"
    fig.savefig(p3, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {p3}")


# ---------------------------------------------------------------------------
# Fig 4 — Peak gain vs average-error cost: per-round dual axis (mean over algos).
# ---------------------------------------------------------------------------
pb = np.array([DATA[a][1] for a in algos])      # [algo, seed, R]
pa = np.array([DATA[a][2] for a in algos])
mbn = np.array([DATA[a][3] for a in algos])
man = np.array([DATA[a][4] for a in algos])
pap_eff = pa - pb                               # PAPE: absolute percentage-point
mae_eff = (man - mbn) / mbn * 100.0             # MAE: relative %


def _align_zero(ax_main, ax_sec):
    l1, h1 = ax_main.get_ylim()
    f = (0 - l1) / (h1 - l1)
    t = max(ax_sec.get_ylim()[1], 1e-3)
    b = -t * f / (1 - f) if f < 1 else ax_sec.get_ylim()[0]
    ax_sec.set_ylim(b, t)


# Final-round per-algorithm effect: peak error drop (Δ pp) vs average error (rel %).
pm_f, ps_f = pap_eff[:, :, -1].mean(1), pap_eff[:, :, -1].std(1, ddof=1)
mm_f, ms_f = mae_eff[:, :, -1].mean(1), mae_eff[:, :, -1].std(1, ddof=1)
x = np.arange(len(algos))
w = 0.38

fig, ax = plt.subplots(figsize=(8.4, 5.0))
ax.axhline(0, color="#888", lw=1.0)
ax2 = ax.twinx()
b1 = ax.bar(x - w / 2, pm_f, w, yerr=ps_f, color=PAPE_C, capsize=3,
            label="Peak error (PAPE) — left axis, Δ points")
b2 = ax2.bar(x + w / 2, mm_f, w, yerr=ms_f, color=MAE_C, capsize=3,
             label="Average error (MAE) — right axis, relative %")
for i in range(len(algos)):
    ax.text(i - w / 2, pm_f[i] - 0.18, f"{pm_f[i]:+.1f}", ha="center", va="top",
            fontsize=9.5, color=PAPE_C)
    ax2.text(i + w / 2, mm_f[i] + 0.05, f"{mm_f[i]:+.1f}%", ha="center", va="bottom",
             fontsize=9.5, color="#9a6a00")
ax.set_xticks(x)
ax.set_xticklabels(algos)
ax.set_ylabel("Peak error change (Δ percentage points)", color=PAPE_C)
ax2.set_ylabel("Average error change (relative %)", color="#9a6a00")
ax.tick_params(axis="y", labelcolor=PAPE_C)
ax2.tick_params(axis="y", labelcolor="#9a6a00")
ax.set_ylim(float((pm_f - ps_f).min()) - 1.0, 1.5)
ax2.set_ylim(0, max(2.0, float((mm_f + ms_f).max()) + 0.4))
_align_zero(ax, ax2)
ax.set_title("Large Peak Gain, Negligible Average Cost\n"
             "Final-round effect per algorithm",
             fontsize=12, fontweight="bold")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(handles=[b1, b2], loc="lower right", frameon=True, framealpha=0.9,
          facecolor="white", edgecolor="none", fontsize=10.5)
fig.tight_layout()
p4 = OUT / "fig4_peak_gain_vs_mae_cost.png"
fig.savefig(p4, dpi=160)
plt.close(fig)
print(f"saved: {p4}")

# ---------------------------------------------------------------------------
# Fig 5 — Per-round codebook effect (mean over five algorithms): peak error
#         drops sharply, average error barely moves. Dual axis, zero-aligned.
# ---------------------------------------------------------------------------
def _agg(eff):
    per_seed = eff.mean(axis=0)            # [seed, R]
    return per_seed.mean(0), per_seed.std(0, ddof=1)


pm, ps = _agg(pap_eff)
mm, ms = _agg(mae_eff)

fig, ax = plt.subplots(figsize=(8.4, 5.0))
ax.axhline(0, color="#888", lw=1.0)
ax2 = ax.twinx()
l1, = ax.plot(ROUNDS, pm, "-o", color=PAPE_C, lw=2.2, ms=4,
              label="Peak error (PAPE) — left axis, Δ points")
ax.fill_between(ROUNDS, pm - ps, pm + ps, color=PAPE_C, alpha=0.15)
l2, = ax2.plot(ROUNDS, mm, "-o", color=MAE_C, lw=2.2, ms=4,
               label="Average error (MAE) — right axis, relative %")
ax2.fill_between(ROUNDS, mm - ms, mm + ms, color=MAE_C, alpha=0.15)
ax.set_xlabel("Communication round")
ax.set_ylabel("Peak error change (Δ percentage points)", color=PAPE_C)
ax2.set_ylabel("Average error change (relative %)", color="#9a6a00")
ax.tick_params(axis="y", labelcolor=PAPE_C)
ax2.tick_params(axis="y", labelcolor="#9a6a00")
ax2.axhspan(-2, 2, color="gray", alpha=0.07)
ax2.text(NR, 1.0, "±2% band", ha="right", va="center", fontsize=9, color="#888")
ax.set_title("Per Round: Large Peak Gain, Tiny Cost\n"
             "Averaged over five FL algorithms",
             fontsize=12, fontweight="bold")
ax.set_xticks(XT)
ax.grid(True, alpha=0.3)
mhi = max(3.0, float((mm + ms).max()) + 1.0)
ax2.set_ylim(-mhi, mhi)
_align_zero(ax, ax2)
ax.legend(handles=[l1, l2], loc="center right", frameon=True, framealpha=0.9,
          facecolor="white", edgecolor="none", fontsize=10.5)
fig.tight_layout()
p5 = OUT / "fig5_codebook_effect_per_round.png"
fig.savefig(p5, dpi=160)
plt.close(fig)
print(f"saved: {p5}")


# ---------------------------------------------------------------------------
# Fig 6 — Codebook effect vs offset strength per round (mean over five algos).
#         Panel-D of the evolution figure, but averaged across algorithms:
#         ΔPAPE (negative = codebook helps) + mean CMO offset L2 norm.
# ---------------------------------------------------------------------------
import torch  # noqa: E402  (only needed for the codebook_history snapshots)

root = BASE / "outputs" / NS
off_curves = []   # [algo*seed, R]  mean-over-cluster offset L2 norm per round
for algo in COLORS:
    for s in SEEDS:
        hist = torch.load(root / f"seed{s}" / f"V9-RoundCB-{algo}" / "codebook_history.pt",
                          map_location="cpu")
        offsets = hist["offsets"].float().numpy()          # (R, M, H)
        off_curves.append(np.linalg.norm(offsets, axis=2).mean(axis=1))   # (R,)
off_curves = np.array(off_curves)
off_m, off_s = off_curves.mean(0), off_curves.std(0, ddof=1)

# ΔPAPE per round: pap_eff is [algo, seed, R]; mean/std across algo*seed.
dp_flat = pap_eff.reshape(-1, pap_eff.shape[-1])
dp_m, dp_s = dp_flat.mean(0), dp_flat.std(0, ddof=1)

DP_C, OFF_C = "#E69F00", "#8c564b"
fig, ax = plt.subplots(figsize=(8.4, 5.0))
ax.axhline(0, ls="--", color="gray", alpha=0.6, lw=1.0)
ld, = ax.plot(ROUNDS, dp_m, "-o", color=DP_C, lw=2.2, ms=4,
              label="Peak-error change ΔPAPE (negative = codebook helps)")
ax.fill_between(ROUNDS, dp_m - dp_s, dp_m + dp_s, color=DP_C, alpha=0.15)
ax.set_ylabel("ΔPAPE  (after − before)", color="#9a6a00")
ax.tick_params(axis="y", labelcolor="#9a6a00")
ax2 = ax.twinx()
lo, = ax2.plot(ROUNDS, off_m, "-^", color=OFF_C, lw=2.0, ms=4,
               label="Correction strength (mean offset L2 norm)")
ax2.fill_between(ROUNDS, off_m - off_s, off_m + off_s, color=OFF_C, alpha=0.15)
ax2.set_ylabel("Mean correction-offset magnitude", color=OFF_C)
ax2.tick_params(axis="y", labelcolor=OFF_C)
ax.set_xlabel("Communication round")
ax.set_xticks(XT)
ax.grid(True, alpha=0.3)
ax.set_title("Correction Settles as Offsets Shrink\n"
             "Averaged over five FL algorithms",
             fontsize=12, fontweight="bold")
ax.legend(handles=[ld, lo], loc="center right", frameon=True, framealpha=0.9,
          facecolor="white", edgecolor="none", fontsize=10.5)
fig.tight_layout()
p6 = OUT / "fig6_codebook_effect_vs_offset.png"
fig.savefig(p6, dpi=160)
plt.close(fig)
print(f"saved: {p6}")


# ---------------------------------------------------------------------------
# Fig 8 — Combined: federated baseline (solid) vs codebook-corrected (dashed),
#         one color per algorithm, on a single axis. Merges fig1 + fig2 so the
#         downward shift from the codebook is visible as one cluster sliding down.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 5.6))
for algo, color in COLORS.items():
    _, bef, aft, _, _ = DATA[algo]
    ax.plot(ROUNDS, bef.mean(0), "-o", color=color, lw=1.8, markersize=3.5)
    ax.plot(ROUNDS, aft.mean(0), "--o", color=color, lw=2.2, markersize=4.0)
ax.set_xlabel("Communication round")
ax.set_ylabel("Peak Absolute Percentage Error (%)")
ax.set_title("Codebook Shifts Every Algorithm Down\n"
             "Baseline (solid) vs corrected (dashed)",
             fontsize=12, fontweight="bold")
ax.set_xticks(XT)
ax.grid(True, alpha=0.3)

# Single legend: gray solid/dashed convey the correction (line style), colored
# marker-only swatches convey the algorithm (color). No titles, no '+'.
handles = [
    Line2D([0], [0], color="#444", lw=2.0, ls="-", label="Federated baseline (no codebook)"),
    Line2D([0], [0], color="#444", lw=2.4, ls="--", label="global codebook correction"),
] + [Line2D([0], [0], color=c, ls="none", marker="o", markersize=7, label=a)
     for a, c in COLORS.items()]
ax.legend(handles=handles, loc="upper right", frameon=True,
          framealpha=0.95, facecolor="white", edgecolor="#ccc", fontsize=10)
fig.tight_layout()
p8 = OUT / "fig8_baseline_vs_corrected.png"
fig.savefig(p8, dpi=160)
plt.close(fig)
print(f"saved: {p8}")


# ---------------------------------------------------------------------------
# Fig 9 — Global-baseline comparison (ranked horizontal bars).
#   Where RoundCB stands vs the best zero-shot foundation model and the
#   privacy-free centralized upper bound (same NBEATSx backbone).
#   All numbers use the v09 per-client protocol (114 households, test split,
#   per-client-mean PAPE) EXCEPT centralized NBEATSx (v06, E=40 — see note).
# ---------------------------------------------------------------------------
def _read_pape(ns, cell, seeds=SEEDS):
    vals = []
    for s in seeds:
        p = BASE / "outputs" / ns / f"seed{s}" / cell / "result.json"
        if p.exists():
            vals.append(json.load(open(p))["test_terminal"]["pape_mean"])
    a = np.array(vals)
    return float(a.mean()), float(a.std(ddof=1) if len(a) > 1 else 0.0)

V09 = "v09_round_vq_codebook"
ch_m, ch_s = _read_pape(V09, "fm_chronos_bolt_small")
tf_m, tf_s = _read_pape(V09, "fm_timesfm")
dl_m, dl_s = _read_pape(V09, "nf_dlinear")
# Centralized NBEATSx: v06 (E=40) pooled-SGD upper bound — same backbone as ours.
cn_m, cn_s = 49.43, 0.36

before_all = np.concatenate([DATA[a][1][:, -1] for a in COLORS])   # 15 runs
after_all = np.concatenate([DATA[a][2][:, -1] for a in COLORS])
fl_m, fl_s = float(before_all.mean()), float(before_all.std(ddof=1))
rc_m, rc_s = float(after_all.mean()), float(after_all.std(ddof=1))

CAT = {"foundation": "#7B7B7B", "centralized": "#E69F00",
       "ours_base": "#9ecae1", "ours_main": "#0072B2"}
# (label, mean, std, category, is_ours_main)
items = [
    ("Chronos-Bolt-small (zero-shot)", ch_m, ch_s, "foundation", False),
    ("TimesFM (zero-shot)", tf_m, tf_s, "foundation", False),
    ("DLinear (centralized)", dl_m, dl_s, "centralized", False),
    ("NBEATSx (centralized) †", cn_m, cn_s, "centralized", False),
    ("FL baseline (ours)", fl_m, fl_s, "ours_base", False),
    ("RoundCB (ours)", rc_m, rc_s, "ours_main", True),
]
items.sort(key=lambda t: t[1])             # ascending PAPE (best first)

with plt.rc_context({"font.size": 12}):
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ypos = np.arange(len(items))
    for y, (lbl, m, s, cat, main) in zip(ypos, items):
        ax.barh(y, m, xerr=s, height=0.62, color=CAT[cat],
                edgecolor="black", linewidth=1.4 if main else 0.5,
                error_kw={"elinewidth": 1.0, "capsize": 3.0, "ecolor": "#333"},
                zorder=3)
        ax.text(m + s + 0.12, y, f"{m:.1f}", va="center", ha="left",
                fontsize=11, fontweight="bold" if main else "normal")
    # RoundCB reference line: shows it sits left of (better than) centralized.
    ax.axvline(rc_m, color=CAT["ours_main"], ls="--", lw=1.3, alpha=0.7, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([t[0] for t in items])
    ax.set_xlabel("Peak Absolute Percentage Error (%)   —   lower is better")
    ax.set_xlim(44, 54)
    ax.invert_yaxis()                       # best on top
    ax.set_title("RoundCB vs Global Baselines\n"
                 "Federated correction matches non-federated models",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, zorder=0)
    legend_handles = [
        Patch(facecolor=CAT["foundation"], edgecolor="black", label="Foundation model (zero-shot)"),
        Patch(facecolor=CAT["centralized"], edgecolor="black", label="Centralized (pooled, privacy-free)"),
        Patch(facecolor=CAT["ours_main"], edgecolor="black", label="Ours (federated)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True,
              framealpha=0.95, facecolor="white", edgecolor="#ccc", fontsize=10)
    fig.text(0.01, -0.02,
             "† Centralized NBEATSx measured under the v06 budget (E=40), not the "
             "R=20 federated budget — training-amount caveat.",
             fontsize=8.5, color="#666", ha="left")
    fig.tight_layout()
    p9 = OUT / "fig9_global_baseline_comparison.png"
    fig.savefig(p9, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {p9}")


# ---------------------------------------------------------------------------
# Fig 10 — Trajectory vs global reference lines.
#   Per-round codebook-corrected curve for EACH FL algorithm (codebook only)
#   against horizontal global-baseline references. Shows every federated
#   algorithm, once corrected, crossing below the centralized upper bound
#   toward the best foundation-model level.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.6, 6.0))

# Reference lines: the two anchors only — best zero-shot foundation model and
# the same-backbone centralized upper bound. Thick + large.
ax.axhline(ch_m, color="#5a5a5a", ls="--", lw=2.8, zorder=1)
ax.text(NR, ch_m - 0.12, f"Chronos-Bolt-small (Zero-shot)  {ch_m:.1f}", color="#3a3a3a",
        fontsize=13, fontweight="bold", va="top", ha="right")
ax.axhline(cn_m, color="#E69F00", ls="--", lw=2.8, zorder=1)
ax.text(NR, cn_m + 0.10, f"NBEATSx (Centralised)  {cn_m:.1f}", color="#9a6a00",
        fontsize=13, fontweight="bold", va="bottom", ha="right")

# Codebook-corrected curve per FL algorithm.
for algo, color in COLORS.items():
    ax.plot(ROUNDS, DATA[algo][2].mean(0), "-o", color=color, lw=2.6,
            markersize=5, label=algo, zorder=4)

ax.set_xlabel("Communication round", fontsize=12)
ax.set_ylabel("Peak Absolute Percentage Error (%)", fontsize=12)
ax.set_title("Corrected FL Crosses the Centralized Bound\n"
             "Codebook curves vs global references",
             fontsize=13, fontweight="bold")
ax.set_xticks(XT)
ax.set_ylim(45.8, 51.6)
ax.tick_params(labelsize=11)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper center", frameon=True, framealpha=0.95, facecolor="white",
          edgecolor="#ccc", fontsize=10, ncol=5, title="Codebook-corrected (per FL algorithm)",
          columnspacing=1.0, handletextpad=0.4)
fig.tight_layout()
p10 = OUT / "fig10_trajectory_vs_baselines.png"
fig.savefig(p10, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"saved: {p10}")


# ---------------------------------------------------------------------------
# Fig 11 — Lollipop dot plot (ranked) on a single PAPE axis.
#   Same data as fig9 but stems+dots: honest value comparison, compact.
# ---------------------------------------------------------------------------
with plt.rc_context({"font.size": 12}):
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ypos = np.arange(len(items))
    for y, (lbl, m, s, cat, main) in zip(ypos, items):
        ax.hlines(y, 44, m, color=CAT[cat], lw=2.5 if main else 1.8,
                  alpha=0.9, zorder=2)
        ax.errorbar(m, y, xerr=s, fmt="o", color=CAT[cat],
                    markersize=13 if main else 9,
                    markeredgecolor="black", markeredgewidth=1.6 if main else 0.6,
                    ecolor="#333", elinewidth=1.0, capsize=3.0, zorder=3)
        ax.text(m + s + 0.12, y, f"{m:.1f}", va="center", ha="left",
                fontsize=11, fontweight="bold" if main else "normal")
    ax.axvline(rc_m, color=CAT["ours_main"], ls="--", lw=1.3, alpha=0.6, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([t[0] for t in items])
    ax.set_xlabel("Peak Absolute Percentage Error (%)   —   lower is better")
    ax.set_xlim(44, 54)
    ax.invert_yaxis()
    ax.set_title("RoundCB vs Global Baselines\n"
                 "Federated correction matches non-federated models",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, zorder=0)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CAT["foundation"],
               markeredgecolor="black", markersize=10, label="Foundation model (zero-shot)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CAT["centralized"],
               markeredgecolor="black", markersize=10, label="Centralized (pooled, privacy-free)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CAT["ours_main"],
               markeredgecolor="black", markersize=10, label="Ours (federated)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True,
              framealpha=0.95, facecolor="white", edgecolor="#ccc", fontsize=10)
    fig.text(0.01, -0.02,
             "† Centralized NBEATSx measured under the v06 budget (E=40), not the "
             "R=20 federated budget — training-amount caveat.",
             fontsize=8.5, color="#666", ha="left")
    fig.tight_layout()
    p11 = OUT / "fig11_global_baseline_lollipop.png"
    fig.savefig(p11, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {p11}")


# ---------------------------------------------------------------------------
# Fig 12 — Codebook is communication-light vs the FL weight upload (linear).
#   Per-client/round upload for each FL algorithm (full weights, with FedRep
#   encoder-only and FedProto +prototype refinements — matching src/fl/round_aux.py)
#   next to the codebook add-on. Linear scale → the codebook bar is a visible
#   sliver, making "negligible" intuitive. Single panel.
# ---------------------------------------------------------------------------
import sys as _sys  # noqa: E402
_src = str(BASE / "src")
if _src not in _sys.path:
    _sys.path.insert(0, _src)
from fl.fedavg_aux import init_backbone_aux  # noqa: E402

_B, _D, _M, _H, _Kloc = 4, 64, 32, 24, 2
_sd = init_backbone_aux(42).state_dict()


def _is_head(n):                              # FedRep head = forecast proj + aux head
    return (".proj." in n) or n.startswith("aux_head.")


_full = sum(v.numel() * v.element_size() for v in _sd.values())
_enc = sum(v.numel() * v.element_size() for k, v in _sd.items() if not _is_head(k))
_proto = _M * _D * _B                         # FedProto extra prototype payload
cb_pc = _Kloc * (_D + 1) * _B + _M * (_H + 1) * _B   # codebook add-on / client / round

# Per-client/round upload bytes per FL algorithm (src/fl/round_aux.py semantics).
ALGO_UP = {
    "FedAvg": _full, "FedProx": _full, "FedRep": _enc,
    "Ditto": _full, "FedProto": _full + _proto,
}
overhead = cb_pc / _full * 100.0              # codebook vs a full weight update

algos_c = list(ALGO_UP)
weight_kb = np.array([ALGO_UP[a] / 1024.0 for a in algos_c])
cb_kb = cb_pc / 1024.0
C_CB = "#D62728"

fig, ax = plt.subplots(figsize=(9.6, 5.6))
x = np.arange(len(algos_c))
# FL weight upload per algorithm (shared palette, lightened).
ax.bar(x, weight_kb, width=0.62, color=[COLORS[a] for a in algos_c], alpha=0.5,
       edgecolor="black", linewidth=0.7, zorder=3)
for xi, w in zip(x, weight_kb):
    ax.text(xi, w * 1.04, f"{w:.1f} KB", ha="center", va="bottom",
            fontsize=11, fontweight="bold")
# Codebook magnitude as a dotted reference line (fig10 baseline style); log
# scale makes the 3.6 KB line clearly separated from the ~280 KB weight bars.
ax.axhline(cb_kb, color=C_CB, ls=":", lw=2.6, zorder=2)
ax.text(len(algos_c) - 0.5, cb_kb * 1.15, f"Codebook  {cb_kb:.1f} KB", color=C_CB,
        fontsize=13, fontweight="bold", va="bottom", ha="right")
ax.set_xticks(x)
ax.set_xticklabels(algos_c, fontsize=12)
ax.set_ylabel("Upload per client per round  (KB, log scale)", fontsize=12)
ax.set_yscale("log")
ax.set_ylim(1, 700)
ax.set_title("Codebook Adds Negligible Communication\n"
             "Per-client upload per round (log scale)",
             fontsize=12.5, fontweight="bold")
ax.grid(True, axis="y", which="both", alpha=0.3, zorder=0)
fig.tight_layout()
p12 = OUT / "fig12_communication_cost.png"
fig.savefig(p12, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"saved: {p12}")


# ---------------------------------------------------------------------------
# Fig 13 — Peak error vs cumulative communication (FL-standard efficiency curve).
#   x = cumulative uplink per client (MB, log); y = PAPE. One marker per round.
#   FL baseline (before) vs RoundCB (after), mean over 5 algos. RoundCB reaches
#   a lower PAPE at essentially the same communication budget (codebook +1.3%).
# ---------------------------------------------------------------------------
bef_stack = np.array([DATA[a][1] for a in COLORS])     # [algo, seed, R]
aft_stack = np.array([DATA[a][2] for a in COLORS])
y_base = bef_stack.mean((0, 1))                        # PAPE per round
y_cb = aft_stack.mean((0, 1))
bb_mean = float(np.mean([ALGO_UP[a] for a in algos_c]))   # bytes / client / round
rds = np.arange(1, NR + 1)
x_base = bb_mean * rds / 1e6                            # cumulative MB / client
x_cb = (bb_mean + cb_pc) * rds / 1e6

fig, ax = plt.subplots(figsize=(9.0, 5.8))
ax.plot(x_base, y_base, "-o", color="#9ecae1", lw=2.6, ms=5,
        label="Standard FL (no codebook)", zorder=3)
ax.plot(x_cb, y_cb, "-o", color="#0072B2", lw=2.6, ms=5,
        label="RoundCB (federated codebook)", zorder=4)
# Final-budget gap annotation.
ax.annotate(f"{y_base[-1]:.1f}", xy=(x_base[-1], y_base[-1]), xytext=(8, 4),
            textcoords="offset points", fontsize=11, fontweight="bold", color="#5a8fb5")
ax.annotate(f"{y_cb[-1]:.1f}", xy=(x_cb[-1], y_cb[-1]), xytext=(8, -4),
            textcoords="offset points", fontsize=11, fontweight="bold", color="#0072B2",
            va="top")
ax.annotate("", xy=(x_cb[-1], y_cb[-1] + 0.2), xytext=(x_base[-1], y_base[-1] - 0.2),
            arrowprops=dict(arrowstyle="->", color="#444", lw=1.6))
ax.text(x_cb[-1] * 1.04, (y_base[-1] + y_cb[-1]) / 2,
        f"−{y_base[-1] - y_cb[-1]:.1f} PAPE\nat +{overhead:.1f}% comm",
        fontsize=11, fontweight="bold", color="#222", va="center", ha="left")
ax.set_xscale("log")
ax.set_xlabel("Cumulative uplink per client  (MB, log scale)", fontsize=12)
ax.set_ylabel("Peak Absolute Percentage Error (%)", fontsize=12)
ax.set_title("Peak Error vs Communication Budget\n"
             "RoundCB reaches a lower PAPE at essentially the same uplink cost",
             fontsize=12.5, fontweight="bold")
ax.set_xlim(x_base[0] * 0.8, x_cb[-1] * 1.6)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper right", fontsize=11, frameon=True, framealpha=0.95,
          facecolor="white", edgecolor="#ccc")
fig.tight_layout()
p13 = OUT / "fig13_pape_vs_communication.png"
fig.savefig(p13, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"saved: {p13}")


print("\n=== final-round summary (MAE-only, 3-seed) ===")
for a in algos:
    i = algos.index(a)
    print(f"{a:9s} before {mb[i]:5.2f}  after {ma[i]:5.2f}  lift {lift[i]:+.2f}")
print(f"\npooled FL baseline {fl_m:.2f}±{fl_s:.2f}  RoundCB {rc_m:.2f}±{rc_s:.2f}")
print(f"baselines: chronos-bolt-small {ch_m:.2f}  timesfm {tf_m:.2f}  "
      f"dlinear {dl_m:.2f}  centralized-nbeatsx {cn_m:.2f}")
print(f"comm/cl/rd: FedAvg {_full:,}  FedRep {_enc:,}  FedProto {_full + _proto:,}  "
      f"codebook {cb_pc:,} B  (overhead {overhead:.2f}% of a weight update)")
