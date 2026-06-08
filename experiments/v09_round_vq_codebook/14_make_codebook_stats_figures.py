"""Codebook-health statistics figures for the conference presentation.

Visualises the federated codebook diagnostics logged each round by
03_fl_per_round_codebook.py (codebook_log.jsonl -> codebook_diag + test-routing
stats), compared across the five FL algorithms and across rounds.

Two figures, both 3-seed mean over the MAE-only run, audience-facing English
titles (no internal jargon):

  fig8_codebook_health.png        2x2 per-round panels:
        A perplexity (effective active codes, max=M)
        B utilization (fraction of codes used) — stays 1.0, no dead codes
        C largest-cluster share on test (routing balance)
        D server codebook inertia (compactness)
  fig9_codebook_stats_by_algo.png 1x3 final-round grouped bars by algorithm:
        perplexity / largest-cluster share / server inertia (mean +/- std).

Writes: papers/conference_draft/figures/fig8_*.png, fig9_*.png
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

COLORS = {
    "FedAvg":   "#0072B2",
    "FedProx":  "#D55E00",
    "FedRep":   "#009E73",
    "Ditto":    "#CC79A7",
    "FedProto": "#56B4E9",
}
M = 32


def _field(row, path):
    cur = row
    for k in path.split("."):
        cur = cur[k]
    return cur


# Per-round diagnostics: {algo: {field: array[seed, R]}}, plus shared rounds.
FIELDS = {
    "perplexity": "codebook_diag.perplexity",
    "utilization": "codebook_diag.utilization",
    "top1_share": "cluster_assignment_top1_share",
    "stage2_inertia": "codebook_diag.stage2_inertia",
}
DATA = {}
ROUNDS = None
for algo in COLORS:
    per_field = {f: [] for f in FIELDS}
    for s in SEEDS:
        rows = [json.loads(l) for l in
                (BASE / "outputs" / NS / f"seed{s}" / f"V9-RoundCB-{algo}"
                 / "codebook_log.jsonl").open()]
        ROUNDS = [r["round"] for r in rows]
        for f, path in FIELDS.items():
            per_field[f].append([_field(r, path) for r in rows])
    DATA[algo] = {f: np.array(v) for f, v in per_field.items()}
NR = max(ROUNDS)
XT = [t for t in (1, 5, 10, 15, 20) if t <= NR]
algos = list(COLORS)


# ---------------------------------------------------------------------------
# Fig 8 — 2x2 per-round codebook health, one line per algorithm (3-seed mean).
# ---------------------------------------------------------------------------
def _panel(ax, field, ylabel, title, *, ref=None, ref_lbl=None):
    for algo, color in COLORS.items():
        m = DATA[algo][field].mean(0)
        ax.plot(ROUNDS, m, "-o", color=color, lw=1.9, markersize=3.8, label=algo)
    if ref is not None:
        ax.axhline(ref, ls="--", color="gray", lw=1.3, alpha=0.8, label=ref_lbl)
    ax.set_xlabel("Communication round")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(XT)
    ax.grid(True, alpha=0.3)


fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

_panel(axes[0, 0], "perplexity",
       f"Perplexity (max = {M})",
       "Codebook diversity (perplexity)",
       ref=M, ref_lbl=f"uniform max = {M}")
axes[0, 0].legend(loc="lower right", fontsize=9, ncol=2, framealpha=0.9)

_panel(axes[0, 1], "utilization",
       "Utilization (fraction of codes used)",
       "Codebook utilization — no collapse")
axes[0, 1].set_ylim(0.0, 1.08)
axes[0, 1].text(0.5, 0.5, "Stays at 1.00 every round, for every algorithm\n"
                "(0 empty clusters — the federated codebook never collapses)",
                transform=axes[0, 1].transAxes, ha="center", va="center",
                fontsize=10.5, color="#333",
                bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="gray", alpha=0.9))

_panel(axes[1, 0], "top1_share",
       "Largest cluster's share of test routings",
       "Routing balance (lower = more balanced)")
axes[1, 0].legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9)

_panel(axes[1, 1], "stage2_inertia",
       "Server codebook inertia",
       "Codebook compactness (Stage-2 inertia)")
axes[1, 1].legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9)

fig.suptitle("Federated Codebook Health Across Rounds\n"
             "Five FL algorithms · 3-seed mean · 114 households",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
p8 = OUT / "fig8_codebook_health.png"
fig.savefig(p8, dpi=150)
plt.close(fig)
print(f"saved: {p8}")


# ---------------------------------------------------------------------------
# Fig 9 — Final-round codebook statistics by algorithm (1x3 grouped bars).
# ---------------------------------------------------------------------------
SPECS = [
    ("perplexity", f"Perplexity (max = {M})", "Diversity", M),
    ("top1_share", "Largest-cluster share", "Routing balance", None),
    ("stage2_inertia", "Server codebook inertia", "Compactness", None),
]
x = np.arange(len(algos))
bar_colors = [COLORS[a] for a in algos]

fig, axs = plt.subplots(1, 3, figsize=(15, 5.0))
for ax, (field, ylabel, sub, ref) in zip(axs, SPECS):
    means = np.array([DATA[a][field][:, -1].mean() for a in algos])
    stds = np.array([DATA[a][field][:, -1].std(ddof=1) for a in algos])
    ax.bar(x, means, yerr=stds, color=bar_colors, alpha=0.9,
           edgecolor="black", linewidth=0.6,
           error_kw={"elinewidth": 1.0, "capsize": 3.0, "ecolor": "#333"})
    if ref is not None:
        ax.axhline(ref, ls="--", color="gray", lw=1.3, alpha=0.8)
        ax.text(len(algos) - 0.5, ref, f" max {ref}", va="bottom", ha="right",
                fontsize=9, color="gray")
    for i, (mn, sd) in enumerate(zip(means, stds)):
        ax.text(i, mn + sd, f"{mn:.2f}" if mn < 10 else f"{mn:.0f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=18, ha="right", fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(sub, fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(means + stds) * 1.18)

fig.suptitle("Final-Round Codebook Statistics by FL Algorithm\n"
             "3-seed mean ± std · 114 households",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.92))
p9 = OUT / "fig9_codebook_stats_by_algo.png"
fig.savefig(p9, dpi=150)
plt.close(fig)
print(f"saved: {p9}")


# Console summary.
print("\n=== final-round codebook stats (3-seed mean) ===")
print(f"{'algo':9s} {'perplexity':>11s} {'util':>6s} {'top1share':>10s} {'stage2_inertia':>15s}")
for a in algos:
    d = DATA[a]
    print(f"{a:9s} {d['perplexity'][:,-1].mean():11.2f} {d['utilization'][:,-1].mean():6.2f} "
          f"{d['top1_share'][:,-1].mean():10.3f} {d['stage2_inertia'][:,-1].mean():15.2f}")
