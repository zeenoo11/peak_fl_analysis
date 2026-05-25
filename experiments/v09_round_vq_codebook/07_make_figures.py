"""v09 explanation figure — structure + seed 42 FedAvg trajectory.

Panel A: schematic showing v06 phase 1 / v06 phase 2 / v09 03 driver structure
         on a single round-axis.
Panel B: per-round PAPE before vs after CMO correction.
Panel C: codebook utilization + perplexity per round.

Reads:  outputs/v09_round_vq_codebook/seed42/V9-RoundCB-FedAvg/codebook_log.jsonl
Writes: outputs/v09_round_vq_codebook/figures/v09_explanation.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Try Malgun Gothic (Windows Korean default); fall back silently.
try:
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
except Exception:
    pass
matplotlib.rcParams["axes.unicode_minus"] = False


BASE = Path(__file__).resolve().parents[2]
LOG = BASE / "outputs/v09_round_vq_codebook/seed42/V9-RoundCB-FedAvg/codebook_log.jsonl"
OUT = BASE / "outputs/v09_round_vq_codebook/figures/v09_explanation.png"
OUT2 = BASE / "outputs/v09_round_vq_codebook/figures/v09_5algos_trajectory.png"
PAPERS_FIG = BASE / "papers/v09_draft/figures"
OUT.parent.mkdir(parents=True, exist_ok=True)
PAPERS_FIG.mkdir(parents=True, exist_ok=True)

rows = [json.loads(line) for line in LOG.open()]
rounds = [r["round"] for r in rows]
pape_before = [r["test_before"]["pape_mean"] for r in rows]
pape_after  = [r["test_after"]["pape_mean"] for r in rows]
util = [r["codebook_diag"]["utilization"] for r in rows]
ppl  = [r["codebook_diag"]["perplexity"] for r in rows]

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[1.55, 1.05, 1.05], hspace=0.55)


# ============================================================================
# Panel A — schematic
# ============================================================================
axA = fig.add_subplot(gs[0])

R_v06 = 20
R_v09 = 10
xw = 0.85

y_p1 = 3.3        # v06 phase 1 backbone row
y_p2 = 2.0        # v06 phase 2 codebook row
y_v09 = 0.35      # v09 backbone row
y_v09_cb = -0.85  # v09 codebook row

BB_FC = "#cfe2f3"  # light blue
BB_EC = "#1f77b4"
CB_FC = "#f4cccc"  # light red
CB_EC = "#d62728"


def draw_box(ax, x, y, w, h, ec, fc, text):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0.9, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.8)


# Row labels (left side)
axA.text(-1.5, y_p1 + 0.3, "v06 phase 1\nbackbone only",
         fontsize=9.5, ha="left", va="center", fontweight="bold")
axA.text(-1.5, y_p2 + 0.3, "v06 phase 2\npost-hoc 1x",
         fontsize=9.5, ha="left", va="center", fontweight="bold")
axA.text(-1.5, (y_v09 + y_v09_cb) / 2 + 0.3, "v09 03\npost-hoc per round",
         fontsize=9.5, ha="left", va="center", fontweight="bold")

# v06 phase 1: R=20 backbone boxes
for r in range(1, R_v06 + 1):
    draw_box(axA, r - 0.4, y_p1, xw, 0.65, BB_EC, BB_FC, f"BB\nR{r}")

# v06 phase 2: one codebook box at final round
draw_box(axA, R_v06 - 0.4, y_p2, xw, 0.65, CB_EC, CB_FC, "CB\nfit")
axA.annotate(
    "", xy=(R_v06 - 0.4 + xw / 2, y_p2 + 0.65),
    xytext=(R_v06 - 0.4 + xw / 2, y_p1),
    arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
)
axA.text(R_v06 + 1.0, y_p2 + 0.3, "(frozen backbone)",
         fontsize=8, ha="left", va="center", color="gray", style="italic")

# v09 03: R=10 backbone + R=10 codebook, each round
for r in range(1, R_v09 + 1):
    draw_box(axA, r - 0.4, y_v09, xw, 0.65, BB_EC, BB_FC, f"BB\nR{r}")
    draw_box(axA, r - 0.4, y_v09_cb, xw, 0.65, CB_EC, CB_FC, "CB\nfit")
    axA.annotate(
        "", xy=(r - 0.4 + xw / 2, y_v09_cb + 0.65),
        xytext=(r - 0.4 + xw / 2, y_v09),
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.7),
    )

# Note on v09: codebook does NOT feed back to backbone
axA.text(R_v09 + 1.0, (y_v09 + y_v09_cb) / 2 + 0.3,
         "(no gradient back to backbone:\n  backbone-codebook decoupled)",
         fontsize=8, ha="left", va="center", color="gray", style="italic")

axA.set_xlim(-2.2, R_v06 + 5.5)
axA.set_ylim(-1.7, 4.3)
axA.set_xticks(range(1, R_v06 + 1))
axA.set_xticklabels([str(r) for r in range(1, R_v06 + 1)], fontsize=7)
axA.set_yticks([])
axA.set_xlabel("Round", fontsize=10)
axA.set_title("Experiment structure: where the codebook fit happens",
              fontsize=12, fontweight="bold")
for spine in ("top", "right", "left"):
    axA.spines[spine].set_visible(False)

bb_patch = patches.Patch(facecolor=BB_FC, edgecolor=BB_EC,
                         label="BB = backbone FL round (1 FedAvg cycle, E local epochs)")
cb_patch = patches.Patch(facecolor=CB_FC, edgecolor=CB_EC,
                         label="CB fit = federated KMeans + CMO offsets + test eval")
axA.legend(handles=[bb_patch, cb_patch], loc="upper right",
           fontsize=8.5, framealpha=0.95)


# ============================================================================
# Panel B — PAPE before/after per round
# ============================================================================
axB = fig.add_subplot(gs[1])
axB.plot(rounds, pape_before, "o-", color="#2ca02c",
         label="pape_before  (backbone only, no codebook)", lw=1.8, markersize=6)
axB.plot(rounds, pape_after,  "s-", color="#d62728",
         label="pape_after  (backbone + CMO correction)",   lw=1.8, markersize=6)
axB.fill_between(
    rounds, pape_after, pape_before,
    where=[a < b for a, b in zip(pape_after, pape_before)],
    alpha=0.15, color="green", interpolate=True,
)
axB.fill_between(
    rounds, pape_after, pape_before,
    where=[a >= b for a, b in zip(pape_after, pape_before)],
    alpha=0.20, color="red", interpolate=True,
)

# Annotations on key rounds
axB.annotate(
    "R1: backbone near-random\npape_before = 69.7", xy=(1, 69.7),
    xytext=(2.7, 78), fontsize=8.5,
    arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"),
)
axB.annotate(
    "R3: codebook hurts (delta = +3.49)", xy=(3, 50.12),
    xytext=(4.0, 32), fontsize=8.5,
    arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"),
)
axB.annotate(
    "R5-10: stabilised (delta ~ -5)", xy=(7.5, 47.85),
    xytext=(5.5, 60), fontsize=8.5,
    arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"),
)

axB.set_xticks(rounds)
axB.set_xlabel("Round")
axB.set_ylabel("test PAPE (%)")
axB.set_title("Panel B — PAPE before vs after CMO correction (FedAvg, seed=42)",
              fontsize=11, fontweight="bold")
axB.legend(loc="upper right", fontsize=8.5)
axB.grid(alpha=0.3)


# ============================================================================
# Panel C — codebook diagnostics
# ============================================================================
axC = fig.add_subplot(gs[2])
axC.plot(rounds, ppl, "D-", color="#9467bd", label="perplexity", lw=1.8, markersize=6)
axC.set_ylabel("codebook perplexity", color="#9467bd")
axC.tick_params(axis="y", labelcolor="#9467bd")
axC.set_xticks(rounds)
axC.set_xlabel("Round")
axC.set_title("Panel C — Codebook diagnostics (utilization stays 1.000; perplexity grows R1→R10)",
              fontsize=11, fontweight="bold")
axC.grid(alpha=0.3)
axC.set_ylim(15, 30)

axC2 = axC.twinx()
axC2.plot(rounds, util, "^-", color="#ff7f0e", label="utilization", lw=1.8, markersize=7)
axC2.set_ylabel("utilization", color="#ff7f0e")
axC2.tick_params(axis="y", labelcolor="#ff7f0e")
axC2.set_ylim(0.0, 1.1)

lines1, labels1 = axC.get_legend_handles_labels()
lines2, labels2 = axC2.get_legend_handles_labels()
axC.legend(lines1 + lines2, labels1 + labels2,
           loc="lower right", fontsize=8.5)

plt.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"saved: {OUT}")
import shutil
shutil.copy2(OUT, PAPERS_FIG / OUT.name)
print(f"copied: {PAPERS_FIG / OUT.name}")


# ============================================================================
# Figure 2 — 5 FL algos Δpape trajectory across R1..R10 (seed=42)
# ============================================================================

ALGOS = [
    ("FedAvg",   "V9-RoundCB-FedAvg",   "#1f77b4"),
    ("FedProx",  "V9-RoundCB-FedProx",  "#2ca02c"),
    ("FedRep",   "V9-RoundCB-FedRep",   "#ff7f0e"),
    ("Ditto",    "V9-RoundCB-Ditto",    "#d62728"),
    ("FedProto", "V9-RoundCB-FedProto", "#9467bd"),
]

fig2, ax2 = plt.subplots(figsize=(11, 5.5))

ax2.axhline(0, ls="--", color="gray", alpha=0.6, lw=1.0)
ax2.text(
    10.05, 0.4, "codebook hurts ->",
    fontsize=8.5, color="gray", style="italic", va="center",
)
ax2.text(
    10.05, -0.4, "codebook helps ->",
    fontsize=8.5, color="gray", style="italic", va="center",
)

for label, cell, color in ALGOS:
    path = BASE / f"outputs/v09_round_vq_codebook/seed42/{cell}/codebook_log.jsonl"
    rows_a = [json.loads(line) for line in path.open()]
    rs = [r["round"] for r in rows_a]
    dp = [r["lift"]["pape_delta"] for r in rows_a]
    lw = 2.4 if label in ("Ditto", "FedRep") else 1.6
    ax2.plot(rs, dp, "o-", color=color, label=label, lw=lw, markersize=6)

# Highlights
ax2.annotate(
    "FedRep: R3-R6 4 rounds\nof positive delta_pape\n(encoder/head split early instability)",
    xy=(5, 5.63), xytext=(6.0, 8.0),
    fontsize=8.5, ha="left",
    arrowprops=dict(arrowstyle="->", lw=0.8, color="#ff7f0e"),
)
ax2.annotate(
    "Ditto: monotone improvement R4 -> R10\n(global + personal twin model)",
    xy=(9, -6.30), xytext=(2.5, -10.5),
    fontsize=8.5, ha="left",
    arrowprops=dict(arrowstyle="->", lw=0.8, color="#d62728"),
)
ax2.annotate(
    "R3 single reversal cluster:\nFedAvg / FedProx / FedProto",
    xy=(3, 3.49), xytext=(0.4, 6.0),
    fontsize=8.5, ha="left",
    arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"),
)

ax2.set_xticks(list(range(1, 11)))
ax2.set_xlabel("Round")
ax2.set_ylabel("delta_pape  (test_after - test_before)")
ax2.set_title(
    "Codebook lift trajectory across 5 FL algorithms (seed=42, R=10, E=5)",
    fontsize=12, fontweight="bold",
)
ax2.legend(loc="lower left", fontsize=9.5, ncol=5)
ax2.grid(alpha=0.3)
ax2.set_xlim(0.4, 13.0)

plt.tight_layout()
plt.savefig(OUT2, dpi=130, bbox_inches="tight")
print(f"saved: {OUT2}")
shutil.copy2(OUT2, PAPERS_FIG / OUT2.name)
print(f"copied: {PAPERS_FIG / OUT2.name}")
