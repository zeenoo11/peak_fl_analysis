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
gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.05, 1.05], hspace=0.55)


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


# ============================================================================
# G1 / G2 — per-round test PAPE, one line per FL algorithm.
#   G1 = test_before  (no codebook).
#   G2 = test_after   (codebook applied), own y-axis so the convergence is
#        visible, with a SINGLE red dashed reference line at the no-codebook
#        converged level (mean of test_before over the last-half rounds).
#        No shaded band. Legend top-right on both.
# Reusable across round budgets: change ``NS`` / ``TAG`` (R20 / R40 / ...).
# ============================================================================

# Wong color-blind-friendly palette (matches the report figures).
_G_COLORS = {
    "FedAvg":   "#0072B2",
    "FedProx":  "#D55E00",
    "FedRep":   "#009E73",
    "Ditto":    "#CC79A7",
    "FedProto": "#56B4E9",
}


def make_before_after_figures(namespace: str, tag: str) -> None:
    """Write v09_{tag}_G1_no_codebook.png and v09_{tag}_G2_codebook.png
    under ``outputs/{namespace}/figures/`` from the seed42 RoundCB logs."""
    root = BASE / "outputs" / namespace / "seed42"
    figdir = BASE / "outputs" / namespace / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    data: dict[str, tuple[list, list, list]] = {}
    for algo in _G_COLORS:
        path = root / f"V9-RoundCB-{algo}" / "codebook_log.jsonl"
        rows_a = [json.loads(line) for line in path.open()]
        data[algo] = (
            [r["round"] for r in rows_a],
            [r["test_before"]["pape_mean"] for r in rows_a],
            [r["test_after"]["pape_mean"] for r in rows_a],
        )
    n_rounds = max(data["FedAvg"][0])
    xticks = [t for t in (1, 5, 10, 15, 20, 25, 30, 35, 40) if t <= n_rounds]

    # No-codebook converged level: mean of test_before over the last-half rounds.
    half = n_rounds // 2
    conv_vals = [v for algo in _G_COLORS
                 for r, v in zip(data[algo][0], data[algo][1]) if r > half]
    conv_mean = sum(conv_vals) / len(conv_vals)

    # --- G1: no codebook (own auto axis) ---
    figg, axg = plt.subplots(figsize=(8.0, 5.0))
    for algo, color in _G_COLORS.items():
        rnd, before, _ = data[algo]
        axg.plot(rnd, before, "-o", color=color, lw=2.0, markersize=4.5, label=algo)
    axg.set_xlabel("Round")
    axg.set_ylabel("Test PAPE (%)")
    axg.set_title(f"v09 RoundCB - test_before (no codebook)\nseed=42, R={n_rounds}, E=5")
    axg.set_xticks(xticks)
    axg.grid(True, alpha=0.3)
    axg.legend(loc="upper right", frameon=True, framealpha=0.85,
               facecolor="white", edgecolor="none", fontsize=11)
    figg.tight_layout()
    out1 = figdir / f"v09_{tag}_G1_no_codebook.png"
    figg.savefig(out1, dpi=160)
    plt.close(figg)
    print(f"saved: {out1}")

    # --- G2: codebook applied (own axis) + single red dashed reference line ---
    figg, axg = plt.subplots(figsize=(8.0, 5.0))
    axg.axhline(conv_mean, color="red", ls="--", lw=1.6,
                label=f"No-codebook converged (~{conv_mean:.0f}%)")
    for algo, color in _G_COLORS.items():
        rnd, _, after = data[algo]
        axg.plot(rnd, after, "-o", color=color, lw=2.0, markersize=4.5, label=algo)
    after_min = min(v for algo in _G_COLORS for v in data[algo][2])
    axg.set_ylim(after_min - 1.0, conv_mean + 1.5)
    axg.set_xlabel("Round")
    axg.set_ylabel("Test PAPE (%)")
    axg.set_title(f"v09 RoundCB - test_after (codebook applied)\nseed=42, R={n_rounds}, E=5")
    axg.set_xticks(xticks)
    axg.grid(True, alpha=0.3)
    axg.legend(loc="upper right", frameon=True, framealpha=0.85,
               facecolor="white", edgecolor="none", fontsize=10.5)
    figg.tight_layout()
    out2 = figdir / f"v09_{tag}_G2_codebook.png"
    figg.savefig(out2, dpi=160)
    plt.close(figg)
    print(f"saved: {out2}")


def make_multiseed_figures(namespace: str, tag: str, seeds=(42, 123, 7),
                           bands: bool = True) -> dict:
    """Mean +/- std (across seeds) per-round test PAPE, one line per FL algo.
    G1 = test_before (no codebook), G2 = test_after (codebook) with a single
    red dashed no-codebook-converged reference. Each curve gets a +/-1 std band.
    Saves under outputs/{namespace}/figures/ with a *_3seed_* name (does NOT
    overwrite the single-seed figures). Returns final-round summary stats.

    bands=False draws the same structure with seed-mean lines only (no +/-std
    fill), saved with a *_meanonly_* name so it does not overwrite the banded
    figures.
    """
    import numpy as np
    root = BASE / "outputs" / namespace
    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # data[algo] = (rounds, before[n_seed, R], after[n_seed, R])
    data: dict[str, tuple] = {}
    for algo in _G_COLORS:
        bef, aft, rounds = [], [], None
        for s in seeds:
            path = root / f"seed{s}" / f"V9-RoundCB-{algo}" / "codebook_log.jsonl"
            rows_a = [json.loads(line) for line in path.open()]
            rounds = [r["round"] for r in rows_a]
            bef.append([r["test_before"]["pape_mean"] for r in rows_a])
            aft.append([r["test_after"]["pape_mean"] for r in rows_a])
        data[algo] = (rounds, np.array(bef), np.array(aft))
    n_rounds = max(data["FedAvg"][0])
    xticks = [t for t in (1, 5, 10, 15, 20, 25, 30, 35, 40) if t <= n_rounds]
    half = n_rounds // 2

    # No-codebook converged reference: mean of before over last-half rounds,
    # across all seeds and algos.
    conv = np.concatenate([
        data[a][1][:, [i for i, r in enumerate(data[a][0]) if r > half]].ravel()
        for a in _G_COLORS
    ])
    conv_mean = float(conv.mean())

    def _panel(which: str, title: str, fname: str, ref_line: bool):
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        if ref_line:
            ax.axhline(conv_mean, color="red", ls="--", lw=1.6,
                       label=f"No-codebook converged (~{conv_mean:.0f}%)")
        for algo, color in _G_COLORS.items():
            rounds, bef, aft = data[algo]
            arr = bef if which == "before" else aft
            m = arr.mean(axis=0)
            sd = arr.std(axis=0, ddof=1)
            ax.plot(rounds, m, "-o", color=color, lw=2.0, markersize=4.0, label=algo)
            if bands:
                ax.fill_between(rounds, m - sd, m + sd, color=color, alpha=0.15)
        ax.set_xlabel("Round")
        ax.set_ylabel("Test PAPE (%)")
        ax.set_title(title)
        ax.set_xticks(xticks)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", frameon=True, framealpha=0.85,
                  facecolor="white", edgecolor="none", fontsize=10.5)
        if which == "after":
            after_min = min(
                data[a][2].mean(axis=0).min()
                - (data[a][2].std(axis=0, ddof=1).max() if bands else 0.0)
                for a in _G_COLORS)
            ax.set_ylim(after_min - 0.5, conv_mean + 1.5)
        fig.tight_layout()
        out = figdir / fname
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"saved: {out}")

    ns = len(seeds)
    sub = f"mean +/- std over {ns} seeds" if bands else f"seed-mean over {ns} seeds"
    slug = "3seed" if bands else "meanonly"
    _panel("before",
            f"v09 RoundCB - test_before (no codebook)\n{sub}, R={n_rounds}, E=5",
            f"v09_{tag}_{slug}_G1_no_codebook.png", ref_line=False)
    _panel("after",
            f"v09 RoundCB - test_after (codebook applied)\n{sub}, R={n_rounds}, E=5",
            f"v09_{tag}_{slug}_G2_codebook.png", ref_line=True)

    # Final-round summary (mean +/- std across seeds).
    summary = {}
    for algo in _G_COLORS:
        _, bef, aft = data[algo]
        b, a = bef[:, -1], aft[:, -1]
        summary[algo] = {
            "before_mean": float(b.mean()), "before_std": float(b.std(ddof=1)),
            "after_mean": float(a.mean()),  "after_std": float(a.std(ddof=1)),
            "delta_mean": float((a - b).mean()), "delta_std": float((a - b).std(ddof=1)),
        }
    return summary


# Default: R=20 namespace (the figures currently in the report).
make_before_after_figures("v09_round_vq_codebook_R20", "R20")
# R=40 extension (seed42).
make_before_after_figures("v09_round_vq_codebook_R40", "R40")
# 3-seed aggregate (aux=0.3, R=20).
_summary = make_multiseed_figures("v09_round_vq_codebook_R20", "R20")
print("\n=== R=20 final-round 3-seed summary (mean +/- std) ===")
for _algo, _s in _summary.items():
    print(f"{_algo:9s} before {_s['before_mean']:5.2f}+/-{_s['before_std']:.2f}  "
          f"after {_s['after_mean']:5.2f}+/-{_s['after_std']:.2f}  "
          f"delta {_s['delta_mean']:+.2f}+/-{_s['delta_std']:.2f}")


def _load_ns(namespace, seeds=(42, 123, 7)):
    """{algo: (rounds, before[n_seed,R], after[n_seed,R])} for a namespace."""
    import numpy as np
    root = BASE / "outputs" / namespace
    out = {}
    for algo in _G_COLORS:
        bef, aft, rounds = [], [], None
        for s in seeds:
            rows = [json.loads(l) for l in
                    (root / f"seed{s}" / f"V9-RoundCB-{algo}" / "codebook_log.jsonl").open()]
            rounds = [r["round"] for r in rows]
            bef.append([r["test_before"]["pape_mean"] for r in rows])
            aft.append([r["test_after"]["pape_mean"] for r in rows])
        out[algo] = (rounds, np.array(bef), np.array(aft))
    return out


def make_combined_figure(namespace, tag, seeds=(42, 123, 7), bands=False):
    """No-codebook (test_before) and codebook (test_after) on one axis.
    Color = FL algo; line style = solid (no codebook) vs dashed (codebook).
    Shows the codebook lift as a downward shift of the whole cluster.
    bands=True adds +/-1 std fills; default mean-only for clarity.
    """
    import numpy as np
    from matplotlib.lines import Line2D
    data = _load_ns(namespace, seeds)
    figdir = BASE / "outputs" / namespace / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    rounds = data["FedAvg"][0]
    n_rounds = max(rounds)
    xticks = [t for t in (1, 5, 10, 15, 20, 25, 30, 35, 40) if t <= n_rounds]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for algo, color in _G_COLORS.items():
        _, bef, aft = data[algo]
        mb, ma = bef.mean(0), aft.mean(0)
        ax.plot(rounds, mb, "-o", color=color, lw=1.8, markersize=3.0)
        ax.plot(rounds, ma, "--o", color=color, lw=2.2, markersize=3.5)
        if bands:
            sb, sa = bef.std(0, ddof=1), aft.std(0, ddof=1)
            ax.fill_between(rounds, mb - sb, mb + sb, color=color, alpha=0.10)
            ax.fill_between(rounds, ma - sa, ma + sa, color=color, alpha=0.10)
    ax.set_xlabel("Round")
    ax.set_ylabel("Test PAPE (%)")
    ax.set_title("v09 RoundCB - no codebook (solid) vs codebook applied (dashed)\n"
                 f"seed-mean over {len(seeds)} seeds, R={n_rounds}, E=5")
    ax.set_xticks(xticks)
    ax.grid(True, alpha=0.3)

    no_handles = [Line2D([0], [0], color=c, lw=2.0, ls="-",
                         label=f"{a} (no CB)") for a, c in _G_COLORS.items()]
    cb_handles = [Line2D([0], [0], color=c, lw=2.4, ls="--",
                         label=f"{a} (CB)") for a, c in _G_COLORS.items()]
    leg1 = ax.legend(handles=no_handles, loc="upper right", frameon=True,
                     framealpha=0.85, facecolor="white", edgecolor="none",
                     fontsize=9, title="No codebook")
    ax.add_artist(leg1)
    ax.legend(handles=cb_handles, loc="center right", frameon=True,
              framealpha=0.85, facecolor="white", edgecolor="none",
              fontsize=9, title="Codebook applied")
    fig.tight_layout()
    slug = "combined_bands" if bands else "combined"
    out = figdir / f"v09_{tag}_{slug}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved: {out}")


def make_codebook_lift_bar(namespace, tag, aux_label, seeds=(42, 123, 7)):
    """v08 F6-style grouped bar: per-algo BEFORE (no codebook, faded) vs
    AFTER (+ codebook CMO, saturated) final-round TEST PAPE, with the Δ lift
    annotated above each pair and +/-std errorbars across seeds.
    """
    import numpy as np
    data = _load_ns(namespace, seeds)
    figdir = BASE / "outputs" / namespace / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    algos = list(_G_COLORS)
    n_rounds = max(data["FedAvg"][0])

    mb = np.array([data[a][1][:, -1].mean() for a in algos])
    sb = np.array([data[a][1][:, -1].std(ddof=1) for a in algos])
    ma = np.array([data[a][2][:, -1].mean() for a in algos])
    sa = np.array([data[a][2][:, -1].std(ddof=1) for a in algos])
    lift = ma - mb

    n = len(algos)
    width = 0.36
    x = np.arange(n)
    y_lo = float(np.floor(np.min(ma - sa) - 2.0))
    y_hi = float(np.ceil(np.max(mb + sb) + 4.0))
    off = (y_hi - y_lo) * 0.04

    with plt.rc_context({"font.size": 13}):
        fig, ax = plt.subplots(figsize=(9.0, 5.5))
        ax.set_ylim(y_lo, y_hi)
        for i, a in enumerate(algos):
            col = _G_COLORS[a]
            ax.bar(x[i] - width / 2, mb[i], width=width, yerr=sb[i],
                   color=col, alpha=0.35, edgecolor="black", linewidth=0.6,
                   error_kw={"elinewidth": 1.0, "capsize": 3.0, "ecolor": "#555555"},
                   label="BEFORE (no codebook)" if i == 0 else None)
            ax.bar(x[i] + width / 2, ma[i], width=width, yerr=sa[i],
                   color=col, alpha=0.95, edgecolor="black", linewidth=0.6,
                   error_kw={"elinewidth": 1.0, "capsize": 3.0, "ecolor": "#222222"},
                   label="AFTER (+ codebook CMO)" if i == 0 else None)
            top = max(mb[i] + sb[i], ma[i] + sa[i])
            sign = "+" if lift[i] >= 0 else ""
            ax.text(x[i], top + off, f"Δ {sign}{lift[i]:.2f}",
                    ha="center", va="bottom", fontsize=11, color="#222222",
                    fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=15, ha="right", fontsize=11)
        ax.set_ylabel("Across-client TEST PAPE (%)")
        ax.set_title(f"Codebook lift on TEST PAPE — {aux_label} "
                     f"(R={n_rounds}, final round)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.18),
                  frameon=False, fontsize=11, ncol=2)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)
        out = figdir / f"v09_{tag}_codebook_lift.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {out}")


def make_aux_ba_bar(ns_aux, ns_mae, tag, seeds=(42, 123, 7)):
    """v08 F6-style bar, 4 bars per algo: before/after x aux=0.3/MAE-only.
    Color = FL algo; alpha = before (0.35) vs after (0.95); hatch '//' = MAE-only.
    Each before->after pair gets its Delta lift annotated above.
    """
    import numpy as np
    from matplotlib.patches import Patch
    A, M = _load_ns(ns_aux, seeds), _load_ns(ns_mae, seeds)
    figdir = BASE / "outputs" / ns_aux / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    algos = list(_G_COLORS)
    n_rounds = max(A["FedAvg"][0])

    def _final(src, algo, idx):  # idx 1=before, 2=after
        v = src[algo][idx][:, -1]
        return float(v.mean()), float(v.std(ddof=1))

    n = len(algos)
    width = 0.20
    offs = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    x = np.arange(n)

    # collect for ylim
    all_top, all_bot = [], []
    for a in algos:
        for src in (A, M):
            for idx in (1, 2):
                m, s = _final(src, a, idx)
                all_top.append(m + s); all_bot.append(m - s)
    y_lo = float(np.floor(min(all_bot) - 2.0))
    y_hi = float(np.ceil(max(all_top) + 4.0))
    off = (y_hi - y_lo) * 0.035

    with plt.rc_context({"font.size": 13}):
        fig, ax = plt.subplots(figsize=(13.0, 6.0))
        ax.set_ylim(y_lo, y_hi)
        # bar specs: (src, idx, alpha, hatch)
        specs = [(A, 1, 0.35, None), (A, 2, 0.95, None),
                 (M, 1, 0.35, "//"), (M, 2, 0.95, "//")]
        for i, a in enumerate(algos):
            col = _G_COLORS[a]
            mvals = {}
            for j, (src, idx, alpha, hatch) in enumerate(specs):
                m, s = _final(src, a, idx)
                mvals[(j)] = (m, s)
                ax.bar(x[i] + offs[j], m, width=width, yerr=s,
                       color=col, alpha=alpha, edgecolor="black", linewidth=0.6,
                       hatch=hatch,
                       error_kw={"elinewidth": 1.0, "capsize": 2.5, "ecolor": "#333333"})
            # Delta labels: aux pair (j=0->1) and MAE pair (j=2->3).
            for (jb, ja), xc in [((0, 1), (offs[0] + offs[1]) / 2),
                                 ((2, 3), (offs[2] + offs[3]) / 2)]:
                mb = mvals[jb][0]; ma = mvals[ja][0]
                top = max(mvals[jb][0] + mvals[jb][1], mvals[ja][0] + mvals[ja][1])
                d = ma - mb
                ax.text(x[i] + xc, top + off, f"Δ{d:+.1f}",
                        ha="center", va="bottom", fontsize=9, color="#222222",
                        fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(algos, fontsize=12)
        ax.set_ylabel("Across-client TEST PAPE (%)")
        ax.set_title("Codebook lift x aux setting — before/after, aux=0.3 vs MAE-only\n"
                     f"3-seed mean +/- std, R={n_rounds}, final round")
        ax.grid(True, axis="y", alpha=0.3)
        legend_handles = [
            Patch(facecolor="0.5", alpha=0.35, edgecolor="black", label="BEFORE (no codebook)"),
            Patch(facecolor="0.5", alpha=0.95, edgecolor="black", label="AFTER (+ codebook CMO)"),
            Patch(facecolor="0.5", alpha=0.7, edgecolor="black", hatch="//", label="MAE-only (λ_aux=0)"),
            Patch(facecolor="0.5", alpha=0.7, edgecolor="black", label="aux=0.3 (no hatch)"),
        ]
        ax.legend(handles=legend_handles, loc="upper right",
                  bbox_to_anchor=(1.0, -0.10), frameon=False, fontsize=11, ncol=4)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.18)
        out = figdir / f"v09_{tag}_aux_ba_bar.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {out}")


def make_aux_ab_figures(ns_aux, ns_mae, tag, seeds=(42, 123, 7)):
    """aux=0.3 vs aux=0 (MAE-only) A/B, 3-seed mean +/- std.
    Fig1: FedRep before-curve (the dip is aux-triggered).
    Fig2: codebook-applied floor, mean over 5 algos."""
    import numpy as np
    A, M = _load_ns(ns_aux, seeds), _load_ns(ns_mae, seeds)
    figdir = BASE / "outputs" / ns_aux / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    rounds = A["FedAvg"][0]
    xticks = [t for t in (1, 5, 10, 15, 20) if t <= max(rounds)]

    def _legend(ax):
        ax.legend(loc="upper right", frameon=True, framealpha=0.85,
                  facecolor="white", edgecolor="none", fontsize=11)

    # Fig 1: FedRep before — aux=0.3 (dip) vs aux=0 (no dip).
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for lbl, src, color in [("aux=0.3 (peak head on)", A, "#D55E00"),
                            ("aux=0 (MAE-only)", M, "#0072B2")]:
        bef = src["FedRep"][1]
        m, sd = bef.mean(0), bef.std(0, ddof=1)
        ax.plot(rounds, m, "-o", color=color, lw=2.2, markersize=4.5, label=lbl)
        ax.fill_between(rounds, m - sd, m + sd, color=color, alpha=0.15)
    ax.set_xlabel("Round")
    ax.set_ylabel("FedRep test PAPE (%) — no codebook")
    ax.set_title("FedRep before-curve: the R5 dip is aux-triggered\nmean +/- std over 3 seeds, R=20, E=5")
    ax.set_xticks(xticks); ax.grid(True, alpha=0.3); _legend(ax)
    fig.tight_layout()
    out = figdir / f"v09_{tag}_auxAB_fedrep_before.png"
    fig.savefig(out, dpi=160); plt.close(fig); print(f"saved: {out}")

    # Fig 2: codebook-applied floor — mean over 5 algos, aux=0.3 vs aux=0.
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for lbl, src, color in [("aux=0.3", A, "#D55E00"),
                            ("aux=0 (MAE-only)", M, "#0072B2")]:
        stk = np.stack([src[a][2] for a in _G_COLORS], axis=0)  # [algo, seed, R]
        per_seed = stk.mean(axis=0)                              # [seed, R]
        m, sd = per_seed.mean(0), per_seed.std(0, ddof=1)
        ax.plot(rounds, m, "-o", color=color, lw=2.2, markersize=4.5, label=lbl)
        ax.fill_between(rounds, m - sd, m + sd, color=color, alpha=0.15)
    ax.set_xlabel("Round")
    ax.set_ylabel("test PAPE (%) — codebook applied")
    ax.set_title("Codebook floor: aux=0.3 vs MAE-only (mean over 5 FL algos)\nmean +/- std over 3 seeds, R=20, E=5")
    ax.set_xticks(xticks); ax.grid(True, alpha=0.3); _legend(ax)
    fig.tight_layout()
    out = figdir / f"v09_{tag}_auxAB_after_floor.png"
    fig.savefig(out, dpi=160); plt.close(fig); print(f"saved: {out}")


# 3-seed aggregate (aux=0 / MAE-only, R=20).
_summary_mae = make_multiseed_figures("v09_round_vq_codebook_R20_MAEonly", "R20_MAEonly")
print("\n=== R=20 MAE-only final-round 3-seed summary (mean +/- std) ===")
for _algo, _s in _summary_mae.items():
    print(f"{_algo:9s} before {_s['before_mean']:5.2f}+/-{_s['before_std']:.2f}  "
          f"after {_s['after_mean']:5.2f}+/-{_s['after_std']:.2f}  "
          f"delta {_s['delta_mean']:+.2f}+/-{_s['delta_std']:.2f}")
# Mean-only variants (same structure, no seed-range bands).
make_multiseed_figures("v09_round_vq_codebook_R20_MAEonly", "R20_MAEonly", bands=False)
make_multiseed_figures("v09_round_vq_codebook_R20", "R20", bands=False)
# No-codebook vs codebook on one axis.
make_combined_figure("v09_round_vq_codebook_R20", "R20")
make_combined_figure("v09_round_vq_codebook_R20_MAEonly", "R20_MAEonly")
# v08 F6-style codebook-lift bar, one per aux setting.
make_codebook_lift_bar("v09_round_vq_codebook_R20", "R20", r"default ($\lambda_{aux}=0.3$)")
make_codebook_lift_bar("v09_round_vq_codebook_R20_MAEonly", "R20_MAEonly",
                       r"MAE-only ($\lambda_{aux}=0$)")
# 4-bar/algo: before/after x aux=0.3/MAE-only.
make_aux_ba_bar("v09_round_vq_codebook_R20", "v09_round_vq_codebook_R20_MAEonly", "R20")
# aux=0.3 vs aux=0 A/B comparison.
make_aux_ab_figures("v09_round_vq_codebook_R20", "v09_round_vq_codebook_R20_MAEonly", "R20")


def make_codebook_effect_figures(namespace, tag, aux_label, seeds=(42, 123, 7)):
    """codebook 효과(after vs before)를 PAPE vs MAE로 대비 (이중 y축).
    PAPE 는 이미 %(percentage-error) 지표라 절대 percentage-point(pp)로,
    MAE 는 z-score 절대값이라 pp가 없어 상대 %로 — 각 지표를 그 자체로
    자연스러운 단위에 둔다. 왼쪽 축=PAPE Δpp, 오른쪽 축=MAE 상대 %.
    Fig1: 알고리즘별 최종(R_final) Δ 막대 (PAPE 큰 음수 vs MAE ~0).
    Fig2: 라운드별 Δ (5 algo 평균) — MAE가 수렴 후 0 근처에 머무름.
    'codebook은 peak(PAPE)를 크게 낮추되 average(MAE)는 거의 안 건드린다'를 정직하게 표현."""
    import numpy as np
    root = BASE / "outputs" / namespace
    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    pb, pa, mb, ma, rounds = [], [], [], [], None
    for algo in _G_COLORS:
        pbs, pas, mbs, mas = [], [], [], []
        for s in seeds:
            rows = [json.loads(l) for l in
                    (root / f"seed{s}" / f"V9-RoundCB-{algo}" / "codebook_log.jsonl").open()]
            rounds = [r["round"] for r in rows]
            pbs.append([r["test_before"]["pape_mean"] for r in rows])
            pas.append([r["test_after"]["pape_mean"] for r in rows])
            mbs.append([r["test_before"]["mae_mean"] for r in rows])
            mas.append([r["test_after"]["mae_mean"] for r in rows])
        pb.append(pbs); pa.append(pas); mb.append(mbs); ma.append(mas)
    pb, pa, mb, ma = (np.array(z) for z in (pb, pa, mb, ma))   # [algo, seed, R]
    pap_eff = (pa - pb)              # PAPE: absolute percentage-point (pp)
    mae_eff = (ma - mb) / mb * 100.0  # MAE: relative %
    algos = list(_G_COLORS)
    PAPE_C, MAE_C = "#0072B2", "#E69F00"

    def _align_zero(ax_main, ax_sec):
        """ax_sec의 ylim을 조정해 두 축의 0이 같은 높이에 오도록 정렬."""
        l1, h1 = ax_main.get_ylim()
        f = (0 - l1) / (h1 - l1)            # main에서 0의 상대 위치
        l2, h2 = ax_sec.get_ylim()
        span = max(h2 - (h2 - l2) * 0, 1e-9)
        # 0이 f 위치에 오도록 sec를 [b, t]로: f = (0-b)/(t-b)
        t = max(h2, 1e-3)
        b = -t * f / (1 - f) if f < 1 else l2
        ax_sec.set_ylim(b, t)

    # Fig1 — final-round Δ bars (dual axis: PAPE pp / MAE rel %).
    pm_f, ps_f = pap_eff[:, :, -1].mean(1), pap_eff[:, :, -1].std(1, ddof=1)
    mm_f, ms_f = mae_eff[:, :, -1].mean(1), mae_eff[:, :, -1].std(1, ddof=1)
    x = np.arange(len(algos)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 5.0)); ax.axhline(0, color="#888", lw=1.0)
    ax2 = ax.twinx()
    b1 = ax.bar(x - w / 2, pm_f, w, yerr=ps_f, color=PAPE_C, capsize=3,
                label="PAPE (peak error) - left, delta pp")
    b2 = ax2.bar(x + w / 2, mm_f, w, yerr=ms_f, color=MAE_C, capsize=3,
                 label="MAE (average error) - right, rel %")
    for i in range(len(algos)):
        ax.text(i - w / 2, pm_f[i] - 0.18, f"{pm_f[i]:+.1f}", ha="center", va="top",
                fontsize=9, color=PAPE_C)
        ax2.text(i + w / 2, mm_f[i] + 0.05, f"{mm_f[i]:+.1f}%", ha="center", va="bottom",
                 fontsize=9, color="#9a6a00")
    ax.set_xticks(x); ax.set_xticklabels(algos)
    ax.set_ylabel("PAPE codebook effect (delta pp)   after - before", color=PAPE_C)
    ax2.set_ylabel("MAE codebook effect (relative %)", color="#9a6a00")
    ax.tick_params(axis="y", labelcolor=PAPE_C)
    ax2.tick_params(axis="y", labelcolor="#9a6a00")
    pp_lo = float((pm_f - ps_f).min())
    ax.set_ylim(pp_lo - 1.0, 1.5)
    ax2.set_ylim(0, max(2.0, float((mm_f + ms_f).max()) + 0.4))
    _align_zero(ax, ax2)
    ax.set_title(f"Codebook: large PAPE gain, negligible MAE cost\n{aux_label}, R={max(rounds)}, E=5, {len(seeds)}-seed")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(handles=[b1, b2], loc="lower right", frameon=True, framealpha=0.85,
              facecolor="white", edgecolor="none", fontsize=10.5)
    fig.tight_layout()
    out1 = figdir / f"v09_{tag}_codebook_pape_vs_mae_pct.png"
    fig.savefig(out1, dpi=160); plt.close(fig); print(f"saved: {out1}")

    # Fig2 — per-round Δ (mean over algos, mean +/- std over seeds), dual axis.
    def _agg(eff):
        per_seed = eff.mean(axis=0)
        return per_seed.mean(0), per_seed.std(0, ddof=1)
    pm, ps = _agg(pap_eff); mm, ms = _agg(mae_eff)
    fig, ax = plt.subplots(figsize=(8.4, 5.0)); ax.axhline(0, color="#888", lw=1.0)
    ax2 = ax.twinx()
    l1, = ax.plot(rounds, pm, "-o", color=PAPE_C, lw=2.2, ms=4,
                  label="PAPE (peak error) - left, delta pp")
    ax.fill_between(rounds, pm - ps, pm + ps, color=PAPE_C, alpha=0.15)
    l2, = ax2.plot(rounds, mm, "-o", color=MAE_C, lw=2.2, ms=4,
                   label="MAE (average error) - right, rel %")
    ax2.fill_between(rounds, mm - ms, mm + ms, color=MAE_C, alpha=0.15)
    ax.set_xlabel("Round")
    ax.set_ylabel("PAPE codebook effect (delta pp)   after - before", color=PAPE_C)
    ax2.set_ylabel("MAE codebook effect (relative %)", color="#9a6a00")
    ax.tick_params(axis="y", labelcolor=PAPE_C)
    ax2.tick_params(axis="y", labelcolor="#9a6a00")
    ax2.axhspan(-2, 2, color="gray", alpha=0.07)
    ax2.text(max(rounds), 1.0, "MAE +-2% band", ha="right", va="center", fontsize=9, color="#888")
    ax.set_title(f"Per-round codebook effect: PAPE large drop, MAE ~+1% (negligible)\n{aux_label}, mean over 5 algos, {len(seeds)}-seed, R={max(rounds)}, E=5")
    ax.set_xticks([t for t in (1, 5, 10, 15, 20) if t <= max(rounds)]); ax.grid(True, alpha=0.3)
    ax2.set_ylim(-max(3.0, float((mm + ms).max()) + 1.0), max(3.0, float((mm + ms).max()) + 1.0))
    _align_zero(ax, ax2)
    ax.legend(handles=[l1, l2], loc="center right", frameon=True, framealpha=0.85,
              facecolor="white", edgecolor="none", fontsize=10.5)
    fig.tight_layout()
    out2 = figdir / f"v09_{tag}_codebook_effect_per_round.png"
    fig.savefig(out2, dpi=160); plt.close(fig); print(f"saved: {out2}")

    # Fig3 — relative-% variant (both PAPE & MAE as relative %, single axis).
    # Kept for reference; pp version (Fig1) is the recommended headline figure.
    pap_rel = (pa - pb) / pb * 100.0
    prm_f, prs_f = pap_rel[:, :, -1].mean(1), pap_rel[:, :, -1].std(1, ddof=1)
    fig, ax = plt.subplots(figsize=(8.4, 5.0)); ax.axhline(0, color="#888", lw=1.0)
    ax.bar(x - w / 2, prm_f, w, yerr=prs_f, color=PAPE_C, capsize=3, label="PAPE (peak error)")
    ax.bar(x + w / 2, mm_f, w, yerr=ms_f, color=MAE_C, capsize=3, label="MAE (average error)")
    for i in range(len(algos)):
        ax.text(i - w / 2, prm_f[i] - 0.6, f"{prm_f[i]:+.0f}%", ha="center", va="top", fontsize=9, color=PAPE_C)
        ax.text(i + w / 2, mm_f[i] + 0.4, f"{mm_f[i]:+.1f}%", ha="center", va="bottom", fontsize=9, color="#9a6a00")
    ax.set_xticks(x); ax.set_xticklabels(algos)
    ax.set_ylabel("Codebook relative effect (%)   after vs before")
    ax.set_title(f"Codebook: large PAPE gain, negligible MAE cost (relative %)\n{aux_label}, R={max(rounds)}, E=5, {len(seeds)}-seed")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", frameon=True, framealpha=0.85, facecolor="white", edgecolor="none", fontsize=11)
    fig.tight_layout()
    out3 = figdir / f"v09_{tag}_codebook_pape_vs_mae_relpct.png"
    fig.savefig(out3, dpi=160); plt.close(fig); print(f"saved: {out3}")


# Peak-vs-average (codebook MAE trade-off) figures, both aux conditions.
make_codebook_effect_figures("v09_round_vq_codebook_R20", "R20", "aux=0.3")
make_codebook_effect_figures("v09_round_vq_codebook_R20_MAEonly", "R20_MAEonly", "aux=0 (MAE-only)")


def make_fedvq_vs_roundcb_figures(ns_fedvq, ns_roundcb, tag, seeds=(42, 123, 7),
                                  fedvq_cell="V9-FedVQ-aux-fedinit"):
    """Controlled comparison (same FedAvg backbone, aux=0.3, M=32, R=20, E=5):
      - RoundCB before  : no codebook (V9-RoundCB-FedAvg test_before)
      - RoundCB after   : post-hoc per-round codebook (test_after)
      - FedVQ           : codebook in forward path (test from result.json history)
    Fig1: per-round test PAPE, 3 curves (3-seed mean +/- std).
    Fig2: FedVQ codebook health (active codes + perplexity per round)."""
    import numpy as np
    figdir = BASE / "outputs" / ns_fedvq / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    rcb_before, rcb_after, rounds = [], [], None
    for s in seeds:
        rows = [json.loads(l) for l in
                (BASE / "outputs" / ns_roundcb / f"seed{s}" / "V9-RoundCB-FedAvg" / "codebook_log.jsonl").open()]
        rounds = [r["round"] for r in rows]
        rcb_before.append([r["test_before"]["pape_mean"] for r in rows])
        rcb_after.append([r["test_after"]["pape_mean"] for r in rows])
    rcb_before, rcb_after = np.array(rcb_before), np.array(rcb_after)

    fv_test, fv_active, fv_ppl = [], [], []
    for s in seeds:
        res = json.loads((BASE / "outputs" / ns_fedvq / f"seed{s}" / fedvq_cell / "result.json").read_text())
        hist = res["history"]
        fv_test.append([h["test"]["pape_mean"] for h in hist])
        fv_active.append([h["server_vq"]["ema_count_active"] for h in hist])
        fv_ppl.append([h["train"]["vq_ppl_mean"] for h in hist])
    fv_test, fv_active, fv_ppl = np.array(fv_test), np.array(fv_active), np.array(fv_ppl)

    def ms(a):
        return a.mean(0), a.std(0, ddof=1)

    # Fig1 — per-round test PAPE comparison.
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for arr, color, ls, lbl in [
        (rcb_before, "#888888", "--", "RoundCB before (no codebook)"),
        (rcb_after,  "#0072B2", "-",  "RoundCB after (post-hoc codebook)"),
        (fv_test,    "#D55E00", "-",  "FedVQ (codebook in forward)"),
    ]:
        m, sd = ms(arr)
        ax.plot(rounds, m, ls, color=color, lw=2.2, marker="o", markersize=4, label=lbl)
        ax.fill_between(rounds, m - sd, m + sd, color=color, alpha=0.15)
    ax.set_xlabel("Round"); ax.set_ylabel("Across-client TEST PAPE (%)")
    ax.set_title("FedVQ vs RoundCB (same FedAvg backbone, aux=0.3, M=32)\nmean +/- std over 3 seeds, R=20, E=5")
    ax.set_xticks([1, 5, 10, 15, 20]); ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, framealpha=0.85, facecolor="white", edgecolor="none", fontsize=10.5)
    fig.tight_layout()
    out1 = figdir / f"v09_{tag}_fedvq_vs_roundcb_test_pape.png"
    fig.savefig(out1, dpi=160); plt.close(fig); print(f"saved: {out1}")

    # Fig2 — FedVQ codebook health.
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    am, asd = ms(fv_active)
    ax.plot(rounds, am, "-o", color="#009E73", lw=2.2, ms=4, label="active codes (of 32)")
    ax.fill_between(rounds, am - asd, am + asd, color="#009E73", alpha=0.15)
    ax.axhline(32, color="#009E73", ls=":", lw=1.0, alpha=0.6)
    ax.set_xlabel("Round"); ax.set_ylabel("active codes (ema_count > 1e-3)", color="#009E73")
    ax.tick_params(axis="y", labelcolor="#009E73"); ax.set_ylim(0, 34)
    ax.set_xticks([1, 5, 10, 15, 20]); ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    pm, psd = ms(fv_ppl)
    ax2.plot(rounds, pm, "-s", color="#CC79A7", lw=2.2, ms=4, label="train perplexity")
    ax2.fill_between(rounds, pm - psd, pm + psd, color="#CC79A7", alpha=0.15)
    ax2.set_ylabel("train perplexity", color="#CC79A7"); ax2.tick_params(axis="y", labelcolor="#CC79A7")
    ax.set_title("FedVQ codebook health: partial collapse + respawn churn\nmean +/- std over 3 seeds, R=20 (respawn every 5 rounds)")
    l1, lab1 = ax.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="lower right", frameon=True, framealpha=0.85,
              facecolor="white", edgecolor="none", fontsize=10.5)
    fig.tight_layout()
    out2 = figdir / f"v09_{tag}_fedvq_codebook_health.png"
    fig.savefig(out2, dpi=160); plt.close(fig); print(f"saved: {out2}")


# FedVQ (in-forward) vs RoundCB (post-hoc) controlled comparison.
make_fedvq_vs_roundcb_figures("v09_round_vq_codebook_R20_FedVQ", "v09_round_vq_codebook_R20", "R20")
