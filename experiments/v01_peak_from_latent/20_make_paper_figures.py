"""Generate paper figures from v01 results.

F1: ablation_e1.png - peak_aux ON/OFF bar chart
F2: multiseed_e3.png - 3-seed PAPE/HR with error bars
F3: arms_pareto.png - all arms PAPE × HR@1 Pareto from iter4
F4: cluster_benefit_e4.png - cluster scatter (already exists, copy to paper figures)
F5: training_curves.png - T2 training history
F6: peak_hour_dist.png - cold true peak hour distribution
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "outputs" / "v01_peak_from_latent"
PAPER_FIG = Path(__file__).resolve().parents[2] / "papers" / "figures"
PAPER_FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "savefig.dpi": 200, "figure.dpi": 100,
})


def load(path):
    return json.load(open(path, encoding="utf-8")) if path.exists() else None


# ─── F1: peak_aux ablation bar ─────────────────────────────────────────────
def fig1_ablation():
    e1 = load(OUT / "E1" / "E1_results.json")
    if e1 is None:
        print("  [F1] E1 missing"); return

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, mech in zip(axes, ["V0", "W5"]):
        t0 = e1["T0"][mech]
        t2 = e1["T2"][mech]
        x = np.arange(2)
        base = [t0["base_pape"], t2["base_pape"]]
        corr = [t0["corr_pape"], t2["corr_pape"]]
        w = 0.35
        ax.bar(x - w/2, base, w, label="base (no KV-VQ)", color="tab:gray", alpha=0.7, edgecolor="k")
        ax.bar(x + w/2, corr, w, label="cold-corrected", color="tab:blue", alpha=0.85, edgecolor="k")
        for i, (b, c) in enumerate(zip(base, corr)):
            delta = (c - b) / b * 100
            ax.text(i + w/2, c + 0.5, f"{delta:+.1f}%", ha="center", fontsize=8.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["T0\n(no peak_aux)", "T2\n(peak_aux)"])
        ax.set_ylabel("cold PAPE (kW)")
        ax.set_title(f"{mech} mechanism")
        ax.set_ylim(0, 60)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(loc="upper right")

    fig.suptitle("E1: peak_aux's isolated effect on cold-start PAPE", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PAPER_FIG / "F1_ablation_e1.png")
    plt.close(fig)
    print(f"  [F1] saved")


# ─── F2: multi-seed bar with error ─────────────────────────────────────────
def fig2_multiseed():
    e3 = load(OUT / "E3" / "E3_results.json")
    if e3 is None:
        print("  [F2] E3 missing"); return

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    metrics_pairs = [
        ("PAPE (kW)", ["base_pape", "corr_pape"]),
        ("HR@1 (%)", ["base_hr@1", "corr_hr@1"]),
        ("HR@2 (%)", ["base_hr@2", "corr_hr@2"]),
    ]
    s = e3["summary"]
    for ax, (title, (kb, kc)) in zip(axes, metrics_pairs):
        bm, bs_ = s[kb]["mean"], s[kb]["std"]
        cm, cs_ = s[kc]["mean"], s[kc]["std"]
        x = [0, 1]
        ax.bar(x, [bm, cm], yerr=[bs_, cs_], capsize=5,
               color=["tab:gray", "tab:blue"], alpha=0.85, edgecolor="k")
        ax.set_xticks(x); ax.set_xticklabels(["base", "Peak-VQ"])
        ax.set_ylabel(title); ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        ax.text(0, bm + bs_ + 0.5, f"{bm:.2f}±{bs_:.2f}", ha="center", fontsize=9)
        ax.text(1, cm + cs_ + 0.5, f"{cm:.2f}±{cs_:.2f}", ha="center", fontsize=9)

    fig.suptitle(f"E3: multi-seed stability (seeds {e3['seeds']})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PAPER_FIG / "F2_multiseed_e3.png")
    plt.close(fig)
    print(f"  [F2] saved")


# ─── F3: arms Pareto from iter4 ────────────────────────────────────────────
def fig3_arms_pareto():
    p = OUT / "iter4" / "iter4_results.json"
    if not p.exists():
        print("  [F3] iter4 missing"); return
    r = load(p)

    fig, ax = plt.subplots(figsize=(7, 5))
    base_pape = r["rows"][0]["base_pape"]; base_hr = r["rows"][0]["base_hr@1"]
    name_to_marker = {"V0": "o", "W1a": "s", "W1b": "^", "W3": "D", "W4": "X", "W5": "P", "W6": "*"}
    name_to_color = {}
    for row in r["rows"]:
        n = row["name"]
        family = n.split()[0]
        name_to_color.setdefault(family, plt.cm.tab10(len(name_to_color) % 10))
    for row in r["rows"]:
        n = row["name"]; family = n.split()[0]
        ax.scatter(row["corr_pape"], row["corr_hr@1"], s=70,
                   c=[name_to_color[family]], marker=name_to_marker.get(family, "o"),
                   alpha=0.85, edgecolors="k", label=n if n in ["V0 (M=32 α=2.0)",
                                                                "W5 α_w1=0.5", "W5 α_w1=0.3"] else None)
    ax.scatter(base_pape, base_hr, s=200, c="red", marker="*",
               label="NBEATSx baseline (no KV-VQ)", zorder=10, edgecolors="k", linewidth=1.2)

    # annotate W5 best
    best = min(r["rows"], key=lambda x: x["corr_pape"])
    ax.annotate(f"  ★ W5 best\n  ({best['corr_pape']:.2f}, {best['corr_hr@1']:.1f})",
                (best["corr_pape"], best["corr_hr@1"]),
                fontsize=9, fontweight="bold", color="darkblue")

    ax.set_xlabel("cold PAPE (kW) ← lower is better")
    ax.set_ylabel("cold HR@1 (%) → higher is better")
    ax.set_title("F3: Pareto frontier across mechanisms (iter4, T2 backbone)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(PAPER_FIG / "F3_arms_pareto.png")
    plt.close(fig)
    print(f"  [F3] saved")


# ─── F4: cluster benefit (already generated; just copy) ────────────────────
def fig4_cluster_benefit():
    src = OUT / "E4" / "figures" / "E4_cluster_benefit_map.png"
    if src.exists():
        shutil.copy(src, PAPER_FIG / "F4_cluster_benefit.png")
        print(f"  [F4] copied")
    else:
        print(f"  [F4] missing")


# ─── F5: T2 training history ────────────────────────────────────────────────
def fig5_training_curves():
    p = OUT / "T2" / "training_log.json"
    if not p.exists():
        print("  [F5] training_log missing"); return
    log = load(p)
    history = log.get("history", [])
    if not history:
        print("  [F5] empty history"); return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ep = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss", np.nan) for h in history]
    val_mae = [h.get("val_mae", np.nan) for h in history]
    val_pape = [h.get("val_pape", np.nan) for h in history]
    train_aux = [h.get("train_aux", np.nan) for h in history]

    axes[0].plot(ep, train_loss, "o-", label="total train loss", color="C0")
    if not all(np.isnan(train_aux)):
        axes[0].plot(ep, train_aux, "s--", label="aux loss", color="C1")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("training loss"); axes[0].grid(alpha=0.3); axes[0].legend()

    axes[1].plot(ep, val_mae, "o-", label="val MAE", color="C2")
    ax2 = axes[1].twinx()
    ax2.plot(ep, val_pape, "s--", label="val PAPE", color="C3")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val MAE", color="C2")
    ax2.set_ylabel("val PAPE", color="C3")
    axes[1].set_title("validation"); axes[1].grid(alpha=0.3)

    fig.suptitle(f"F5: T2 training history (50 train apts, peak_aux λ=0.3)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PAPER_FIG / "F5_training_curves.png")
    plt.close(fig)
    print(f"  [F5] saved")


# ─── F6: peak hour distribution ─────────────────────────────────────────────
def fig6_peak_hour_dist():
    src = OUT / "iter3" / "figures" / "hour_distribution.png"
    if src.exists():
        shutil.copy(src, PAPER_FIG / "F6_peak_hour_dist.png")
        print(f"  [F6] copied")
    else:
        print(f"  [F6] missing")


# ─── F0: architecture diagram (text-only block) ─────────────────────────────
def fig0_architecture_text():
    """Generate a simple architecture diagram via matplotlib boxes."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, label, color="white", fontsize=9.5):
        from matplotlib.patches import FancyBboxPatch
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                            edgecolor="black", facecolor=color, linewidth=1.2)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=fontsize)

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.3))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, label, fontsize=8.5, ha="center", style="italic")

    # Training pipeline (top row)
    box(0.2, 4.3, 1.5, 0.9, "input\nx ∈ ℝ⁹⁶", color="#E8F4FD")
    box(2.0, 4.3, 1.7, 0.9, "MinimalNBEATSx\n(3-stack)", color="#B6E2D3")
    box(4.0, 4.3, 1.7, 0.9, "h_generic\n∈ ℝ⁶⁴", color="#FFD8B1")
    box(6.0, 4.3, 1.7, 0.9, "AuxHead\nMLP", color="#FCE38A")
    box(7.9, 4.7, 1.0, 0.5, "amp", color="#F38181", fontsize=9)
    box(7.9, 4.0, 1.0, 0.5, "hour", color="#F38181", fontsize=9)
    arrow(1.7, 4.75, 2.0, 4.75)
    arrow(3.7, 4.75, 4.0, 4.75)
    arrow(5.7, 4.75, 6.0, 4.75)
    arrow(7.7, 4.85, 7.9, 4.95)
    arrow(7.7, 4.65, 7.9, 4.25)

    box(2.0, 3.3, 1.7, 0.7, "y_hat ∈ ℝ²⁴", color="#B6E2D3", fontsize=9)
    arrow(2.85, 4.3, 2.85, 4.0)
    ax.text(2.85, 3.0, "MAE loss", ha="center", fontsize=8.5, style="italic")
    ax.text(8.4, 3.4, "peak_aux loss\nλ=0.3", ha="center", fontsize=8.5, style="italic", color="darkred")

    # Post-hoc VQ (middle row)
    box(0.2, 1.8, 2.5, 0.8, "{h_generic}_train (∼12K)\nfrom 50 train apts", color="#E8F4FD")
    box(3.2, 1.8, 1.7, 0.8, "KMeans++\nM=32", color="#A8DADC", fontsize=9)
    box(5.4, 1.8, 1.7, 0.8, "Codebook C\n+ offsets o_c", color="#FFD8B1", fontsize=9)
    arrow(2.7, 2.2, 3.2, 2.2)
    arrow(4.9, 2.2, 5.4, 2.2)
    ax.text(2.95, 1.5, "post-hoc, frozen", ha="center", fontsize=8, style="italic")

    # Cold inference (bottom row)
    box(0.2, 0.2, 1.5, 0.8, "cold input\nx_cold", color="#E8F4FD")
    box(2.0, 0.2, 1.7, 0.8, "extract KEY\n(5-d)", color="#FCE38A", fontsize=9)
    box(4.0, 0.2, 1.7, 0.8, "1-NN to train\nKEYs → c*", color="#F8B195", fontsize=9)
    box(6.0, 0.2, 2.0, 0.8, "ŷ + α·o_{c*} + β·g(ĥ,â)", color="#F67280", fontsize=9)
    arrow(1.7, 0.6, 2.0, 0.6)
    arrow(3.7, 0.6, 4.0, 0.6)
    arrow(5.7, 0.6, 6.0, 0.6)

    # Section labels
    ax.text(0.2, 5.4, "(a) Training", fontsize=11, fontweight="bold")
    ax.text(0.2, 2.7, "(b) Post-hoc Codebook fitting", fontsize=11, fontweight="bold")
    ax.text(0.2, 1.1, "(c) Cold-start inference", fontsize=11, fontweight="bold")

    fig.suptitle("F0: Peak-Aware VQ — System Architecture", fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(PAPER_FIG / "F0_architecture.png")
    plt.close(fig)
    print(f"  [F0] saved")


def main():
    print("[generate paper figures]")
    fig0_architecture_text()
    fig1_ablation()
    fig2_multiseed()
    fig3_arms_pareto()
    fig4_cluster_benefit()
    fig5_training_curves()
    fig6_peak_hour_dist()
    print(f"\n[done] saved to {PAPER_FIG}")
    for p in sorted(PAPER_FIG.glob("*.png")):
        print(f"  {p.name}: {p.stat().st_size//1024} KB")


if __name__ == "__main__":
    main()
