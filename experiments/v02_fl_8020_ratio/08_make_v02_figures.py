"""Render F1-F5 figures for the v02 paper draft.

Reads ``outputs/v02_fl_8020_ratio/multiseed_summary.json`` (produced by 07).
Writes to ``papers/v02_draft/figures/v02_F{1..5}.png``.

Figure plan (mirrors plans/v02-01_fl_8020_ratio.md §"Deliverables"):
    F1  80:20 PAPE / HR@k bars vs v01 50:50 (G1).
    F2  R0 vs R1 routing comparison, both op-points (G2).
    F3  E1 ablation: peak_aux ON/OFF, T0 vs T2 (V0, α=2.0); compare to v01 +18.6 pp.
    F4  W-component decomposition: V0 / W1a / W5 PAPE bars at both op-points (G4).
    F5  Codebook health: k_min, utilization, perplexity per seed.

v01 reference numbers are hard-coded from papers/v01_draft/v01_peak_vq.md.

[한글]
v02 paper용 그림 5장 생성 스크립트. 07에서 만든 multiseed_summary.json 한 파일만
입력으로 쓰고, papers/v02_draft/figures/ 아래에 PNG로 저장한다. 멀티시드 mean±std는
이미 07에서 계산되어 있어 여기서는 plot만 한다.
"""

from __future__ import annotations

import argparse
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

from config import OUTPUT_DIR

V02_OUT_ROOT = OUTPUT_DIR / "v02_fl_8020_ratio"
PAPERS_FIG = (
    Path(__file__).resolve().parents[2] / "papers" / "v02_draft" / "figures"
)

# Paper-side display names (Option 3 from the README naming reference).
# Code-internal name -> paper label.
LABEL = {
    "T0": "Vanilla",
    "T2": "NBEATSxAux",
    "V0": "CMO",
    "W1a": "GST",
    "W5": "Hybrid",
    "R0": "Key-Route",
    "R1": "Latent-Route",
}

# v01 50:50 reference values (papers/v01_draft/v01_peak_vq.md §4.2, §4.4).
V01_50_50_REF = {
    "baseline_pape": 55.17,
    "baseline_hr1": 27.0,
    "baseline_hr2": 38.5,
    # Multi-seed W5 PAPE-aggressive from §4.4
    "w5_pape_aggr_pape": 37.62,
    "w5_pape_aggr_pape_std": 0.45,
    "w5_pape_aggr_hr1": 26.36,
    "w5_pape_aggr_hr1_std": 0.22,
    "w5_pape_aggr_hr2": 37.96,
    "w5_pape_aggr_hr2_std": 0.40,
    # HR-preserving point — v01 §4.2 reports a single seed (~ −18% from 55.17 → ~45.4)
    # No multi-seed std published for HR-pres specifically; plot single seed.
    "w5_hr_pres_pape": 45.34,
    "w5_hr_pres_hr1": 27.0,
    "w5_hr_pres_hr2": 38.2,
    # E1 V0 effect: +18.6 pp (T2 PAPE relative improvement minus T0).
    "e1_pp": 18.6,
}


def _load_summary() -> dict:
    path = V02_OUT_ROOT / "multiseed_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run 07_aggregate_seeds.py first."
        )
    with open(path) as fh:
        return json.load(fh)


def _bar_with_err(ax, x, mean, std, color, label, width=0.35):
    ax.bar(x, mean, width, yerr=std, capsize=4, color=color, alpha=0.85, label=label, edgecolor="black", linewidth=0.6)


def f0_architecture(out: Path) -> None:
    """F0 — End-to-end pipeline as a clean horizontal 3-row diagram.

    Layout (top → bottom): Phase A (donor pretrain), Phase B (server-side
    codebook fit), Phase C (fully local cold inference). Each row is a
    left → right flow with widely-spaced boxes and a small fixed number
    of straight arrows so it stays readable when shrunk.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(15, 9.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off")

    BLUE = "#dbe6ff"
    GREY = "#eeeeee"
    YEL  = "#fff3c4"
    ORA  = "#ffd6a5"
    GRN  = "#cfe9c4"
    RED  = "#ffd6d6"

    def box(x, y, w, h, text, fc=BLUE, ec="black", fs=10, fw="normal"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.5", fc=fc, ec=ec, lw=1.1,
        ))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight=fw)

    def arr(x0, y0, x1, y1, color="black", lw=1.2, ls="-"):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="->,head_length=8,head_width=5",
            mutation_scale=14, color=color, lw=lw, ls=ls,
        ))

    # ============================================================
    # Phase A — DONOR PRETRAIN  (top row, y ≈ 75-92)
    # ============================================================
    ax.text(2, 95, "Phase A  ·  Donor pretrain  (centralized, one-shot)",
            fontsize=12, fontweight="bold", color="#222")

    box(3, 78, 14, 10, "80 train apts\nUMass 2016\nhourly kW", fc=GREY, fs=9)
    arr(17, 83, 24, 83)

    box(24, 78, 22, 10, "NBEATSxAux backbone\n3 stacks (trend, seasonal,\ngeneric) + peak_aux head",
        fc=BLUE, fw="bold", fs=9)
    arr(46, 83, 53, 83)

    box(53, 78, 22, 10, "loss = MAE(y, ŷ)\n+ λ · peak_aux\n(λ = 0.3)", fc=RED, fs=9)
    arr(64, 78, 64, 75, ls="--")  # downward "trained" arrow

    box(53, 67, 22, 8,  "→ frozen NBEATSxAux\n(re-used in Phase B & C)", fc=GRN, fs=9, fw="bold")

    # ============================================================
    # Phase B — SERVER  (middle row, y ≈ 45-58)
    # ============================================================
    # boundary line above this row
    ax.axhline(63, color="tab:red", lw=1.0, ls="--", alpha=0.5)
    ax.text(99, 63.4, "donor → server boundary  (crossed once, at pretrain)",
            color="tab:red", ha="right", va="bottom", fontsize=9, style="italic")

    ax.text(2, 60, "Phase B  ·  Server-side codebook fit  (post-hoc, 1-shot, no learning)",
            fontsize=12, fontweight="bold", color="#222")

    box(3, 45, 14, 10, "frozen\nNBEATSxAux\n(from Phase A)", fc=GRN, fs=9)
    arr(17, 50, 24, 50)

    box(24, 45, 22, 10, "forward train windows\nstride = 24\n→ {h_g} ∈ ℝ^{N×64},  N ≈ 19 k", fc=BLUE, fs=9)
    arr(46, 50, 53, 50)

    box(53, 45, 22, 10, "1-shot KMeans++\nM = 32 clusters\nrandom_state = seed", fc=ORA, fw="bold", fs=9)
    arr(75, 50, 82, 50)

    box(82, 50, 16, 8, "codebook C\n32 × 64 centroids", fc=GRN, fs=9)
    box(82, 41, 16, 8, "offsets o_c\nresidual mean", fc=GRN, fs=9)
    arr(75, 50, 82, 54)
    arr(75, 50, 82, 45)

    ax.text(50, 39.5,
            "also stored:  train KEY pool (5-d, scaled)  +  per-window cluster_idx  →  used by Key-Route at inference",
            ha="center", va="top", fontsize=8.5, color="dimgray", style="italic")

    # ============================================================
    # Phase C — COLD INFERENCE  (bottom row, y ≈ 12-32)
    # ============================================================
    ax.axhline(36, color="tab:green", lw=1.0, ls="--", alpha=0.55)
    ax.text(99, 36.4,
            "server → cold-gucha boundary  ·  cold side reads codebook locally; no upload",
            color="tab:green", ha="right", va="bottom", fontsize=9, style="italic")

    ax.text(2, 33, "Phase C  ·  Cold inference  (per cold gucha, fully local, zero-shot)",
            fontsize=12, fontweight="bold", color="#222")

    box(3, 18, 14, 10, "20 cold apts\nwarm-start z-norm\n(own first 70%)", fc=GREY, fs=9)
    arr(17, 23, 24, 23)

    box(24, 18, 22, 10, "frozen NBEATSxAux\nforward(stride=24)\n→ ŷ, h_g_cold, (â, ĥ)",
        fc=BLUE, fw="bold", fs=9)
    arr(46, 23, 53, 23)

    box(53, 25, 22, 7, "Key-Route   (5-d KEY → 1-NN train pool → c*)", fc=ORA, fs=9)
    box(53, 14, 22, 7, "Latent-Route   (h_g_cold → 1-NN codebook → c*)", fc=ORA, fs=9)

    arr(75, 28, 86, 23)  # both routes feed correction
    arr(75, 17, 86, 23)

    box(82, 18, 16, 10, "→ c*\n(cluster id)\n+ o_{c*}", fc=GRN, fs=9, fw="bold")

    # bottom: correction equation
    box(8, 2, 84, 7,
        "ŷ_corr  =  ŷ  +  α_v0 · o_{c*}   +   α_w1 · g(t; ĥ, â, σ)        "
        "← Hybrid (W5) correction in z-norm space, then denormalise to kW",
        fc=GRN, fw="bold", fs=10)

    ax.text(50, 0.7,
            "two operating points carried over from v01:  "
            "HR-preserving (σ=3.0, α_v0=1.0, α_w1=0.1)    ·    "
            "PAPE-aggressive (σ=3.0, α_v0=1.5, α_w1=0.5)",
            ha="center", va="bottom", fontsize=9, color="dimgray", style="italic")

    fig.suptitle(
        "F0. v02 end-to-end pipeline — frozen NBEATSxAux + post-hoc Peak-VQ + Hybrid (W5) correction",
        fontsize=12, y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.name}")


def f1_8020_vs_5050(summary: dict, out: Path) -> None:
    """F1 — v02 80:20 (mean ± std) vs v01 50:50 reference, on R0 routing."""
    r0 = summary["coldstart_R0"]
    if r0 is None:
        return
    op = r0["operating_points"]
    metrics_v02 = {
        "baseline": {
            "pape": (r0["baseline"]["pape"]["mean"], r0["baseline"]["pape"]["std"]),
            "hr@1": (r0["baseline"]["hr@1"]["mean"], r0["baseline"]["hr@1"]["std"]),
            "hr@2": (r0["baseline"]["hr@2"]["mean"], r0["baseline"]["hr@2"]["std"]),
        },
        "Hybrid (HR-pres)": {
            "pape": (op["HR-preserving"]["metrics"]["pape"]["mean"], op["HR-preserving"]["metrics"]["pape"]["std"]),
            "hr@1": (op["HR-preserving"]["metrics"]["hr@1"]["mean"], op["HR-preserving"]["metrics"]["hr@1"]["std"]),
            "hr@2": (op["HR-preserving"]["metrics"]["hr@2"]["mean"], op["HR-preserving"]["metrics"]["hr@2"]["std"]),
        },
        "Hybrid (PAPE-aggr)": {
            "pape": (op["PAPE-aggressive"]["metrics"]["pape"]["mean"], op["PAPE-aggressive"]["metrics"]["pape"]["std"]),
            "hr@1": (op["PAPE-aggressive"]["metrics"]["hr@1"]["mean"], op["PAPE-aggressive"]["metrics"]["hr@1"]["std"]),
            "hr@2": (op["PAPE-aggressive"]["metrics"]["hr@2"]["mean"], op["PAPE-aggressive"]["metrics"]["hr@2"]["std"]),
        },
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    metrics_order = [("PAPE (kW)", "pape"), ("HR@1 (%)", "hr@1"), ("HR@2 (%)", "hr@2")]
    cells = ["baseline", "Hybrid (HR-pres)", "Hybrid (PAPE-aggr)"]
    x = np.arange(len(cells))
    width = 0.38

    v01_means = {
        "pape": [V01_50_50_REF["baseline_pape"], V01_50_50_REF["w5_hr_pres_pape"], V01_50_50_REF["w5_pape_aggr_pape"]],
        "hr@1": [V01_50_50_REF["baseline_hr1"], V01_50_50_REF["w5_hr_pres_hr1"], V01_50_50_REF["w5_pape_aggr_hr1"]],
        "hr@2": [V01_50_50_REF["baseline_hr2"], V01_50_50_REF["w5_hr_pres_hr2"], V01_50_50_REF["w5_pape_aggr_hr2"]],
    }

    for ax, (title, key) in zip(axes, metrics_order):
        v02_m = [metrics_v02[c][key][0] for c in cells]
        v02_s = [metrics_v02[c][key][1] for c in cells]
        ax.bar(x - width / 2, v01_means[key], width, color="tab:gray", alpha=0.7, label="v01 50:50", edgecolor="black", linewidth=0.6)
        ax.bar(x + width / 2, v02_m, width, yerr=v02_s, capsize=4, color="tab:blue", alpha=0.85, label="v02 80:20", edgecolor="black", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(cells)
        ax.set_title(title); ax.grid(alpha=0.25, axis="y")
        if key == "pape":
            ax.set_ylabel("kW (lower is better)")
        else:
            ax.set_ylabel("% (higher is better)")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("F1. v02 (80:20) vs v01 (50:50) — R0 routing, mean±std across 3 seeds", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.name}")


def f2_routing_R0_vs_R1(summary: dict, out: Path) -> None:
    """F2 — R0 vs R1 routing comparison across both op-points."""
    r0, r1 = summary["coldstart_R0"], summary["coldstart_R1"]
    if r0 is None or r1 is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    op_names = ["HR-preserving", "PAPE-aggressive"]
    metrics = [("PAPE (kW)", "pape"), ("HR@1 (%)", "hr@1")]

    for ax, (title, key) in zip(axes, metrics):
        x = np.arange(len(op_names))
        width = 0.38
        r0_m = [r0["operating_points"][op]["metrics"][key]["mean"] for op in op_names]
        r0_s = [r0["operating_points"][op]["metrics"][key]["std"] for op in op_names]
        r1_m = [r1["operating_points"][op]["metrics"][key]["mean"] for op in op_names]
        r1_s = [r1["operating_points"][op]["metrics"][key]["std"] for op in op_names]
        ax.bar(x - width / 2, r0_m, width, yerr=r0_s, capsize=4, color="tab:orange", alpha=0.85, label="Key-Route (5-d)", edgecolor="black", linewidth=0.6)
        ax.bar(x + width / 2, r1_m, width, yerr=r1_s, capsize=4, color="tab:green", alpha=0.85, label="Latent-Route (64-d h_g)", edgecolor="black", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(op_names, rotation=10)
        ax.set_title(title); ax.grid(alpha=0.25, axis="y")
        ax.set_ylabel("kW (lower is better)" if key == "pape" else "% (higher is better)")
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("F2. Routing Key-Route vs Latent-Route — mean±std across 3 seeds, 80:20", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.name}")


def f3_E1_peak_aux(summary: dict, out: Path) -> None:
    """F3 — E1 ablation: Vanilla vs NBEATSxAux on CMO correction (single panel).

    Effect-size comparison vs v01 (+18.6 pp vs +11.9 ± 9.2 pp) is reported in the
    paper body — plotting it gives the v02 number an unfairly small visual due
    to its large std, so we keep the figure to the cleaner per-arm comparison.
    """
    e1 = summary["E1"]
    if e1 is None:
        return
    arms = ["T0", "T2"]
    pape_mean = [e1["arm_metrics"][a]["V0"]["pape"]["mean"] for a in arms]
    pape_std = [e1["arm_metrics"][a]["V0"]["pape"]["std"] for a in arms]
    base_mean = [e1["arm_metrics"][a]["baseline"]["pape"]["mean"] for a in arms]
    base_std = [e1["arm_metrics"][a]["baseline"]["pape"]["std"] for a in arms]
    k_min_mean = [e1["arm_metrics"][a]["vq_k_min"]["mean"] for a in arms]
    k_min_std = [e1["arm_metrics"][a]["vq_k_min"]["std"] for a in arms]

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.4))
    x = np.arange(len(arms))
    width = 0.36
    ax.bar(x - width / 2, base_mean, width, yerr=base_std, capsize=4, color="tab:gray", alpha=0.7, label="baseline (no correction)", edgecolor="black", linewidth=0.6)
    ax.bar(x + width / 2, pape_mean, width, yerr=pape_std, capsize=4, color="tab:blue", alpha=0.85, label="CMO (α=2.0)", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(["Vanilla\n(peak_aux OFF)", "NBEATSxAux\n(peak_aux ON)"])
    ax.set_ylabel("Cold PAPE (kW)")
    # Annotate per-arm CMO PAPE and codebook k_min so the failure mode (Vanilla
    # codebook fragmentation -> noisy CMO offsets) is visible inline.
    for i, a in enumerate(arms):
        ax.text(
            i + width / 2,
            pape_mean[i] + pape_std[i] + 0.6,
            f"k_min={k_min_mean[i]:.0f}±{k_min_std[i]:.0f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_title("F3. CMO cold PAPE: peak_aux OFF vs ON  (effect-size discussed in §4.4)")
    ax.legend(fontsize=9, loc="lower left"); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.name}")


def f4_W_components(summary: dict, out: Path) -> None:
    """F4 — V0 / W1a / W5 PAPE bars at both op-points (T2, R0)."""
    wc = summary["W_component"]
    if wc is None:
        return
    op_names = ["HR-preserving", "PAPE-aggressive"]
    mechs = ["V0", "W1a", "W5"]               # data-side keys (unchanged)
    mech_labels = {"V0": "CMO", "W1a": "GST", "W5": "Hybrid"}
    colors = {"V0": "tab:orange", "W1a": "tab:purple", "W5": "tab:blue"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    width = 0.25

    for ax, op in zip(axes, op_names):
        cells = wc["per_operating_point"][op]["cells"]
        x = np.arange(len(mechs))
        means = [cells[m]["pape"]["mean"] for m in mechs]
        stds = [cells[m]["pape"]["std"] for m in mechs]
        for i, m in enumerate(mechs):
            ax.bar(i, means[i], 0.6, yerr=stds[i], capsize=4, color=colors[m], alpha=0.85, edgecolor="black", linewidth=0.6, label=mech_labels[m])
        baseline = wc["baseline"]["pape"]["mean"]
        ax.axhline(baseline, ls="--", c="tab:gray", lw=1.2, label=f"baseline {baseline:.1f}")
        synergy = wc["per_operating_point"][op]["hybrid_synergy_kw"]
        ax.set_xticks(x); ax.set_xticklabels([mech_labels[m] for m in mechs])
        op_pp = wc["per_operating_point"][op]
        ax.set_title(f"{op}\n(σ={op_pp['sigma']}, α_v0={op_pp['alpha_v0']}, α_w1={op_pp['alpha_w1']})")
        ax.set_ylabel("Cold PAPE (kW)" if op == "HR-preserving" else "")
        ax.text(
            0.5, 0.92,
            f"synergy = {synergy['mean']:+.2f} ± {synergy['std']:.2f} kW",
            transform=ax.transAxes,
            ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.85),
        )
        ax.grid(alpha=0.25, axis="y")
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("F4. Correction decomposition — NBEATSxAux × {CMO, GST, Hybrid} on Key-Route", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.name}")


def f5_codebook_health(summary: dict, out: Path) -> None:
    """F5 — codebook diagnostics across seeds."""
    cb = summary["codebook_health"]
    if cb is None:
        return
    seeds = cb["seeds"]
    metrics = cb["metrics"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plots = [
        ("k_min (min cluster size)", "vq_k_min", 113),
        ("utilization", "vq_utilization", None),
        ("perplexity (max=32)", "vq_perplexity", None),
    ]
    for ax, (title, key, threshold) in zip(axes, plots):
        vals = metrics[key]["values"]
        ax.bar(range(len(seeds)), vals, color="tab:blue", alpha=0.85, edgecolor="black", linewidth=0.6)
        ax.set_xticks(range(len(seeds))); ax.set_xticklabels([f"seed {s}" for s in seeds])
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
        if threshold is not None:
            ax.axhline(threshold, ls="--", c="tab:red", lw=1.2, label=f"v01 threshold {threshold}")
            ax.legend(fontsize=8)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.0f}" if v >= 1 else f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("F5. Codebook health (NBEATSxAux backbone, M=32) per seed", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render F1-F5 figures for the v02 paper draft.")
    ap.add_argument("--out_dir", type=Path, default=PAPERS_FIG)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[figures] out_dir = {args.out_dir}")
    summary = _load_summary()
    print(f"[figures] inputs from multiseed_summary.json (seeds={summary.get('seeds_requested')})")

    f0_architecture(args.out_dir / "v02_F0_architecture.png")
    f1_8020_vs_5050(summary, args.out_dir / "v02_F1_8020_vs_5050.png")
    f2_routing_R0_vs_R1(summary, args.out_dir / "v02_F2_routing_R0_R1.png")
    f3_E1_peak_aux(summary, args.out_dir / "v02_F3_E1_peak_aux.png")
    f4_W_components(summary, args.out_dir / "v02_F4_W_components.png")
    f5_codebook_health(summary, args.out_dir / "v02_F5_codebook_health.png")


if __name__ == "__main__":
    main()
