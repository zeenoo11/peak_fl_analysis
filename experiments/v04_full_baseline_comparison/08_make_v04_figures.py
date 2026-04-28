"""v04 figures — F1 Pareto, F2 G5 cross-cell, F3 NF/FM vs trained, F4 G6 het, F5 G7 comm.

Reads ``outputs/v04_full_baseline_comparison/multiseed_summary.json`` (from
07_aggregate.py) plus the seed-independent ``heterogeneity_summary.json``
and ``communication_summary.json``. Writes PNGs under
``papers/v04_draft/figures/``.

CLI (no per-seed argument):

    uv run python experiments/v04_full_baseline_comparison/08_make_v04_figures.py
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

V04_OUT_ROOT = OUTPUT_DIR / "v04_full_baseline_comparison"
PAPERS_FIG = (
    Path(__file__).resolve().parents[2] / "papers" / "v04_draft" / "figures"
)


def _load(p: Path) -> dict | None:
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


# Display labels + colour groupings.
LABELS = {
    "fedavg":              ("FedAvg",        "FL"),
    "fedprox":             ("FedProx",       "FL"),
    "fedrep":              ("FedRep",        "FL"),
    "ditto":               ("Ditto",         "FL"),
    "local_only":          ("Local-only",    "no-FL"),
    "nf_dlinear":          ("DLinear",       "NF"),
    "nf_nhits":            ("NHITS",         "NF"),
    "nf_crossformer":      ("Crossformer",   "NF"),
    "fm_chronos_bolt_small": ("Chronos-Bolt", "FM"),
    "fm_chronos_t5_tiny":    ("Chronos-T5",   "FM"),
    "fm_timesfm":            ("TimesFM",      "FM"),
    "peakvq_on_fedavg":    ("Peak-VQ on FedAvg", "G5"),
    "peakvq_on_fedrep":    ("Peak-VQ on FedRep", "G5"),
}
GROUP_COLOR = {
    "FL":   "tab:blue",
    "no-FL":"tab:gray",
    "NF":   "tab:green",
    "FM":   "tab:purple",
    "G5":   "tab:red",
}


def f1_pareto_pape_hr1(summary: dict, out: Path) -> None:
    """F1 — PAPE × HR@1 Pareto across all baselines + ours."""
    fig, ax = plt.subplots(figsize=(10, 7))
    legend_seen: set[str] = set()
    for m in summary["methods"]:
        agg = m["agg"]
        if agg["pape"]["n"] == 0:
            continue
        x = agg["pape"]["mean"]
        y = agg["hr@1"]["mean"]
        x_err = agg["pape"]["std"] if agg["pape"]["n"] > 1 else 0
        y_err = agg["hr@1"]["std"] if agg["hr@1"]["n"] > 1 else 0
        label_full = LABELS.get(m["method"], (m["method"], "?"))
        label, group = label_full
        color = GROUP_COLOR.get(group, "black")
        legend_label = group if group not in legend_seen else None
        legend_seen.add(group)
        ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt="o", color=color, alpha=0.85,
                    label=legend_label, capsize=3, markersize=8)
        ax.annotate(label, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Cold PAPE (kW) — lower better")
    ax.set_ylabel("Cold HR@1 (%) — higher better")
    ax.set_title("F1. v04 Pareto: cold PAPE × HR@1 across baselines (mean ± std, 3 seeds)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.name}")


def f2_g5_cross_cell(summary: dict, out: Path) -> None:
    """F2 — Peak-VQ delta on FedAvg / FedRep backbone."""
    pairs = [("fedavg", "peakvq_on_fedavg"), ("fedrep", "peakvq_on_fedrep")]
    rows = []
    for backbone, peakvq in pairs:
        b = next((m for m in summary["methods"] if m["method"] == backbone), None)
        p = next((m for m in summary["methods"] if m["method"] == peakvq), None)
        if b is None or p is None:
            continue
        if b["agg"]["pape"]["n"] == 0 or p["agg"]["pape"]["n"] == 0:
            continue
        rows.append({
            "name": LABELS[backbone][0],
            "backbone_pape": b["agg"]["pape"]["mean"],
            "backbone_pape_std": b["agg"]["pape"]["std"],
            "peakvq_pape": p["agg"]["pape"]["mean"],
            "peakvq_pape_std": p["agg"]["pape"]["std"],
        })
    if not rows:
        print("  F2 skipped (no G5 data)")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(rows))
    w = 0.36
    ax.bar(x - w/2, [r["backbone_pape"] for r in rows], w,
           yerr=[r["backbone_pape_std"] for r in rows], capsize=4,
           color="tab:blue", alpha=0.8, label="FL backbone (no Peak-VQ)", edgecolor="black", linewidth=0.6)
    ax.bar(x + w/2, [r["peakvq_pape"] for r in rows], w,
           yerr=[r["peakvq_pape_std"] for r in rows], capsize=4,
           color="tab:red", alpha=0.8, label="FL backbone + Peak-VQ (PAPE-aggressive)", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels([r["name"] for r in rows])
    ax.set_ylabel("Cold PAPE (kW)")
    ax.set_title("F2. G5 cross-cell — Peak-VQ delta on top of FL backbone")
    for i, r in enumerate(rows):
        delta = r["backbone_pape"] - r["peakvq_pape"]
        ax.text(i, max(r["backbone_pape"], r["peakvq_pape"]) + 1.0,
                f"Δ = {delta:+.2f} kW", ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y"); ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> {out.name}")


def f3_nf_fm_vs_trained(summary: dict, out: Path) -> None:
    """F3 — NF / FM cold PAPE vs trained baselines."""
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = []
    for m in summary["methods"]:
        if LABELS.get(m["method"], (None, ""))[1] in {"FL", "no-FL", "NF", "FM"}:
            if m["agg"]["pape"]["n"] > 0:
                methods.append(m)
    methods = sorted(methods, key=lambda m: m["agg"]["pape"]["mean"])
    names = [LABELS[m["method"]][0] for m in methods]
    means = [m["agg"]["pape"]["mean"] for m in methods]
    stds = [m["agg"]["pape"]["std"] for m in methods]
    colors = [GROUP_COLOR[LABELS[m["method"]][1]] for m in methods]
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Cold PAPE (kW)")
    ax.set_title("F3. v04 baselines — cold PAPE (mean ± std, 3 seeds; sorted by PAPE)")
    # Group legend
    handles = [plt.Rectangle((0,0),1,1, color=GROUP_COLOR[g], alpha=0.85) for g in ["FL", "no-FL", "NF", "FM"]]
    ax.legend(handles, ["FL", "no-FL (Local)", "NF (trained)", "FM (zero-shot)"], loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> {out.name}")


def f4_heterogeneity(out: Path) -> None:
    """F4 — heterogeneity heatmap + correlation scatter (G6)."""
    p = V04_OUT_ROOT / "heterogeneity_summary.json"
    if not p.exists():
        print(f"  F4 skipped (no {p.name})")
        return
    with open(p) as fh:
        d = json.load(fh)
    W = np.asarray(d["pairwise"]["W1"], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    im = axes[0].imshow(W, cmap="viridis", aspect="auto")
    axes[0].set_title(f"F4a. Pairwise W1 over {d['n_apts']} train apts")
    axes[0].set_xlabel("apt index"); axes[0].set_ylabel("apt index")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    corr = d.get("correlation")
    if corr:
        het = corr["per_apt_heterogeneity"]
        gap = corr["per_apt_gap"]
        valid = [(h, g) for h, g in zip(het, gap) if g is not None and not np.isnan(g)]
        if valid:
            xs, ys = zip(*valid)
            axes[1].scatter(xs, ys, alpha=0.7, color="tab:blue")
            axes[1].set_xlabel("Apt heterogeneity (mean W1 to others)")
            axes[1].set_ylabel("Local-only PAPE − Shared(FedAvg) PAPE  (kW)")
            axes[1].set_title(
                f"F4b. Het vs (Local−Shared) gap, "
                f"r={corr['pearson_corr_heterogeneity_vs_localShared_gap']:+.3f} "
                f"(n={corr['n_valid_apts']})"
            )
            axes[1].axhline(0, ls="--", c="gray", lw=1)
            axes[1].grid(alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "no valid pairs", ha="center", va="center", transform=axes[1].transAxes)
    else:
        axes[1].text(0.5, 0.5, "correlation block missing\n(needs local_only + fedavg results)",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()

    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> {out.name}")


def f5_communication(out: Path) -> None:
    """F5 — communication-cost table as a horizontal bar (log scale)."""
    p = V04_OUT_ROOT / "communication_summary.json"
    if not p.exists():
        print(f"  F5 skipped (no {p.name})")
        return
    with open(p) as fh:
        d = json.load(fh)
    rows = sorted(d["methods"], key=lambda m: m.get("total_bytes", 0))
    names = [m["method"] for m in rows]
    bytes_total = [max(m["total_bytes"], 1) for m in rows]   # log-safe
    crosses = [m["boundary_crosses"] for m in rows]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.barh(names, bytes_total, color="tab:orange", alpha=0.85, edgecolor="black", linewidth=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Total upload bytes (log scale)")
    ax.set_title("F5. G7 communication cost — total upload bytes & boundary crosses")
    for bar, m, c in zip(bars, rows, crosses):
        ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
                f"{m['total_bytes']:,} B   ({c} crosses)",
                va="center", fontsize=8.5)
    ax.grid(alpha=0.3, axis="x", which="both")
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="v04 figures.")
    ap.add_argument("--out_dir", type=Path, default=PAPERS_FIG)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[v04 fig] out_dir = {args.out_dir}")
    summary = _load(V04_OUT_ROOT / "multiseed_summary.json")
    if summary is None:
        raise SystemExit("missing multiseed_summary.json — run 07_aggregate.py first")

    f1_pareto_pape_hr1(summary, args.out_dir / "v04_F1_pareto.png")
    f2_g5_cross_cell(summary, args.out_dir / "v04_F2_g5_cross_cell.png")
    f3_nf_fm_vs_trained(summary, args.out_dir / "v04_F3_nf_fm_vs_trained.png")
    f4_heterogeneity(args.out_dir / "v04_F4_heterogeneity.png")
    f5_communication(args.out_dir / "v04_F5_communication.png")


if __name__ == "__main__":
    main()
