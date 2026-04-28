"""Render the unified pFL paper's figures from the parent v04 + 09_fix_rerun outputs.

Reads:
    outputs/v04_full_baseline_comparison/multiseed_summary.json          (parent v04)
    outputs/v04_full_baseline_comparison/09_fix_rerun/multiseed_summary.json
    outputs/v02_fl_8020_ratio/multiseed_summary.json                     (M=32 reference)

Writes (to papers/pfl_unified/figures/):
    F1_pareto_unified.png         — PAPE x HR@1 Pareto, includes ours, Stacked-Aux,
                                    FedProto, Local-only holdout/self, all FL/NF/FM
    F6_decomposition.png          — Three-way decomposition stacked bar
    F7_m_sensitivity.png          — Cluster-count sensitivity curve M in {8,16,32,64}
    F8_sorted_unified.png         — Sorted-by-PAPE bar chart with ours reference

CLI:

    uv run python papers/pfl_unified/figures/make_pfl_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

OUT_FIG = Path(__file__).resolve().parent
PARENT_V04 = ROOT / "outputs" / "v04_full_baseline_comparison" / "multiseed_summary.json"
FIX_RERUN = ROOT / "outputs" / "v04_full_baseline_comparison" / "09_fix_rerun" / "multiseed_summary.json"

# Reference rows (from v02 multiseed_summary, hard-coded since they don't change).
PROPOSED_PAPE_MEAN, PROPOSED_PAPE_STD = 35.70, 0.49
PROPOSED_HR_MEAN, PROPOSED_HR_STD = 26.3, 2.2

# Group colours, matching the unified paper's text references.
COLOR = {
    "ours": "#d62728",
    "Stacked-Aux": "#9467bd",
    "Stacked": "#e377c2",
    "FL": "#1f77b4",
    "pFL": "#17becf",
    "NF": "#2ca02c",
    "FM": "#bcbd22",
    "no-FL": "#8c564b",
    "no-FL-fair": "#7f7f7f",
}


def load_jsons():
    with open(PARENT_V04) as fh:
        v04 = json.load(fh)
    with open(FIX_RERUN) as fh:
        fix = json.load(fh)
    return v04, fix


def build_table_rows(v04, fix):
    """Assemble the unified Table 1 row list with (label, mean, std, hr_mean, hr_std, group)."""
    rows = []
    rows.append(("Proposed (centralised)", PROPOSED_PAPE_MEAN, PROPOSED_PAPE_STD,
                 PROPOSED_HR_MEAN, PROPOSED_HR_STD, "ours"))
    # Stacked-Aux from fix-rerun
    sa = fix["fedavg_nbeatsx_aux"]["with_codebook_PAPE_aggressive"]
    rows.append(("Stacked-Aux\n(fed NBEATSxAux + codebook)", sa["cold_pape"]["mean"], sa["cold_pape"]["std"],
                 sa["cold_hr@1"]["mean"], sa["cold_hr@1"]["std"], "Stacked-Aux"))
    # Stacked from parent v04
    for m in v04["methods"]:
        if m["method"] == "peakvq_on_fedavg":
            rows.append(("Codebook on FedAvg\n(Stacked)", m["agg"]["pape"]["mean"], m["agg"]["pape"]["std"],
                         m["agg"]["hr@1"]["mean"], m["agg"]["hr@1"]["std"], "Stacked"))
        if m["method"] == "peakvq_on_fedrep":
            rows.append(("Codebook on FedRep\n(Stacked)", m["agg"]["pape"]["mean"], m["agg"]["pape"]["std"],
                         m["agg"]["hr@1"]["mean"], m["agg"]["hr@1"]["std"], "Stacked"))
    # NF
    nf_label = {"nf_dlinear": "DLinear", "nf_crossformer": "Crossformer"}
    for m in v04["methods"]:
        if m["method"] in nf_label:
            rows.append((nf_label[m["method"]], m["agg"]["pape"]["mean"], m["agg"]["pape"]["std"],
                         m["agg"]["hr@1"]["mean"], m["agg"]["hr@1"]["std"], "NF"))
    # NHITS corrected from fix-rerun
    nh = fix["nf_nhits_fixed"]
    rows.append(("NHITS (corrected)", nh["cold_pape"]["mean"], nh["cold_pape"]["std"],
                 nh["cold_hr@1"]["mean"], nh["cold_hr@1"]["std"], "NF"))
    # FM
    fm_label = {"fm_chronos_bolt_small": "Chronos-Bolt", "fm_chronos_t5_tiny": "Chronos-T5",
                "fm_timesfm": "TimesFM"}
    for m in v04["methods"]:
        if m["method"] in fm_label:
            rows.append((fm_label[m["method"]], m["agg"]["pape"]["mean"], m["agg"]["pape"]["std"],
                         m["agg"]["hr@1"]["mean"], m["agg"]["hr@1"]["std"], "FM"))
    # FL
    fl_label = {"fedavg": "FedAvg", "fedprox": "FedProx", "fedrep": "FedRep", "ditto": "Ditto"}
    for m in v04["methods"]:
        if m["method"] in fl_label:
            rows.append((fl_label[m["method"]], m["agg"]["pape"]["mean"], m["agg"]["pape"]["std"],
                         m["agg"]["hr@1"]["mean"], m["agg"]["hr@1"]["std"], "FL"))
    # FedProto
    fp = fix["fedproto"]
    rows.append(("FedProto", fp["cold_pape"]["mean"], fp["cold_pape"]["std"],
                 fp["cold_hr@1"]["mean"], fp["cold_hr@1"]["std"], "pFL"))
    # FedAvg-NBEATSxAux raw (no codebook)
    fa = fix["fedavg_nbeatsx_aux"]["fl_only"]
    rows.append(("FedAvg-NBEATSxAux\n(no codebook)", fa["cold_pape"]["mean"], fa["cold_pape"]["std"],
                 fa["cold_hr@1"]["mean"], fa["cold_hr@1"]["std"], "FL"))
    # Local-only self-eval
    lo_self = fix["local_only_holdout"]["self_eval"]
    rows.append(("Local-only\n(self-eval, overfit UB)", lo_self["cold_pape"]["mean"], lo_self["cold_pape"]["std"],
                 lo_self["cold_hr@1"]["mean"], lo_self["cold_hr@1"]["std"], "no-FL"))
    # Local-only holdout
    lo_h = fix["local_only_holdout"]["holdout_eval"]
    rows.append(("Local-only\n(holdout, fair)", lo_h["cold_pape"]["mean"], lo_h["cold_pape"]["std"],
                 lo_h["cold_hr@1"]["mean"], lo_h["cold_hr@1"]["std"], "no-FL-fair"))
    return rows


def fig_pareto(rows, out_path):
    fig, ax = plt.subplots(figsize=(11, 7.5))
    for label, p_m, p_s, h_m, h_s, group in rows:
        ax.errorbar(p_m, h_m, xerr=p_s, yerr=h_s, fmt="o",
                    color=COLOR[group], markersize=7, capsize=3, alpha=0.85,
                    label=group)
        # text label, offset slightly
        ax.annotate(label, (p_m, h_m), fontsize=8, xytext=(7, 5),
                    textcoords="offset points")
    ax.set_xlabel("Cold PAPE (%) — lower better")
    ax.set_ylabel("Cold HR@1 (%) — higher better")
    ax.set_title("Unified Pareto: cold PAPE x HR@1\n(13 baselines + ours + ours-under-federation; mean ± std, 3 seeds)")
    # Dedupe legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    keep = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    ax.legend([h for h, _ in keep], [l for _, l in keep], loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved {out_path.name}")


def fig_decomposition(rows, out_path):
    """Three-way decomposition of the 20.6 pp gap. Uses §4.8 numbers."""
    # Bars are configurations from Table 6, sorted by PAPE.
    # We pull the three transition costs explicitly.
    configs = [
        ("Proposed\n(centralised)", 35.70, 0.49, "ours"),
        ("Stacked-Aux\n(fed NBEATSxAux\n+ codebook)", 41.93, 1.30, "Stacked-Aux"),
        ("Codebook on FedRep\n(Stacked,\nself-derived aux)", 47.50, 1.36, "Stacked"),
        ("FedAvg\n(raw,\nno aux, no W5)", 56.34, 1.41, "FL"),
    ]
    deltas = [
        (configs[0][1], configs[1][1], "+6.23 pp\nfederation cost"),
        (configs[1][1], configs[2][1], "+5.57 pp\naux-head cost"),
        (configs[2][1], configs[3][1], "+8.84 pp\nW5-hybrid cost"),
    ]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    x = np.arange(len(configs))
    means = [c[1] for c in configs]
    stds = [c[2] for c in configs]
    colors = [COLOR[c[3]] for c in configs]
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, edgecolor="black", linewidth=0.6)
    for i, c in enumerate(configs):
        ax.text(i, c[1] + c[2] + 1.5, f"{c[1]:.2f}", ha="center", fontsize=10, fontweight="bold")
    # Delta annotations as horizontal arrows between bar tops
    for i, (a, b, lbl) in enumerate(deltas):
        y_mid = (a + b) / 2 + 5
        ax.annotate("", xy=(i + 1, b + 0.5), xytext=(i, a + 0.5),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="gray"))
        ax.text(i + 0.5, y_mid + 4, lbl, ha="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", lw=0.6))
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in configs], fontsize=9)
    ax.set_ylabel("Cold PAPE (%) — lower better")
    ax.set_title("Three-way decomposition of the 20.6 pp gap from FedAvg to ours\n"
                 "(20.6 = 6.2 federation + 5.6 aux head + 8.8 W5 hybrid)")
    ax.set_ylim(0, 75)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved {out_path.name}")


def fig_m_sensitivity(fix, out_path):
    ms_block = fix["m_sensitivity"]
    M_vals = [8, 16, 32, 64]
    means, stds = [], []
    for M in M_vals:
        if M == 32:
            means.append(PROPOSED_PAPE_MEAN)
            stds.append(PROPOSED_PAPE_STD)
        else:
            block = ms_block[f"M_{M}"]["PAPE-aggressive"]["pape"]
            means.append(block["mean"])
            stds.append(block["std"])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(M_vals, means, yerr=stds, fmt="o-", color=COLOR["ours"], markersize=8,
                capsize=4, lw=2, label="cold PAPE (PAPE-aggressive)")
    for M, m, s in zip(M_vals, means, stds):
        ax.text(M, m + s + 0.25, f"{m:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.axvspan(31, 33, alpha=0.15, color="gray", label="M = 32 default")
    ax.set_xscale("log", base=2)
    ax.set_xticks(M_vals)
    ax.set_xticklabels([str(M) for M in M_vals])
    ax.set_xlabel("Codebook size M (log2 scale)")
    ax.set_ylabel("Cold PAPE (%) — lower better")
    envelope = max(means) - min(means)
    ax.set_title(f"Cluster-count sensitivity (3-seed mean ± std)\n"
                 f"All four M values within a {envelope:.2f} pp envelope around M = 32")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(min(means) - 1.5, max(means) + 2.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved {out_path.name}")


def fig_sorted_bar(rows, out_path):
    sorted_rows = sorted(rows, key=lambda r: r[1])
    labels = [r[0].replace("\n", " ") for r in sorted_rows]
    means = [r[1] for r in sorted_rows]
    stds = [r[2] for r in sorted_rows]
    colors = [COLOR[r[5]] for r in sorted_rows]
    fig, ax = plt.subplots(figsize=(13, 7.5))
    x = np.arange(len(sorted_rows))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=3, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Cold PAPE (%) — lower better")
    ax.set_title("Cold PAPE across all baselines + ours\n(mean ± std, 3 seeds, sorted by PAPE)")
    # Group legend
    seen = set()
    handles = []
    for r in sorted_rows:
        if r[5] not in seen:
            seen.add(r[5])
            handles.append(plt.Rectangle((0, 0), 1, 1, fc=COLOR[r[5]], label=r[5]))
    ax.legend(handles=handles, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved {out_path.name}")


def main():
    v04, fix = load_jsons()
    rows = build_table_rows(v04, fix)
    print(f"[unified figures] {len(rows)} rows assembled; rendering 4 figures -> {OUT_FIG}")
    fig_pareto(rows, OUT_FIG / "F1_pareto_unified.png")
    fig_decomposition(rows, OUT_FIG / "F6_decomposition.png")
    fig_m_sensitivity(fix, OUT_FIG / "F7_m_sensitivity.png")
    fig_sorted_bar(rows, OUT_FIG / "F8_sorted_unified.png")
    print("[unified figures] done")


if __name__ == "__main__":
    main()
