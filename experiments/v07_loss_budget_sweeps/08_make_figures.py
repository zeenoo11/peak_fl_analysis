"""v07 paper figures — F-aux (λ_aux sweep), F-budget (TBD), F-traj (TBD).

(한글 요약)
plan ``v07-01_loss_and_budget_sweeps.md`` §1 figure: F-aux — algorithm × λ_aux
matrix 시각화. v07-B / v07-C figure (F-budget / F-traj) 는 이후 단계에서 추가.

CLI 모드:

    --section aux      # F-aux 만 출력 (v07-A 마무리 단계)
    --section all      # 모든 출력 가능한 figure (현재 = F-aux 만)

산출:

    outputs/v07_loss_budget_sweeps/figures/F-aux.png  -- final test PAPE vs λ_aux per algorithm
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np

from config import OUTPUT_DIR  # noqa: E402

V07_NAMESPACE = "v07_loss_budget_sweeps"


# Wong-2011 colour palette (matches v06 figures).
_COLORS = {
    "centralised": "#000000",
    "fedavg":      "#0072B2",
    "fedprox":     "#D55E00",
    "fedrep":      "#009E73",
    "ditto":       "#CC79A7",
    "fedproto":    "#E69F00",
}
_PRETTY = {
    "centralised": "Centralised",
    "fedavg":      "FedAvg",
    "fedprox":     "FedProx",
    "fedrep":      "FedRep",
    "ditto":       "Ditto",
    "fedproto":    "FedProto",
}
_ALGO_ORDER = list(_COLORS.keys())


def _load_summary(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# F-aux — λ_aux sweep
# ---------------------------------------------------------------------------


def make_f_aux(summary: dict, fig_dir: Path) -> None:
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(1, 1, figsize=(11.0, 6.5))

    lambdas_str = summary["lambdas"]
    lambdas = [float(s) for s in lambdas_str]

    # Table -> per-algo line plot.
    for algo in _ALGO_ORDER:
        col = _COLORS[algo]
        xs, means, stds = [], [], []
        for lam_str, lam_f in zip(lambdas_str, lambdas):
            cell = summary["test_pape"].get(algo, {}).get(lam_str)
            if cell is None or cell.get("mean") is None:
                continue
            xs.append(lam_f)
            means.append(cell["mean"])
            stds.append(cell["std"])
        if not xs:
            continue
        xs = np.asarray(xs); means = np.asarray(means); stds = np.asarray(stds)
        ax.errorbar(
            xs, means, yerr=stds,
            marker="o", markersize=8, color=col, linewidth=2.0,
            markeredgecolor="black", markeredgewidth=0.6,
            capsize=3.0, elinewidth=0.9, ecolor="#666666",
            label=_PRETTY[algo],
        )

    # Annotate v06 default (λ=0.3).
    ax.axvline(0.3, color="#888888", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.text(0.305, ax.get_ylim()[1],
            "  v06 default (λ=0.3)",
            fontsize=10, color="#444444", va="top")

    ax.set_xlabel(r"$\lambda_{\mathrm{aux}}$  (peak-aux loss weight)")
    ax.set_ylabel(r"Final test PAPE  (mean $\pm$ std over 3 seeds)")
    ax.set_title(
        r"F-aux: $\lambda_{\mathrm{aux}}$ sensitivity at v06 round-level FL protocol"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize=11, title="Algorithm",
              title_fontsize=11)

    fig.tight_layout()
    out_path = fig_dir / "F-aux.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[v07 figures] wrote {out_path}")


# ---------------------------------------------------------------------------
# F-hr — hr_weight sweep at λ_aux=0.1 (v07-A2)
# ---------------------------------------------------------------------------


_PAT_AUX01_HR = re.compile(
    r"^V6-Dyn-(?P<algo>A_centralised|B-(?:FedAvg|FedProx|FedRep|Ditto|FedProto))"
    r"-aux0\.1(?:-hr(?P<hr>[0-9.]+))?$"
)
_PAT_ALGO_TO_KEY = {
    "A_centralised": "centralised", "B-FedAvg": "fedavg", "B-FedProx": "fedprox",
    "B-FedRep": "fedrep", "B-Ditto": "ditto", "B-FedProto": "fedproto",
}


def _hr_table() -> dict:
    """Walk outputs/v07_loss_budget_sweeps/seed{S}/<aux0.1[-hrV]>/result.json.

    Returns {algo -> {hr -> [test_pape, ...]}} aggregated across seeds.
    """
    root = OUTPUT_DIR / V07_NAMESPACE
    out: dict = {a: {} for a in _ALGO_ORDER}
    for seed_dir in root.glob("seed*"):
        for cell_dir in seed_dir.iterdir():
            m = _PAT_AUX01_HR.match(cell_dir.name)
            if not m: continue
            algo = _PAT_ALGO_TO_KEY[m.group("algo")]
            hr = float(m.group("hr")) if m.group("hr") else 0.1
            rp = cell_dir / "result.json"
            if not rp.exists(): continue
            with rp.open() as fh:
                r = json.load(fh)
            out[algo].setdefault(hr, []).append(float(r["test_terminal"]["pape_mean"]))
    return out


def make_f_hr(fig_dir: Path) -> None:
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(1, 1, figsize=(11.0, 6.5))

    hrs = [0.05, 0.1, 0.5, 1.0]
    table = _hr_table()
    for algo in _ALGO_ORDER:
        col = _COLORS[algo]
        xs, means, stds = [], [], []
        for hr in hrs:
            vals = table.get(algo, {}).get(hr, [])
            if not vals:
                continue
            xs.append(hr)
            a = np.asarray(vals, dtype=np.float64)
            means.append(float(a.mean()))
            stds.append(float(a.std(ddof=1)) if a.size > 1 else 0.0)
        if not xs:
            continue
        xs = np.asarray(xs); means = np.asarray(means); stds = np.asarray(stds)
        ax.errorbar(
            xs, means, yerr=stds,
            marker="o", markersize=8, color=col, linewidth=2.0,
            markeredgecolor="black", markeredgewidth=0.6,
            capsize=3.0, elinewidth=0.9, ecolor="#666666",
            label=_PRETTY[algo],
        )

    ax.axvline(0.1, color="#888888", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.text(0.105, ax.get_ylim()[1],
            "  v06 default (hr=0.1)",
            fontsize=10, color="#444444", va="top")
    ax.set_xscale("log")
    ax.set_xticks(hrs)
    ax.set_xticklabels([str(h) for h in hrs])
    ax.set_xlabel(r"hr_weight  (peak-hour CE weight inside peak-aux)")
    ax.set_ylabel(r"Final test PAPE  (mean $\pm$ std over 3 seeds)")
    ax.set_title(
        r"F-hr: hr_weight sweep at $\lambda_{\mathrm{aux}}=0.1$ "
        r"(v07-A centralised optimum)"
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", frameon=True, fontsize=11, title="Algorithm",
              title_fontsize=11)

    fig.tight_layout()
    out_path = fig_dir / "F-hr.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[v07 figures] wrote {out_path}")


# ---------------------------------------------------------------------------
# F-codebook-vs-lambda — centralised λ ∈ {0, 0.1, 0.3} × {before, after}
# ---------------------------------------------------------------------------


def _centralised_codebook_table() -> dict:
    """Read codebook_lift.json across the three centralised backbone variants.

    Returns {lambda_label -> {before|after -> [pape_per_seed]}}.
    """
    cases = [
        ("λ=0",   "v06_round_dynamics",     "V6-Dyn-A_centralised-MAEonly"),
        ("λ=0.1", "v07_loss_budget_sweeps", "V6-Dyn-A_centralised-aux0.1"),
        ("λ=0.3", "v06_round_dynamics",     "V6-Dyn-A_centralised"),
    ]
    out: dict = {}
    for label, ns, cell in cases:
        out[label] = {"before": [], "after": []}
        for seed in (42, 123, 7):
            p = OUTPUT_DIR / ns / f"seed{seed}" / cell / "codebook_lift.json"
            if not p.exists():
                continue
            with p.open() as fh:
                r = json.load(fh)
            out[label]["before"].append(float(r["test_before"]["pape_mean"]))
            out[label]["after"].append(float(r["test_after"]["pape_mean"]))
    return out


def make_f_codebook_vs_lambda(fig_dir: Path) -> None:
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 6.0))

    table = _centralised_codebook_table()
    labels = list(table.keys())
    n = len(labels)
    xs = np.arange(n)
    width = 0.35

    def _std_ddof1(vals: list[float]) -> float:
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    bef_means = [float(np.mean(table[l]["before"])) if table[l]["before"] else 0.0 for l in labels]
    bef_stds  = [_std_ddof1(table[l]["before"]) for l in labels]
    aft_means = [float(np.mean(table[l]["after"]))  if table[l]["after"]  else 0.0 for l in labels]
    aft_stds  = [_std_ddof1(table[l]["after"])  for l in labels]

    bars1 = ax.bar(xs - width/2, bef_means, width, yerr=bef_stds,
                   color="#888888", edgecolor="black", linewidth=0.8,
                   capsize=3.5, ecolor="#444444", label="Backbone only (before codebook)")
    bars2 = ax.bar(xs + width/2, aft_means, width, yerr=aft_stds,
                   color="#0072B2", edgecolor="black", linewidth=0.8,
                   capsize=3.5, ecolor="#444444", label="Backbone + federated codebook")

    for x, bm, am in zip(xs, bef_means, aft_means):
        ax.text(x - width/2, bm + 0.15, f"{bm:.2f}", ha="center", va="bottom", fontsize=11)
        ax.text(x + width/2, am + 0.15, f"{am:.2f}", ha="center", va="bottom", fontsize=11)
        delta = am - bm
        ax.annotate(
            f"{delta:+.2f}",
            xy=(x, am - 0.5), xytext=(x, am - 1.8),
            ha="center", fontsize=11, color="#005577",
            arrowprops=dict(arrowstyle="->", color="#005577", lw=0.9),
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"Backbone $\lambda_{\mathrm{aux}}$  (centralised cell)")
    ax.set_ylabel(r"Test PAPE  (mean $\pm$ std over 3 seeds)")
    ax.set_title(
        r"F-codebook$\times$λ: codebook absorbs the backbone $\lambda_{\mathrm{aux}}$ choice"
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", frameon=True, fontsize=11)
    y_lo = float(min(aft_means) - max(aft_stds) - 2.0)
    y_hi = float(max(bef_means) + max(bef_stds) + 2.0)
    ax.set_ylim(y_lo, y_hi)

    fig.tight_layout()
    out_path = fig_dir / "F-codebook-vs-lambda.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[v07 figures] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="v07 paper figures.")
    ap.add_argument("--section", type=str, default="aux",
                    choices=["aux", "hr", "codebook_lambda", "budget", "traj", "all"],
                    help="Which v07 section's figure to render.")
    ap.add_argument("--summary", type=Path, default=None,
                    help="Override aux summary path "
                         "(default: outputs/v07_loss_budget_sweeps/aux_sweep_summary.json).")
    ap.add_argument("--fig_dir", type=Path, default=None,
                    help="Override output dir (default: outputs/v07_loss_budget_sweeps/figures/).")
    args = ap.parse_args()

    fig_dir = args.fig_dir or (OUTPUT_DIR / V07_NAMESPACE / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.section in ("aux", "all"):
        summary_path = args.summary or (OUTPUT_DIR / V07_NAMESPACE / "aux_sweep_summary.json")
        if not summary_path.exists():
            print(f"[v07 figures] WARN — {summary_path} missing. Run 05_aggregate_aux.py first.")
            sys.exit(1)
        summary = _load_summary(summary_path)
        make_f_aux(summary, fig_dir)

    if args.section in ("hr", "all"):
        make_f_hr(fig_dir)

    if args.section in ("codebook_lambda", "all"):
        make_f_codebook_vs_lambda(fig_dir)

    if args.section in ("budget", "all"):
        print("[v07 figures] F-budget not yet implemented (waiting on v07-B runs).")

    if args.section in ("traj", "all"):
        print("[v07 figures] F-traj not yet implemented (waiting on v07-C runs).")


if __name__ == "__main__":
    main()
