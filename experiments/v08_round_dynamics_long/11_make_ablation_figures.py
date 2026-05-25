"""V6 Phase 2 ablation figures — F7 (alpha_v0 Pareto) + F8 (K_local sweep).

Reads the per-seed × per-cell × ablation results that
``08_codebook_stacking.py --ablation_suffix ...`` wrote during the v06
follow-up sweep (2026-05-03):

    seed{S}/{cell}/codebook_lift.json              -- v06 baseline (alpha=1.0, K=2, lambda=0.3)
    seed{S}/{cell}/codebook_lift_alpha{V}.json     -- alpha sweep    (V in {0.5, 1.5, 2.0})
    seed{S}/{cell}/codebook_lift_K{K}.json         -- K_local sweep  (K in {1, 4, 8}; FL cells only)

and produces two paper-grade figures:

    figures/F7_alpha_pareto.png    -- ΔMAE vs ΔPAPE Pareto curve, 6 cells × 4 alphas
    figures/F8_klocal_sweep.png    -- ΔPAPE vs K_local, 5 FL cells, diminishing-returns curve

Both figures use the v06 Wong colour palette so they sit visually next to
F1-F6.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np

from config import OUTPUT_DIR


_COLORS = {
    "V6-Dyn-A_centralised": "#000000",
    "V6-Dyn-B-FedAvg":      "#0072B2",
    "V6-Dyn-B-FedProx":     "#D55E00",
    "V6-Dyn-B-FedRep":      "#009E73",
    "V6-Dyn-B-Ditto":       "#CC79A7",
    "V6-Dyn-B-FedProto":    "#E69F00",
}

CELLS_DEFAULT = list(_COLORS.keys())
FL_CELLS = CELLS_DEFAULT[1:]
SEEDS = [42, 123, 7]


def _short(cell: str) -> str:
    return cell.replace("V6-Dyn-A_centralised", "Centralised").replace("V6-Dyn-B-", "")


def _load(seed: int, cell: str, suffix: str = "") -> dict | None:
    p = OUTPUT_DIR / "v08_round_dynamics_long" / f"seed{seed}" / cell / f"codebook_lift{suffix}.json"
    if not p.exists():
        return None
    with p.open() as fh:
        return json.load(fh)


def _agg(values: list[float]) -> tuple[float, float]:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def _alpha_grid(cell: str) -> dict[float, dict[str, tuple[float, float]]]:
    """Return {alpha -> {key -> (mean, std)}} for a cell across SEEDS."""
    out: dict[float, dict[str, tuple[float, float]]] = {}
    for a in (0.5, 1.0, 1.5, 2.0):
        suf = "" if a == 1.0 else f"_alpha{a}"
        papes_after, dpapes, dmaes = [], [], []
        for s in SEEDS:
            d = _load(s, cell, suf)
            if d is None:
                continue
            papes_after.append(d["test_after"]["pape_mean"])
            dpapes.append(d["lift"]["pape_delta"])
            dmaes.append(d["lift"]["mae_delta"])
        out[a] = {
            "pape_after": _agg(papes_after),
            "dpape": _agg(dpapes),
            "dmae":  _agg(dmaes),
        }
    return out


def _klocal_grid(cell: str) -> dict[int, dict[str, tuple[float, float]]]:
    out: dict[int, dict[str, tuple[float, float]]] = {}
    for k in (1, 2, 4, 8):
        suf = "" if k == 2 else f"_K{k}"
        dpapes, papes_after = [], []
        for s in SEEDS:
            d = _load(s, cell, suf)
            if d is None:
                continue
            dpapes.append(d["lift"]["pape_delta"])
            papes_after.append(d["test_after"]["pape_mean"])
        out[k] = {
            "dpape":      _agg(dpapes),
            "pape_after": _agg(papes_after),
        }
    return out


# ---------------------------------------------------------------------------
# F7 — alpha_v0 Pareto curve (ΔMAE on x, ΔPAPE on y)
# ---------------------------------------------------------------------------


def make_f7(fig_dir: Path) -> None:
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(1, 1, figsize=(13.0, 7.5))

    alphas = [0.5, 1.0, 1.5, 2.0]
    marker_for_alpha = {0.5: "o", 1.0: "s", 1.5: "^", 2.0: "D"}

    for cell in CELLS_DEFAULT:
        col = _COLORS[cell]
        grid = _alpha_grid(cell)
        xs = [grid[a]["dmae"][0] for a in alphas]
        ys = [grid[a]["dpape"][0] for a in alphas]
        ax.plot(xs, ys, color=col, linewidth=2.0, alpha=0.85, zorder=2)
        for a in alphas:
            mx, sx = grid[a]["dmae"]
            my, sy = grid[a]["dpape"]
            ax.errorbar(
                mx, my, xerr=sx, yerr=sy,
                fmt=marker_for_alpha[a], color=col,
                markersize=10, markeredgecolor="black", markeredgewidth=0.7,
                elinewidth=0.8, capsize=2.5, ecolor="#666666",
                zorder=3,
            )

    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)

    # Sweet-spot annotations on the data area.
    ax.annotate(
        "MAE-zero-cost\n" + r"($\alpha_{v0}=0.5$)",
        xy=(-0.0004, -2.7), xytext=(0.005, -1.0),
        fontsize=11, color="#222222", ha="left",
        arrowprops=dict(arrowstyle="->", color="#444444", lw=0.9),
    )
    ax.annotate(
        "PAPE-aggressive\n" + r"($\alpha_{v0}=2.0$)",
        xy=(0.04, -8.65), xytext=(0.025, -10.5),
        fontsize=11, color="#222222", ha="left",
        arrowprops=dict(arrowstyle="->", color="#444444", lw=0.9),
    )

    ax.set_xlabel(r"$\Delta$MAE  (kW; positive = MAE worse with codebook)")
    ax.set_ylabel(r"$\Delta$PAPE  (%; negative = PAPE better with codebook)")
    ax.set_title(
        r"F7: Codebook correction strength $\alpha_{v0}$ — PAPE vs MAE Pareto"
    )
    ax.grid(True, alpha=0.3)

    # Two legends, both on the right margin outside the plotting area.
    alpha_handles = [
        plt.Line2D([0], [0], marker=marker_for_alpha[a], linestyle="",
                   color="#444444", markersize=10, markeredgecolor="black",
                   markeredgewidth=0.7,
                   label=fr"$\alpha_{{v0}} = {a}$")
        for a in alphas
    ]
    cell_handles = [
        plt.Line2D([0], [0], color=_COLORS[c], linewidth=2.5, label=_short(c))
        for c in CELLS_DEFAULT
    ]
    leg_alpha = ax.legend(
        handles=alpha_handles, loc="upper left",
        bbox_to_anchor=(1.02, 1.0), frameon=True,
        title="Operating point", fontsize=11, title_fontsize=12,
    )
    ax.add_artist(leg_alpha)
    ax.legend(
        handles=cell_handles, loc="upper left",
        bbox_to_anchor=(1.02, 0.55), frameon=True,
        fontsize=11, title="Cell", title_fontsize=12,
    )

    fig.tight_layout()
    fig.subplots_adjust(right=0.80)
    out_path = fig_dir / "F7_alpha_pareto.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[v06 figures] wrote {out_path}")


# ---------------------------------------------------------------------------
# F8 — K_local sweep on FL cells (ΔPAPE vs K_local)
# ---------------------------------------------------------------------------


def make_f8(fig_dir: Path) -> None:
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.5))

    ks = [1, 2, 4, 8]
    for cell in FL_CELLS:
        col = _COLORS[cell]
        grid = _klocal_grid(cell)
        ys = [grid[k]["dpape"][0] for k in ks]
        es = [grid[k]["dpape"][1] for k in ks]
        ax.errorbar(
            ks, ys, yerr=es,
            marker="o", markersize=8, color=col, linewidth=2.0,
            markeredgecolor="black", markeredgewidth=0.6,
            capsize=3.0, elinewidth=0.9, ecolor="#666666",
            label=_short(cell),
        )

    # Vertical line at K=2 (v05/v06 baseline).
    ax.axvline(2, color="#888888", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.text(2.06, ax.get_ylim()[1] - 0.15,
            "  K=2  (v05 / v06 baseline)",
            fontsize=10, color="#444444", va="top")

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel(r"$K_{\mathrm{local}}$ (Stage-1 cluster count per client)")
    ax.set_ylabel(r"$\Delta$PAPE on TEST  (% — more negative = larger lift)")
    ax.set_title(
        r"F8: Federated codebook K$_{local}$ sweep — diminishing returns past K=2"
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", frameon=True, fontsize=11, title="FL cell",
              title_fontsize=11, ncol=1)

    fig.tight_layout()
    out_path = fig_dir / "F8_klocal_sweep.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[v06 figures] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="v06 Phase 2 ablation figures (F7 + F8).")
    ap.add_argument("--fig_dir", type=Path, default=None,
                    help="Override output directory (default: outputs/v08_round_dynamics_long/figures/).")
    args = ap.parse_args()

    fig_dir = args.fig_dir or (OUTPUT_DIR / "v08_round_dynamics_long" / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    make_f7(fig_dir)
    make_f8(fig_dir)


if __name__ == "__main__":
    main()
