"""V6 Phase 2 figure — F6: codebook lift bar chart (per-cell BEFORE vs AFTER).

(한글 요약)
``09_aggregate_codebook.py`` 가 생성한 ``codebook_lift_summary.json`` 을 읽어
12 cell 각각의 test PAPE 를 BEFORE (codebook 미적용) 와 AFTER (CMO 적용)
grouped bar 로 그린다. λ_aux=0.3 default 와 λ_aux=0 (MAEonly) 를 두 개의
subplot panel 로 분리해 paper readability 를 유지 (F4/F5 와 동일 컨벤션).

Output: ``outputs/v08_round_dynamics_long/figures/F6_codebook_lift.png``.

Style: matplotlib, color-blind friendly palette (Wong 2011 Nature Methods),
large fonts (paper-style). Mean±std 가 errorbar 로 표시되며 lift Δ 는 막대
위에 +/- 라벨로 적힌다.
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


# Color-blind friendly palette (Wong 2011, Nature Methods) — same as F1/F2/F3.
_COLORS = {
    "V6-Dyn-A_centralised": "#000000",
    "V6-Dyn-B-FedAvg":      "#0072B2",
    "V6-Dyn-B-FedProx":     "#D55E00",
    "V6-Dyn-B-FedRep":      "#009E73",
    "V6-Dyn-B-Ditto":       "#CC79A7",
    "V6-Dyn-B-FedProto":    "#E69F00",
}

_DEFAULT_PANEL = [
    "V6-Dyn-A_centralised",
    "V6-Dyn-B-FedAvg",
    "V6-Dyn-B-FedProx",
    "V6-Dyn-B-FedRep",
    "V6-Dyn-B-Ditto",
    "V6-Dyn-B-FedProto",
]
_MAEONLY_PANEL = [c + "-MAEonly" for c in _DEFAULT_PANEL]


def _short_label(cell: str) -> str:
    s = cell
    s = s.replace("V6-Dyn-A_centralised", "Centralised")
    s = s.replace("V6-Dyn-B-", "")
    s = s.replace("-MAEonly", "")
    return s


def _color_for(cell: str) -> str:
    base = cell.replace("-MAEonly", "")
    return _COLORS.get(base, "#888888")


def _draw_panel(
    ax,
    summary: dict,
    cells: list[str],
    title: str,
) -> bool:
    """Draw one panel (default or MAEonly) — grouped bars per cell.

    Returns True if at least one cell was drawn, False otherwise.
    """
    cells_present = [c for c in cells if c in summary.get("cells", {})]
    if not cells_present:
        return False

    n = len(cells_present)
    width = 0.36
    x = np.arange(n)

    means_before = np.array([
        summary["cells"][c]["test_before"]["pape_mean"]["mean"] for c in cells_present
    ], dtype=np.float64)
    stds_before = np.array([
        summary["cells"][c]["test_before"]["pape_mean"]["std"] for c in cells_present
    ], dtype=np.float64)
    means_after = np.array([
        summary["cells"][c]["test_after"]["pape_mean"]["mean"] for c in cells_present
    ], dtype=np.float64)
    stds_after = np.array([
        summary["cells"][c]["test_after"]["pape_mean"]["std"] for c in cells_present
    ], dtype=np.float64)
    lift_means = np.array([
        summary["cells"][c]["lift"]["pape_delta"]["mean"] for c in cells_present
    ], dtype=np.float64)

    # Zoom y-axis to the data range so the ~5-PAPE lift is visually salient.
    # ylim = [floor(min_after) - 2, ceil(max_before) + 4]   (4 of headroom for
    # the Δ label + legend without overlap).
    y_lo = float(np.floor(np.min(means_after - stds_after) - 2.0))
    y_hi = float(np.ceil(np.max(means_before + stds_before) + 4.0))
    ax.set_ylim(y_lo, y_hi)
    label_offset = (y_hi - y_lo) * 0.04   # ~4% of y-range above each bar pair

    # BEFORE bars (faded), AFTER bars (saturated, color per algorithm).
    for i, c in enumerate(cells_present):
        col = _color_for(c)
        ax.bar(
            x[i] - width / 2, means_before[i], width=width,
            yerr=stds_before[i], color=col, alpha=0.35,
            edgecolor="black", linewidth=0.6,
            error_kw={"elinewidth": 1.0, "capsize": 3.0, "ecolor": "#555555"},
            label="BEFORE (no codebook)" if i == 0 else None,
        )
        ax.bar(
            x[i] + width / 2, means_after[i], width=width,
            yerr=stds_after[i], color=col, alpha=0.95,
            edgecolor="black", linewidth=0.6,
            error_kw={"elinewidth": 1.0, "capsize": 3.0, "ecolor": "#222222"},
            label="AFTER (+ codebook CMO)" if i == 0 else None,
        )
        # Lift annotation centered between the two bars, scaled to y-range.
        top = max(means_before[i] + stds_before[i], means_after[i] + stds_after[i])
        sign = "+" if lift_means[i] >= 0 else ""
        ax.text(
            x[i], top + label_offset,
            f"Δ {sign}{lift_means[i]:.2f}",
            ha="center", va="bottom", fontsize=11, color="#222222",
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_short_label(c) for c in cells_present],
        rotation=15, ha="right", fontsize=11,
    )
    ax.set_ylabel("Across-client TEST PAPE (%)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    # Legend OUTSIDE the plotting area so it never overlaps the first-cell Δ label.
    ax.legend(
        loc="upper right", bbox_to_anchor=(1.0, -0.18),
        frameon=False, fontsize=11, ncol=2,
    )
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="v06 Phase 2 figure renderer (F6).")
    ap.add_argument("--summary_path", type=Path, default=None,
                    help="Override path to codebook_lift_summary.json (default: "
                         "outputs/v08_round_dynamics_long/codebook_lift_summary.json).")
    ap.add_argument("--fig_dir", type=Path, default=None,
                    help="Override output directory (default: "
                         "outputs/v08_round_dynamics_long/figures/).")
    args = ap.parse_args()

    summary_path = args.summary_path or (
        OUTPUT_DIR / "v08_round_dynamics_long" / "codebook_lift_summary.json"
    )
    fig_dir = args.fig_dir or (OUTPUT_DIR / "v08_round_dynamics_long" / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        raise FileNotFoundError(
            f"v06 Phase 2 figure requires {summary_path}. Run "
            f"experiments/v08_round_dynamics_long/09_aggregate_codebook.py first."
        )
    with summary_path.open() as fh:
        summary = json.load(fh)

    plt.rcParams.update({"font.size": 13})

    # Decide layout based on which panels have data so an empty MAEonly subplot
    # does not steal half the figure (the v06 Phase 2 plan only runs λ=0.3
    # cells, so the MAEonly panel is typically absent).
    default_present = any(c in summary.get("cells", {}) for c in _DEFAULT_PANEL)
    maeonly_present = any(c in summary.get("cells", {}) for c in _MAEONLY_PANEL)

    if not default_present and not maeonly_present:
        print("[v06 figures] F6 skipped — no codebook_lift entries in summary.")
        return

    if default_present and maeonly_present:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11.0, 9.0), sharey=False)
        _draw_panel(
            ax_top, summary, _DEFAULT_PANEL,
            title=r"F6 (a): Codebook lift on TEST PAPE — default ($\lambda_{aux} = 0.3$)",
        )
        _draw_panel(
            ax_bot, summary, _MAEONLY_PANEL,
            title=r"F6 (b): Codebook lift on TEST PAPE — MAE-only ablation ($\lambda_{aux} = 0$)",
        )
    elif default_present:
        # Single-panel layout: only λ=0.3 cells available.
        fig, ax = plt.subplots(1, 1, figsize=(11.0, 5.5))
        _draw_panel(
            ax, summary, _DEFAULT_PANEL,
            title=r"F6: Codebook lift on TEST PAPE — default ($\lambda_{aux} = 0.3$)",
        )
    else:
        # Single-panel: only MAEonly (rare).
        fig, ax = plt.subplots(1, 1, figsize=(11.0, 5.5))
        _draw_panel(
            ax, summary, _MAEONLY_PANEL,
            title=r"F6: Codebook lift on TEST PAPE — MAE-only ablation ($\lambda_{aux} = 0$)",
        )

    fig.tight_layout()
    # Reserve bottom space for the legend bbox below the axes.
    fig.subplots_adjust(bottom=0.22)
    out_path = fig_dir / "F6_codebook_lift.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[v06 figures] wrote {out_path}")


if __name__ == "__main__":
    main()
