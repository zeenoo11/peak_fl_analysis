"""V6 Phase 1 figures — round-vs-{val,test,train_loss}, bytes-vs-PAPE,
drift-vs-round, plus optional MAE-only ablation panels.

(한글 요약)
plan v06-01 §"Build order" step 9 + 변경 1/2 (round-level test trajectory +
training-loss trajectory + λ_aux=0 ablation namespace). ``trajectories.npz``
로부터 다음 figure 들을 생성:

Default (λ_aux=0.3) — main paper figures
    F1_round_vs_val_pape.png    — 5종 FL trajectory (mean ± std band) + V6-Dyn-A 점선.
    F1b_round_vs_test_pape.png  — 같은 5종 trajectory 의 round-level test PAPE
                                  (McMahan2017 Figure 2 / FedProx Figure 6 convention).
    F1c_round_vs_train_loss.png — round-averaged training main loss
                                  (FedProx Figure 2 convention; Y = MAE on z-norm).
    F2_bytes_vs_val_pape.png    — 같은 trajectory 를 cumulative upload bytes 축으로.
    F3_drift_vs_round.png       — 5종 FL drift_l2 curve (V6-Dyn-A drift=0 점선).

Ablation (λ_aux=0, suffix ``-MAEonly``) — separate figures, never mixed
into the default panels (paper readability):
    F4_round_vs_test_pape_MAEonly.png
    F5_round_vs_train_loss_MAEonly.png

Style: matplotlib, color-blind friendly palette, large fonts (paper-style).
Read-only — does not modify the npz / json sources. Legacy npz files without
the new keys (test/train_loss) trigger a graceful skip + warning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np

from config import OUTPUT_DIR


# Color-blind-friendly palette (Wong 2011, Nature Methods).
_COLORS = {
    "V6-Dyn-A_centralised": "#000000",  # black for the upper-bound reference
    "V6-Dyn-B-FedAvg":      "#0072B2",  # blue
    "V6-Dyn-B-FedProx":     "#D55E00",  # vermillion
    "V6-Dyn-B-FedRep":      "#009E73",  # bluish green
    "V6-Dyn-B-Ditto":       "#CC79A7",  # reddish purple
    "V6-Dyn-B-FedProto":    "#E69F00",  # orange
}

_FL_CELLS = [
    "V6-Dyn-B-FedAvg",
    "V6-Dyn-B-FedProx",
    "V6-Dyn-B-FedRep",
    "V6-Dyn-B-Ditto",
    "V6-Dyn-B-FedProto",
]

_CENTRALISED = "V6-Dyn-A_centralised"


def _color_for(cell: str) -> str:
    """Color lookup that strips the ``-MAEonly`` / ``-aux*`` suffix so ablation
    cells inherit their parent algorithm's color in the dedicated F4/F5 panels."""
    if cell in _COLORS:
        return _COLORS[cell]
    for suffix in ("-MAEonly", "-aux"):
        if suffix in cell:
            base = cell.split(suffix)[0]
            # If we split on "-aux" we may have stripped a numeric tag too —
            # so strip back to a known cell.
            if base in _COLORS:
                return _COLORS[base]
    return "#888888"


def _has(traj: dict, *keys: str) -> bool:
    return all(k in traj for k in keys)


def _all_nan(arr: np.ndarray) -> bool:
    return bool(np.all(np.isnan(arr)))


def _mean_std(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean + std (ddof=1 if >=2 seeds)."""
    arr = np.asarray(arr, dtype=np.float64)
    m = np.nanmean(arr, axis=0)
    if arr.shape[0] >= 2:
        s = np.nanstd(arr, axis=0, ddof=1)
    else:
        s = np.zeros_like(m)
    return m, s


def _label(cell: str) -> str:
    return cell.replace("V6-Dyn-B-", "").replace("V6-Dyn-A_centralised", "Centralised (V6-Dyn-A)")


# Display-only helpers for the centralised cell (raw npz untouched).
#   _truncate_x: clip the centralised epoch axis to match the FL round axis.
#   _smooth_centralised: rolling-mean smoothing of mean / std curves so the
#   centralised reference line reads cleanly against the FL bands.
def _truncate_x(x: np.ndarray, m: np.ndarray, s: np.ndarray, max_x: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_x is None or max_x <= 0:
        return x, m, s
    mask = x <= max_x
    return x[mask], m[mask], s[mask]


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1 or arr.size == 0:
        return arr
    pad = window // 2
    padded = np.concatenate([
        np.full(pad, arr[0]),
        arr,
        np.full(window - 1 - pad, arr[-1]),
    ])
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def _smooth_centralised(m: np.ndarray, s: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    return _rolling_mean(m, window), _rolling_mean(s, window)


# Module-level display config; overridden by CLI flags in main().
_CENTRALISED_MAX_X = 20
_CENTRALISED_SMOOTH_WINDOW = 3


def _figure_round_vs_pape(traj: dict, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    plt.rcParams.update({"font.size": 13})

    for cell in _FL_CELLS:
        key_x = f"{cell}_round_idx"
        key_y = f"{cell}_val_pape_mean"
        if key_x not in traj or key_y not in traj:
            continue
        x = traj[key_x][0]  # round indices same across seeds
        m, _ = _mean_std(traj[key_y])
        ax.plot(x, m, label=_label(cell), color=_COLORS[cell], linewidth=2.0)

    cent_x_key = "V6-Dyn-A_centralised_round_idx"
    cent_y_key = "V6-Dyn-A_centralised_val_pape_mean"
    if cent_x_key in traj and cent_y_key in traj:
        x = traj[cent_x_key][0]
        m, s = _mean_std(traj[cent_y_key])
        x, m, s = _truncate_x(x, m, s, _CENTRALISED_MAX_X)
        m, _ = _smooth_centralised(m, s, _CENTRALISED_SMOOTH_WINDOW)
        ax.plot(x, m, label=_label("V6-Dyn-A_centralised"),
                color=_COLORS["V6-Dyn-A_centralised"], linewidth=1.8, linestyle="--")

    ax.set_xlim(1, 20)
    ax.set_xlabel("Round (FL) / Epoch (Centralised)")
    ax.set_ylabel("Across-client validation PAPE (%)")
    ax.set_title("F1: Round-by-round validation PAPE")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "F1_round_vs_val_pape.png", dpi=160)
    plt.close(fig)


def _figure_bytes_vs_pape(traj: dict, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    plt.rcParams.update({"font.size": 13})

    for cell in _FL_CELLS:
        key_x = f"{cell}_upload_bytes_cum"
        key_y = f"{cell}_val_pape_mean"
        if key_x not in traj or key_y not in traj:
            continue
        # bytes_cum is per-seed; mean across seeds (typically all-equal across seeds).
        x = np.nanmean(traj[key_x], axis=0) / (1024 ** 2)  # MB
        m, s = _mean_std(traj[key_y])
        ax.plot(x, m, label=_label(cell), color=_COLORS[cell], linewidth=2.0)
        ax.fill_between(x, m - s, m + s, color=_COLORS[cell], alpha=0.20)

    # V6-Dyn-A has 0 bytes; plot as a single horizontal line at the
    # final epoch's PAPE for reference.
    cent_y_key = "V6-Dyn-A_centralised_val_pape_mean"
    if cent_y_key in traj:
        m, s = _mean_std(traj[cent_y_key])
        # Use the final-epoch reference PAPE as the centralised baseline.
        ax.axhline(y=float(m[-1]), label=_label("V6-Dyn-A_centralised"),
                   color=_COLORS["V6-Dyn-A_centralised"], linewidth=1.5, linestyle="--")

    ax.set_xlabel("Cumulative upload bytes (MB)")
    ax.set_ylabel("Across-client validation PAPE (%)")
    ax.set_title("F2: Comm budget vs validation PAPE")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "F2_bytes_vs_val_pape.png", dpi=160)
    plt.close(fig)


def _figure_drift_vs_round(traj: dict, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    plt.rcParams.update({"font.size": 13})

    for cell in _FL_CELLS:
        key_x = f"{cell}_round_idx"
        key_y = f"{cell}_drift_l2"
        if key_x not in traj or key_y not in traj:
            continue
        x = traj[key_x][0]
        m, s = _mean_std(traj[key_y])
        ax.plot(x, m, label=_label(cell), color=_COLORS[cell], linewidth=2.0)
        ax.fill_between(x, m - s, m + s, color=_COLORS[cell], alpha=0.20)

    # V6-Dyn-A drift = 0 by definition; horizontal reference line at 0.
    ax.axhline(y=0.0, label=_label("V6-Dyn-A_centralised"),
               color=_COLORS["V6-Dyn-A_centralised"], linewidth=1.5, linestyle="--")

    ax.set_xlabel("Round")
    ax.set_ylabel(r"Mean client drift  $\| \theta_i - \theta_{global} \|_2$")
    ax.set_title("F3: Per-round client drift")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "F3_drift_vs_round.png", dpi=160)
    plt.close(fig)


def _figure_round_vs_test_pape(traj: dict, fig_dir: Path) -> None:
    """F1b — round-by-round across-client TEST PAPE (paper convention)."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    plt.rcParams.update({"font.size": 13})

    plotted_any = False
    for cell in _FL_CELLS:
        key_x = f"{cell}_round_idx"
        key_y = f"{cell}_test_pape_mean"
        if not _has(traj, key_x, key_y):
            continue
        if _all_nan(traj[key_y]):
            continue  # legacy run — no round-level test logged
        x = traj[key_x][0]
        m, s = _mean_std(traj[key_y])
        ax.plot(x, m, label=_label(cell), color=_color_for(cell), linewidth=2.0)
        ax.fill_between(x, m - s, m + s, color=_color_for(cell), alpha=0.20)
        plotted_any = True

    cent_x_key = f"{_CENTRALISED}_round_idx"
    cent_y_key = f"{_CENTRALISED}_test_pape_mean"
    if _has(traj, cent_x_key, cent_y_key) and not _all_nan(traj[cent_y_key]):
        x = traj[cent_x_key][0]
        m, s = _mean_std(traj[cent_y_key])
        x, m, s = _truncate_x(x, m, s, _CENTRALISED_MAX_X)
        m, s = _smooth_centralised(m, s, _CENTRALISED_SMOOTH_WINDOW)
        ax.plot(x, m, label=_label(_CENTRALISED),
                color=_color_for(_CENTRALISED), linewidth=1.8, linestyle="--")
        ax.fill_between(x, m - s, m + s,
                        color=_color_for(_CENTRALISED), alpha=0.10)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print("[v06 figures] F1b skipped - no round-level test trajectory in npz "
              "(legacy run; re-train to populate the `test` block per-round).")
        return

    ax.set_xlabel("Round (FL) / Epoch (Centralised)")
    ax.set_ylabel("Across-client TEST PAPE (%)")
    ax.set_title("F1b: Round-by-round test PAPE (McMahan/FedProx convention)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "F1b_round_vs_test_pape.png", dpi=160)
    plt.close(fig)


def _figure_round_vs_train_loss(traj: dict, fig_dir: Path) -> None:
    """F1c — round-averaged training main loss (MAE on z-norm; FedProx
    Figure 2 convention).

    For Centralised (V6-Dyn-A) the curve is the per-epoch pooled training
    main loss; for the FL cells it is the average-across-clients main loss
    of the last local epoch (= ``train.loss_mean_last_epoch`` from the
    jsonl).
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    plt.rcParams.update({"font.size": 13})

    plotted_any = False
    for cell in _FL_CELLS:
        key_x = f"{cell}_round_idx"
        key_y = f"{cell}_train_loss_main"
        if not _has(traj, key_x, key_y):
            continue
        if _all_nan(traj[key_y]):
            continue
        x = traj[key_x][0]
        m, s = _mean_std(traj[key_y])
        ax.plot(x, m, label=_label(cell), color=_color_for(cell), linewidth=2.0)
        ax.fill_between(x, m - s, m + s, color=_color_for(cell), alpha=0.20)
        plotted_any = True

    cent_x_key = f"{_CENTRALISED}_round_idx"
    cent_y_key = f"{_CENTRALISED}_train_loss_main"
    if _has(traj, cent_x_key, cent_y_key) and not _all_nan(traj[cent_y_key]):
        x = traj[cent_x_key][0]
        m, s = _mean_std(traj[cent_y_key])
        x, m, s = _truncate_x(x, m, s, _CENTRALISED_MAX_X)
        m, s = _smooth_centralised(m, s, _CENTRALISED_SMOOTH_WINDOW)
        ax.plot(x, m, label=_label(_CENTRALISED),
                color=_color_for(_CENTRALISED), linewidth=1.8, linestyle="--")
        ax.fill_between(x, m - s, m + s,
                        color=_color_for(_CENTRALISED), alpha=0.10)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print("[v06 figures] F1c skipped - no train_loss_main in npz "
              "(legacy aggregate without train-loss column).")
        return

    ax.set_xlabel("Round (FL) / Epoch (Centralised)")
    ax.set_ylabel("Round-averaged training main loss (MAE on z-norm)")
    ax.set_title("F1c: Training-loss trajectory (FedProx Figure 2 convention)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "F1c_round_vs_train_loss.png", dpi=160)
    plt.close(fig)


def _maeonly_cells(traj: dict) -> list[str]:
    """Discover ``*-MAEonly`` cells (λ_aux=0 ablation namespace) present
    in the npz. Returns them in the same algorithmic order as ``_FL_CELLS``,
    with V6-Dyn-A first if present.
    """
    keys = {k.split("_round_idx")[0] for k in traj.keys() if k.endswith("_round_idx")}
    out: list[str] = []
    for cent in (f"{_CENTRALISED}-MAEonly",):
        if cent in keys:
            out.append(cent)
    for fl in _FL_CELLS:
        nm = f"{fl}-MAEonly"
        if nm in keys:
            out.append(nm)
    return out


def _figure_round_vs_test_pape_maeonly(traj: dict, fig_dir: Path) -> None:
    """F4 — same as F1b but only for ``-MAEonly`` cells (λ_aux=0 ablation)."""
    cells = _maeonly_cells(traj)
    if not cells:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    plt.rcParams.update({"font.size": 13})
    plotted_any = False
    for cell in cells:
        key_x = f"{cell}_round_idx"
        key_y = f"{cell}_test_pape_mean"
        if not _has(traj, key_x, key_y) or _all_nan(traj[key_y]):
            continue
        x = traj[key_x][0]
        m, s = _mean_std(traj[key_y])
        is_cent = cell.startswith(_CENTRALISED)
        if is_cent:
            x, m, s = _truncate_x(x, m, s, _CENTRALISED_MAX_X)
            m, s = _smooth_centralised(m, s, _CENTRALISED_SMOOTH_WINDOW)
        ax.plot(x, m,
                label=_label(cell.replace("-MAEonly", "")) + r" ($\lambda_{aux}=0$)",
                color=_color_for(cell),
                linewidth=1.8 if is_cent else 2.0,
                linestyle="--" if is_cent else "-")
        ax.fill_between(x, m - s, m + s, color=_color_for(cell),
                        alpha=0.10 if is_cent else 0.20)
        plotted_any = True
    if not plotted_any:
        plt.close(fig)
        return
    ax.set_xlabel("Round (FL) / Epoch (Centralised)")
    ax.set_ylabel("Across-client TEST PAPE (%)")
    ax.set_title(r"F4: Test PAPE — MAE-only ablation ($\lambda_{aux}=0$)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "F4_round_vs_test_pape_MAEonly.png", dpi=160)
    plt.close(fig)


def _figure_round_vs_train_loss_maeonly(traj: dict, fig_dir: Path) -> None:
    """F5 — same as F1c but only for ``-MAEonly`` cells."""
    cells = _maeonly_cells(traj)
    if not cells:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    plt.rcParams.update({"font.size": 13})
    plotted_any = False
    for cell in cells:
        key_x = f"{cell}_round_idx"
        key_y = f"{cell}_train_loss_main"
        if not _has(traj, key_x, key_y) or _all_nan(traj[key_y]):
            continue
        x = traj[key_x][0]
        m, s = _mean_std(traj[key_y])
        is_cent = cell.startswith(_CENTRALISED)
        if is_cent:
            x, m, s = _truncate_x(x, m, s, _CENTRALISED_MAX_X)
            m, s = _smooth_centralised(m, s, _CENTRALISED_SMOOTH_WINDOW)
        ax.plot(x, m,
                label=_label(cell.replace("-MAEonly", "")) + r" ($\lambda_{aux}=0$)",
                color=_color_for(cell),
                linewidth=1.8 if is_cent else 2.0,
                linestyle="--" if is_cent else "-")
        ax.fill_between(x, m - s, m + s, color=_color_for(cell),
                        alpha=0.10 if is_cent else 0.20)
        plotted_any = True
    if not plotted_any:
        plt.close(fig)
        return
    ax.set_xlabel("Round (FL) / Epoch (Centralised)")
    ax.set_ylabel("Round-averaged training main loss (MAE on z-norm)")
    ax.set_title(r"F5: Training loss — MAE-only ablation ($\lambda_{aux}=0$)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "F5_round_vs_train_loss_MAEonly.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="v06 Phase 1 figure renderer.")
    ap.add_argument("--traj_path", type=Path, default=None,
                    help="Override path to trajectories.npz (default: outputs/v06_round_dynamics/trajectories.npz).")
    ap.add_argument("--fig_dir", type=Path, default=None,
                    help="Override output directory (default: outputs/v06_round_dynamics/figures/).")
    ap.add_argument("--centralised_max_x", type=int, default=20,
                    help="Truncate the centralised reference curve at this epoch index "
                         "so its X-axis matches the FL round axis. 0 disables (default: 20).")
    ap.add_argument("--smooth_window", type=int, default=3,
                    help="Rolling-mean window applied to the centralised mean / std "
                         "before plotting (raw npz untouched). 0 or 1 disables (default: 3).")
    args = ap.parse_args()

    traj_path = args.traj_path or (OUTPUT_DIR / "v06_round_dynamics" / "trajectories.npz")
    fig_dir = args.fig_dir   or (OUTPUT_DIR / "v06_round_dynamics" / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    traj = dict(np.load(traj_path, allow_pickle=False))

    global _CENTRALISED_MAX_X, _CENTRALISED_SMOOTH_WINDOW
    _CENTRALISED_MAX_X = args.centralised_max_x
    _CENTRALISED_SMOOTH_WINDOW = args.smooth_window

    # Default-lambda main paper figures.
    _figure_round_vs_pape(traj, fig_dir)
    _figure_round_vs_test_pape(traj, fig_dir)
    _figure_round_vs_train_loss(traj, fig_dir)
    _figure_bytes_vs_pape(traj, fig_dir)
    _figure_drift_vs_round(traj, fig_dir)

    # MAE-only (λ_aux=0) ablation panels — drawn only if those cells exist.
    _figure_round_vs_test_pape_maeonly(traj, fig_dir)
    _figure_round_vs_train_loss_maeonly(traj, fig_dir)

    print(f"[v06 figures] wrote default panels (F1, F1b, F1c, F2, F3) and any "
          f"MAE-only ablation panels (F4, F5) found in npz under {fig_dir}/")


if __name__ == "__main__":
    main()
