"""Replicate paper H4 (amplitude-dominant heterogeneity) on our 50 households.

Paper §4.6 reports:
    hour-of-day cosine 0.970 mean, 0.811 min
    W1 amplitude         0.379 kW mean, 1.439 kW max

We compute the same two quantities on our 50hh train portions only.
If shapes are very similar (cosine ~ 0.97) but amplitudes spread widely (W1 in kW),
the cluster mechanism is amplitude-routing, not archetype discovery.

Run:
    uv run python experiments/scratch_daily_peak_periodicity/06_verify_paper_h4.py --n 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "Umass" / "2016"
OUT_DIR = PROJECT_ROOT / "outputs" / "scratch_daily_peak_periodicity"


def list_household_ids() -> list[int]:
    ids = []
    for p in RAW_DIR.glob("Apt*_2016.csv"):
        try:
            ids.append(int(p.stem.replace("Apt", "").replace("_2016", "")))
        except ValueError:
            continue
    return sorted(ids)


def load_hourly(apt: int) -> pd.Series:
    """Hourly resampling (mean within hour) — matches paper's hourly setup."""
    df = pd.read_csv(RAW_DIR / f"Apt{apt}_2016.csv", header=None, names=["ts", "kw"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts").sort_index()["kw"].resample("h").mean().dropna()


def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = list_household_ids()[: args.n]

    profiles_kw = []   # 24-d hour-of-day mean profile (raw kW)
    profiles_norm = [] # same, unit-norm (for cosine)
    train_means_kw = []
    train_arrays_kw = []  # full train series, for W1
    apts = []

    for apt in ids:
        try:
            y = load_hourly(apt)
        except Exception:
            continue
        n_tr = int(len(y) * 0.7)
        y_tr = y.iloc[:n_tr]
        # hour-of-day mean profile, in raw kW
        prof = y_tr.groupby(y_tr.index.hour).mean().reindex(range(24)).ffill().bfill().to_numpy()
        if np.any(np.isnan(prof)) or np.linalg.norm(prof) < 1e-9:
            continue
        profiles_kw.append(prof)
        profiles_norm.append(prof / np.linalg.norm(prof))
        train_means_kw.append(float(y_tr.mean()))
        train_arrays_kw.append(y_tr.to_numpy())
        apts.append(apt)

    P_kw = np.stack(profiles_kw)         # N x 24
    P_norm = np.stack(profiles_norm)     # N x 24, unit-norm
    N = len(apts)
    print(f"Households: {N}")

    # 1) hour-of-day cosine, pairwise on UNIT-NORM profiles --------------------
    cos_mat = P_norm @ P_norm.T
    iu = np.triu_indices(N, k=1)
    cos_off = cos_mat[iu]
    print("\n=== Hour-of-day cosine (unit-norm 24-d profiles) ===")
    print(f"  median = {np.median(cos_off):.4f}")
    print(f"  mean   = {np.mean(cos_off):.4f}")
    print(f"  min    = {np.min(cos_off):.4f}")
    print(f"  max    = {np.max(cos_off):.4f}")
    print(f"  paper's reference: mean=0.970, min=0.811")

    # 2) W1 amplitude in raw kW, pairwise on full train series -----------------
    print("\n=== Wasserstein-1 (amplitude) in kW ===")
    w1 = []
    for i in range(N):
        for j in range(i + 1, N):
            w1.append(wasserstein_distance(train_arrays_kw[i], train_arrays_kw[j]))
    w1 = np.array(w1)
    print(f"  mean = {w1.mean():.3f} kW")
    print(f"  max  = {w1.max():.3f} kW")
    print(f"  paper's reference: mean=0.379, max=1.439")

    # 3) Per-household train-mean kW spread (cheap proxy for amplitude axis) ----
    means = np.array(train_means_kw)
    print("\n=== Train-portion mean kW (amplitude axis) ===")
    print(f"  median   = {np.median(means):.3f} kW")
    print(f"  IQR      = [{np.percentile(means, 25):.3f}, {np.percentile(means, 75):.3f}]")
    print(f"  min/max  = {means.min():.3f} / {means.max():.3f} kW")
    print(f"  spread max/min = {means.max() / max(means.min(), 0.01):.1f}x")

    # 4) Decompose: shape vs amplitude split ----------------------------------
    # Shape: each profile / its own peak amplitude
    shape_norm = P_kw / P_kw.max(axis=1, keepdims=True)  # N x 24, peak=1
    cos_shape = (shape_norm / np.linalg.norm(shape_norm, axis=1, keepdims=True)) @ \
                (shape_norm / np.linalg.norm(shape_norm, axis=1, keepdims=True)).T
    cos_shape_off = cos_shape[iu]
    print("\n=== Shape similarity (peak-normalized 24-d, cosine) ===")
    print(f"  mean = {cos_shape_off.mean():.4f}, min = {cos_shape_off.min():.4f}")

    # Amplitude axis: scalar peak kW per household
    peak_kw = P_kw.max(axis=1)
    print(f"\n=== Per-household peak amplitude (kW) ===")
    print(f"  range = [{peak_kw.min():.2f}, {peak_kw.max():.2f}]   "
          f"spread = {peak_kw.max() / max(peak_kw.min(), 0.01):.1f}x")

    # ---- figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].hist(cos_off, bins=30, color="C0", edgecolor="k", alpha=0.85)
    axes[0].axvline(0.970, ls="--", color="C3", lw=1.0, label="paper mean=0.970")
    axes[0].axvline(0.811, ls=":",  color="C3", lw=1.0, label="paper min=0.811")
    axes[0].axvline(np.median(cos_off), ls="-", color="C2", lw=1.0,
                    label=f"ours median={np.median(cos_off):.3f}")
    axes[0].set_xlabel("hour-of-day cosine"); axes[0].set_title("Shape similarity")
    axes[0].legend(fontsize=7)

    axes[1].hist(w1, bins=30, color="C1", edgecolor="k", alpha=0.85)
    axes[1].axvline(0.379, ls="--", color="C3", lw=1.0, label="paper mean=0.379")
    axes[1].axvline(1.439, ls=":",  color="C3", lw=1.0, label="paper max=1.439")
    axes[1].axvline(w1.mean(), ls="-", color="C2", lw=1.0,
                    label=f"ours mean={w1.mean():.3f}")
    axes[1].set_xlabel("W1 amplitude (kW)"); axes[1].set_title("Amplitude heterogeneity")
    axes[1].legend(fontsize=7)

    # plot all 50 hour-of-day profiles, raw kW
    for p in P_kw:
        axes[2].plot(np.arange(24), p, alpha=0.35, lw=0.8)
    axes[2].plot(np.arange(24), P_kw.mean(0), "k-", lw=2, label="population mean")
    axes[2].set_xlabel("hour of day"); axes[2].set_ylabel("kW")
    axes[2].set_title(f"All {N} hour-of-day profiles (raw kW)")
    axes[2].legend(fontsize=8)

    # plot all 50 hour-of-day profiles, peak-normalised (= shape only)
    for s in shape_norm:
        axes[3].plot(np.arange(24), s, alpha=0.35, lw=0.8)
    axes[3].plot(np.arange(24), shape_norm.mean(0), "k-", lw=2, label="population mean")
    axes[3].set_xlabel("hour of day"); axes[3].set_ylabel("kW / peak")
    axes[3].set_title(f"Same, peak-normalised (= shape only)")
    axes[3].legend(fontsize=8)

    fig.suptitle(f"H4 verification on N={N}: shape similar? amplitude spread?", fontsize=11)
    fig.tight_layout()
    fig_path = OUT_DIR / f"h4_verify_n{N}.png"
    fig.savefig(fig_path, dpi=120)
    print(f"\nSaved: {fig_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    main(ap.parse_args())
