"""Same decomposition as 01_, but across many households.

Loops over the first N apartments, runs STL(period=7), records variance shares,
strength-of-trend / strength-of-seasonality, and ACF at notable lags. Outputs a
CSV summary plus a panel figure (distributions across households).

Run:
    uv run python experiments/scratch_daily_peak_periodicity/02_decompose_50hh.py --n 50
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

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


def load_daily_peak(apt: int) -> pd.Series:
    csv = RAW_DIR / f"Apt{apt}_2016.csv"
    df = pd.read_csv(csv, header=None, names=["ts", "kw"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()
    return df["kw"].resample("D").max().dropna()


def analyze(y: pd.Series, period: int = 7) -> dict | None:
    n = len(y)
    if n < 2 * period + 10:
        return None
    obs = y.to_numpy()
    if np.var(obs) < 1e-9:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stl = STL(y, period=period, robust=True).fit()
    trend = stl.trend.to_numpy()
    seas = stl.seasonal.to_numpy()
    resid = stl.resid.to_numpy()
    var_obs = np.var(obs)
    Ft = max(0.0, 1.0 - np.var(resid) / max(np.var(trend + resid), 1e-12))
    Fs = max(0.0, 1.0 - np.var(resid) / max(np.var(seas + resid), 1e-12))
    acf_vals = acf(obs, nlags=30, fft=True)
    return dict(
        n=n,
        mean_kw=float(obs.mean()),
        std_kw=float(obs.std()),
        var_trend=float(np.var(trend) / var_obs),
        var_seas=float(np.var(seas) / var_obs),
        var_resid=float(np.var(resid) / var_obs),
        Ft=float(Ft),
        Fs=float(Fs),
        acf1=float(acf_vals[1]),
        acf7=float(acf_vals[7]),
        acf14=float(acf_vals[14]),
        acf21=float(acf_vals[21]),
        acf28=float(acf_vals[28]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="# of households to analyze")
    ap.add_argument("--period", type=int, default=7)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = list_household_ids()[: args.n]
    rows = []
    for apt in ids:
        try:
            y = load_daily_peak(apt)
            r = analyze(y, period=args.period)
            if r is None:
                continue
            r["apt"] = apt
            rows.append(r)
        except Exception as e:  # pragma: no cover (exploratory script)
            print(f"  skipped Apt{apt}: {e}")
            continue
    df = pd.DataFrame(rows).set_index("apt")
    csv_path = OUT_DIR / f"summary_first{args.n}hh.csv"
    df.to_csv(csv_path)

    print(f"\nAnalyzed {len(df)} households (target N={args.n})")
    print(f"\n=== Variance shares (median [IQR]) ===")
    for col in ["var_trend", "var_seas", "var_resid"]:
        q = df[col].quantile([0.25, 0.5, 0.75])
        print(f"  {col:<10s}: median={q[0.5]:.3f}  IQR=[{q[0.25]:.3f}, {q[0.75]:.3f}]")

    print(f"\n=== Strength of trend / seasonality ===")
    for col in ["Ft", "Fs"]:
        q = df[col].quantile([0.25, 0.5, 0.75])
        print(f"  {col}: median={q[0.5]:.3f}  IQR=[{q[0.25]:.3f}, {q[0.75]:.3f}]"
              f"   max={df[col].max():.3f}  min={df[col].min():.3f}")

    print(f"\n=== ACF at weekly multiples (median across households) ===")
    for lag in [1, 7, 14, 21, 28]:
        col = f"acf{lag}"
        q = df[col].quantile([0.25, 0.5, 0.75])
        print(f"  lag {lag:>2d}d : median={q[0.5]:+.3f}  IQR=[{q[0.25]:+.3f}, {q[0.75]:+.3f}]")

    # How many households actually look "weekly seasonal"?
    fs_strong = (df["Fs"] >= 0.4).sum()
    fs_mid = ((df["Fs"] >= 0.2) & (df["Fs"] < 0.4)).sum()
    fs_weak = (df["Fs"] < 0.2).sum()
    print(f"\nFs categorization (Hyndman convention: >=0.4 = clearly seasonal):")
    print(f"  strong (Fs>=0.4): {fs_strong:3d}/{len(df)}")
    print(f"  mid    (0.2-0.4): {fs_mid:3d}/{len(df)}")
    print(f"  weak   (Fs<0.2) : {fs_weak:3d}/{len(df)}")

    # ---- panel figure ------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    axes[0, 0].hist(df["var_trend"], bins=20, color="C1", edgecolor="k", alpha=0.85)
    axes[0, 0].set_title("var(trend) / var(obs)"); axes[0, 0].set_xlim(0, 1)
    axes[0, 1].hist(df["var_seas"], bins=20, color="C2", edgecolor="k", alpha=0.85)
    axes[0, 1].set_title("var(seasonal) / var(obs)"); axes[0, 1].set_xlim(0, 1)
    axes[0, 2].hist(df["var_resid"], bins=20, color="C3", edgecolor="k", alpha=0.85)
    axes[0, 2].set_title("var(residual) / var(obs)"); axes[0, 2].set_xlim(0, 1)

    axes[1, 0].hist(df["Ft"], bins=20, color="C1", edgecolor="k", alpha=0.85)
    axes[1, 0].set_title(f"Strength-of-trend Ft (median={df['Ft'].median():.2f})")
    axes[1, 0].set_xlim(0, 1)
    axes[1, 1].hist(df["Fs"], bins=20, color="C2", edgecolor="k", alpha=0.85)
    axes[1, 1].axvline(0.4, ls="--", color="k", lw=0.8, label="Fs=0.4 (Hyndman)")
    axes[1, 1].axvline(0.2, ls=":",  color="gray", lw=0.8, label="Fs=0.2")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title(f"Strength-of-seasonality Fs (median={df['Fs'].median():.2f})")
    axes[1, 1].set_xlim(0, 1)

    box_data = [df[f"acf{l}"].values for l in [1, 7, 14, 21, 28]]
    axes[1, 2].boxplot(box_data, labels=["1", "7", "14", "21", "28"])
    axes[1, 2].axhline(0, ls=":", color="k", lw=0.5)
    axes[1, 2].set_title("ACF across households")
    axes[1, 2].set_xlabel("lag [days]")

    fig.suptitle(f"Daily-peak decomposition across {len(df)} UMass households (period={args.period}d)")
    fig.tight_layout()
    fig_path = OUT_DIR / f"summary_first{args.n}hh.png"
    fig.savefig(fig_path, dpi=120)

    print(f"\nSaved CSV : {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved fig : {fig_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
