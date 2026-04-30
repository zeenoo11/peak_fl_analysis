"""Quick exploratory: decompose one household's daily-peak series.

Loads UMass Apt{N}_2016.csv (minute-level), aggregates to daily peak (max per calendar day),
runs STL with weekly period and reports variance shares of trend / seasonal / residual.
Also dumps ACF + a periodogram-style FFT spectrum to inspect periodicity.

Run:
    uv run python experiments/scratch_daily_peak_periodicity/01_decompose_daily_peak.py --apt 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "Umass" / "2016"
OUT_DIR = PROJECT_ROOT / "outputs" / "scratch_daily_peak_periodicity"


def load_daily_peak(apt: int) -> pd.Series:
    csv = RAW_DIR / f"Apt{apt}_2016.csv"
    df = pd.read_csv(csv, header=None, names=["ts", "kw"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()
    daily = df["kw"].resample("D").max().dropna()
    return daily


def variance_share(series: np.ndarray) -> float:
    return float(np.var(series, ddof=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apt", type=int, default=1)
    parser.add_argument("--period", type=int, default=7, help="STL seasonal period (days)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    y = load_daily_peak(args.apt)
    print(f"[Apt{args.apt}] daily-peak series: n={len(y)}, "
          f"range={y.index.min().date()}..{y.index.max().date()}, "
          f"mean={y.mean():.3f} kW, std={y.std():.3f} kW")

    stl = STL(y, period=args.period, robust=True).fit()
    trend = stl.trend.to_numpy()
    seas = stl.seasonal.to_numpy()
    resid = stl.resid.to_numpy()
    obs = y.to_numpy()

    var_obs = variance_share(obs)
    shares = {
        "trend":    variance_share(trend) / var_obs,
        "seasonal": variance_share(seas) / var_obs,
        "residual": variance_share(resid) / var_obs,
    }
    # Strength-of-trend / strength-of-seasonality (Wang, Smith, Hyndman 2006)
    Ft = max(0.0, 1.0 - np.var(resid) / np.var(trend + resid))
    Fs = max(0.0, 1.0 - np.var(resid) / np.var(seas + resid))

    print(f"\nVariance shares (raw, sum can != 1 because components are correlated):")
    for k, v in shares.items():
        print(f"  {k:<9s}: {v:.3f}")
    print(f"\nStrength-of-trend       Ft = {Ft:.3f}  (1 = pure trend)")
    print(f"Strength-of-seasonality Fs = {Fs:.3f}  (1 = pure {args.period}-day cycle)")

    # ACF on raw daily peak (out to 60 lags ~ 2 months)
    acf_vals = acf(obs, nlags=60, fft=True)
    print("\nACF at notable lags:")
    for lag in [1, 2, 3, 7, 14, 21, 28, 30]:
        print(f"  lag {lag:>3d} d : {acf_vals[lag]:+.3f}")

    # FFT magnitude spectrum on detrended series
    detrended = obs - np.nanmean(obs)
    fft = np.fft.rfft(detrended)
    freqs = np.fft.rfftfreq(len(detrended), d=1.0)  # cycles/day
    mag = np.abs(fft)
    # Top 5 dominant frequencies (skip DC)
    order = np.argsort(mag[1:])[::-1] + 1
    print("\nTop-5 FFT frequencies on detrended daily peak (period in days):")
    for i in order[:5]:
        period_d = 1.0 / freqs[i] if freqs[i] > 0 else float("inf")
        print(f"  freq={freqs[i]:.4f} cyc/day  period={period_d:6.2f} d  mag={mag[i]:.2f}")

    # ---- plot ---------------------------------------------------------------
    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=False)
    axes[0].plot(y.index, obs, lw=0.9); axes[0].set_title(f"Apt{args.apt} daily peak (kW)")
    axes[1].plot(y.index, trend, lw=1.0, color="C1"); axes[1].set_title(f"STL trend (Ft={Ft:.2f})")
    axes[2].plot(y.index, seas, lw=0.8, color="C2");  axes[2].set_title(f"STL seasonal, period={args.period}d (Fs={Fs:.2f})")
    axes[3].plot(y.index, resid, lw=0.6, color="C3"); axes[3].axhline(0, lw=0.5, color="k"); axes[3].set_title("STL residual")
    axes[4].stem(np.arange(len(acf_vals)), acf_vals, basefmt=" ")
    axes[4].axhline(0, lw=0.5, color="k")
    axes[4].axhline(+1.96/np.sqrt(len(obs)), ls="--", lw=0.5, color="gray")
    axes[4].axhline(-1.96/np.sqrt(len(obs)), ls="--", lw=0.5, color="gray")
    axes[4].set_title("ACF of raw daily peak (lag = days)")
    axes[4].set_xlabel("lag [days]")
    fig.tight_layout()
    out_png = OUT_DIR / f"apt{args.apt}_period{args.period}_decomp.png"
    fig.savefig(out_png, dpi=120)
    print(f"\nSaved: {out_png.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
