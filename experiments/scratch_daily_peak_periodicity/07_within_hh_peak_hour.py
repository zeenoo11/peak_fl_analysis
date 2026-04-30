"""Within-household peak-hour variability — explains the HR@k ceiling.

For each household, compute per-day peak hour from train portion (hourly series).
Aggregate within-HH circular std and the *oracle* HR@k ceiling (= fraction of days
the per-day peak falls within +-k of the household's modal peak hour). Compare to
paper's reported HR@1 = 26.4 / HR@2 = 38.0 (proposed) vs chance 12.5 / 20.8.

If oracle HR@2 ~= paper's 38%, the method is at the ceiling and HR isn't improvable.
If oracle HR@2 >> 38%, there is structural room paper hasn't extracted yet.

Run:
    uv run python experiments/scratch_daily_peak_periodicity/07_within_hh_peak_hour.py --n 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    df = pd.read_csv(RAW_DIR / f"Apt{apt}_2016.csv", header=None, names=["ts", "kw"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts").sort_index()["kw"].resample("h").mean().dropna()


def daily_peak_hours(hourly: pd.Series) -> np.ndarray:
    """Peak hour (0..23) per calendar day, from hourly series."""
    df = hourly.to_frame("kw"); df["date"] = df.index.date
    out = []
    for _, sub in df.groupby("date"):
        if len(sub) < 12:
            continue
        ph = int(sub["kw"].idxmax().hour)
        out.append(ph)
    return np.array(out)


def circular_std_h(hours: np.ndarray) -> tuple[float, float]:
    """Return (circular std in hours, mean resultant length R)."""
    if len(hours) == 0:
        return float("nan"), float("nan")
    th = 2 * np.pi * hours / 24
    R = np.abs(np.mean(np.exp(1j * th)))
    if R < 1e-9:
        return float("inf"), 0.0
    sd_rad = np.sqrt(-2.0 * np.log(R))
    return sd_rad * 24 / (2 * np.pi), float(R)


def circular_dist(a: int, b: int) -> int:
    d = abs(int(a) - int(b)) % 24
    return min(d, 24 - d)


def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = list_household_ids()[: args.n]

    rows = []
    all_hours = []
    all_modes = []
    for apt in ids:
        try:
            y = load_hourly(apt)
        except Exception:
            continue
        n_tr = int(len(y) * 0.7)
        ph = daily_peak_hours(y.iloc[:n_tr])
        if len(ph) < 30:
            continue
        sd, R = circular_std_h(ph)
        # modal hour
        mode_h = int(np.bincount(ph, minlength=24).argmax())
        # oracle HR@k = fraction within +-k of mode (circular)
        dists = np.array([circular_dist(h, mode_h) for h in ph])
        hr1 = float((dists <= 1).mean())
        hr2 = float((dists <= 2).mean())
        rows.append(dict(apt=apt, n_days=len(ph), circ_std_h=sd, R=R,
                         mode_h=mode_h, oracle_hr1=hr1, oracle_hr2=hr2))
        all_hours.append(ph); all_modes.append(mode_h)

    df = pd.DataFrame(rows).set_index("apt")
    df.to_csv(OUT_DIR / f"within_hh_hour_n{len(df)}.csv")

    print(f"Households: {len(df)}")
    print("\n=== Within-HH circular std of daily peak hour (in hours) ===")
    q = df["circ_std_h"].quantile([0.25, 0.5, 0.75])
    print(f"  median = {q[0.5]:.2f} h   IQR = [{q[0.25]:.2f}, {q[0.75]:.2f}]")
    print(f"  min / max = {df['circ_std_h'].min():.2f} / {df['circ_std_h'].max():.2f}")
    print(f"  fraction with circ_std > 4 h: {(df['circ_std_h'] > 4).mean():.1%}")
    print(f"  fraction with circ_std > 6 h: {(df['circ_std_h'] > 6).mean():.1%}")

    print("\n=== Oracle HR@k ceiling (= within-HH consistency to modal hour) ===")
    for col, label, chance in [("oracle_hr1", "HR@1 oracle", 12.5),
                                ("oracle_hr2", "HR@2 oracle", 20.8)]:
        q = df[col].quantile([0.25, 0.5, 0.75])
        print(f"  {label}: median={q[0.5]*100:5.1f}%  "
              f"IQR=[{q[0.25]*100:.1f}, {q[0.75]*100:.1f}]  "
              f"chance={chance:.1f}%")

    print("\n=== Comparison to paper's reported HR ===")
    print(f"  Proposed (paper §C.1): HR@1=26.4%, HR@2=38.0%")
    print(f"  Oracle ceiling (ours): HR@1={df['oracle_hr1'].median()*100:.1f}%,  "
          f"HR@2={df['oracle_hr2'].median()*100:.1f}%")
    print(f"  Headroom (oracle - proposed): "
          f"HR@1={df['oracle_hr1'].median()*100 - 26.4:+.1f} pp,  "
          f"HR@2={df['oracle_hr2'].median()*100 - 38.0:+.1f} pp")

    # ---- modal hour distribution across HHs --------------------------------
    print("\n=== Modal peak hour across households ===")
    mode_counts = pd.Series(all_modes).value_counts().sort_index()
    print(mode_counts.to_string())

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].hist(df["circ_std_h"], bins=20, color="C0", edgecolor="k", alpha=0.85)
    axes[0].axvline(df["circ_std_h"].median(), ls="--", color="C3",
                    label=f"median={df['circ_std_h'].median():.1f} h")
    axes[0].set_xlabel("within-HH circular std (h)")
    axes[0].set_title("Day-to-day peak-hour variability per HH")
    axes[0].legend(fontsize=8)

    axes[1].hist(df["oracle_hr2"] * 100, bins=20, color="C1", edgecolor="k", alpha=0.85,
                  label="oracle HR@2 ceiling")
    axes[1].axvline(38.0, ls="--", color="C3", lw=1.0, label="paper proposed (38%)")
    axes[1].axvline(20.8, ls=":", color="k", lw=1.0, label="chance (20.8%)")
    axes[1].axvline(df["oracle_hr2"].median() * 100, ls="-", color="C2", lw=1.0,
                    label=f"ours median ceiling={df['oracle_hr2'].median()*100:.1f}%")
    axes[1].set_xlabel("oracle HR@2 (%)")
    axes[1].set_title("Within-HH HR@2 ceiling")
    axes[1].legend(fontsize=7)

    # modal hour histogram across HHs
    axes[2].bar(mode_counts.index, mode_counts.values, color="C2", edgecolor="k", alpha=0.85)
    axes[2].set_xlabel("modal peak hour")
    axes[2].set_xticks(range(0, 24, 2))
    axes[2].set_title(f"Modal peak hour across {len(df)} HH")

    # 2D scatter: circ std vs oracle HR@2
    axes[3].scatter(df["circ_std_h"], df["oracle_hr2"] * 100, alpha=0.7, color="C0")
    axes[3].axhline(38.0, ls="--", color="C3", lw=0.8)
    axes[3].set_xlabel("circular std (h)"); axes[3].set_ylabel("oracle HR@2 (%)")
    axes[3].set_title("std vs achievable HR@2 per HH")

    fig.suptitle(f"Within-HH peak-hour analysis (N={len(df)}) — explains HR@k ceiling",
                 fontsize=11)
    fig.tight_layout()
    fig_path = OUT_DIR / f"within_hh_hour_n{len(df)}.png"
    fig.savefig(fig_path, dpi=120)
    print(f"\nSaved: {fig_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    main(ap.parse_args())
