"""1-minute test: is there enough shared peak structure across households for A->B transfer?

Three views on the same T x N daily-peak matrix:
  (1) pairwise Pearson correlation distribution (off-diagonal)
  (2) PCA scree -- how much variance is in the top common modes
  (3) leave-one-out: predict household i from the mean of all others, report R^2
       -- this is a strict lower bound on "what cross-household info buys you"

Run:
    uv run python experiments/scratch_daily_peak_periodicity/03_intercorr_pca_loo.py --n 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

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
    df = pd.read_csv(RAW_DIR / f"Apt{apt}_2016.csv", header=None, names=["ts", "kw"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts").sort_index()["kw"].resample("D").max().dropna()


def build_matrix(ids: list[int]) -> pd.DataFrame:
    series = {f"Apt{a}": load_daily_peak(a) for a in ids}
    df = pd.concat(series, axis=1)
    return df.dropna(axis=0, how="any")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ids = list_household_ids()[: args.n]
    X = build_matrix(ids)  # T x N
    T, N = X.shape
    print(f"Matrix: T={T} days × N={N} households (overlapping date range)")
    print(f"  span : {X.index.min().date()} .. {X.index.max().date()}")

    # (1) pairwise Pearson r --------------------------------------------------
    corr = X.corr().to_numpy()
    iu = np.triu_indices(N, k=1)
    r_off = corr[iu]
    print(f"\n=== (1) Pairwise Pearson r (off-diag, {len(r_off)} pairs) ===")
    print(f"  median = {np.median(r_off):+.3f}")
    print(f"  IQR    = [{np.percentile(r_off, 25):+.3f}, {np.percentile(r_off, 75):+.3f}]")
    print(f"  >|0.5| : {np.mean(np.abs(r_off) > 0.5):.1%} of pairs")
    print(f"  >|0.3| : {np.mean(np.abs(r_off) > 0.3):.1%} of pairs")

    # (2) PCA scree -----------------------------------------------------------
    Xz = (X - X.mean()) / X.std()  # standardize per household so big homes don't dominate
    pca = PCA(n_components=min(10, N)).fit(Xz.values)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    print(f"\n=== (2) PCA on standardized daily peaks ===")
    for k in range(min(10, N)):
        print(f"  PC{k+1}: var={evr[k]*100:5.2f}%   cumulative={cum[k]*100:5.2f}%")

    # (3) Leave-one-out: predict household i from the mean of others ----------
    Xv = X.values
    others_mean = (Xv.sum(axis=1, keepdims=True) - Xv) / (N - 1)
    r2_list = []
    for i in range(N):
        y = Xv[:, i]
        x = others_mean[:, i]
        # OLS y = a + b*x, R^2
        b, a = np.polyfit(x, y, 1)
        yhat = a + b * x
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        r2_list.append(r2)
    r2 = np.array(r2_list)
    print(f"\n=== (3) Leave-one-out: y_i ~ a + b*mean(others) ===")
    print(f"  R^2 median = {np.median(r2):.3f}")
    print(f"  R^2 IQR    = [{np.percentile(r2, 25):.3f}, {np.percentile(r2, 75):.3f}]")
    print(f"  R^2 max    = {r2.max():.3f}  (Apt{ids[r2.argmax()]})")
    print(f"  R^2 min    = {r2.min():.3f}  (Apt{ids[r2.argmin()]})")
    print(f"  fraction with R^2 > 0.30: {(r2 > 0.30).mean():.1%}")
    print(f"  fraction with R^2 > 0.50: {(r2 > 0.50).mean():.1%}")

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    im = axes[0].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_title("Pairwise Pearson r (50×50)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].hist(r_off, bins=30, color="C0", edgecolor="k", alpha=0.85)
    axes[1].axvline(0, ls=":", color="k", lw=0.8)
    axes[1].axvline(np.median(r_off), ls="--", color="C3", lw=1.0,
                    label=f"median={np.median(r_off):+.2f}")
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel("Pearson r"); axes[1].set_title("Off-diag r distribution")

    axes[2].bar(np.arange(1, len(evr) + 1), evr * 100, color="C1", edgecolor="k")
    axes[2].plot(np.arange(1, len(cum) + 1), cum * 100, "o-", color="C3", lw=1.5,
                 label="cumulative")
    axes[2].set_xlabel("PC"); axes[2].set_ylabel("% variance")
    axes[2].set_title(f"PCA scree (PC1={evr[0]*100:.0f}%, top3={cum[2]*100:.0f}%)")
    axes[2].legend(fontsize=8)

    axes[3].hist(r2, bins=20, color="C2", edgecolor="k", alpha=0.85)
    axes[3].axvline(np.median(r2), ls="--", color="C3", lw=1.0,
                    label=f"median R²={np.median(r2):.2f}")
    axes[3].set_xlabel("LOO R²"); axes[3].set_xlim(0, 1)
    axes[3].set_title("Predict-i-from-others R²")
    axes[3].legend(fontsize=8)

    fig.suptitle(f"Cross-household peak transferability — N={N}, T={T}d", fontsize=11)
    fig.tight_layout()
    fig_path = OUT_DIR / f"intercorr_pca_loo_n{N}.png"
    fig.savefig(fig_path, dpi=120)

    # also dump the per-household R^2 + PC loadings
    pd.DataFrame({"apt": ids, "loo_r2": r2}).to_csv(
        OUT_DIR / f"loo_r2_n{N}.csv", index=False
    )
    print(f"\nSaved fig : {fig_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
