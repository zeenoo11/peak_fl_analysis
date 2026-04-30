"""K-sweep with leakage-free protocol + random-cluster null.

Reuses the 70/10/20 daily-peak ridge setup from 04_, but:
  * sweeps K=2..8 on train-only archetype features
  * picks K* by VAL MAE of the cluster-pool ridge (no test peek)
  * reports TEST MAE only at K*
  * adds a random-cluster control (shuffled labels, same K, same sizes)
    -- shows whether cluster-conditioning is informative or just subsampling

Run:
    uv run python experiments/scratch_daily_peak_periodicity/05_k_sweep_random_control.py --n 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score
from statsmodels.tsa.stattools import acf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "Umass" / "2016"
OUT_DIR = PROJECT_ROOT / "outputs" / "scratch_daily_peak_periodicity"

LAGS = (1, 7, 14)


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


def build_features(y_z, dates):
    n = len(y_z)
    max_lag = max(LAGS)
    rows, targets, valid = [], [], []
    for i in range(max_lag, n):
        feat = [y_z[i - L] for L in LAGS]
        d = dates[i]
        feat += [
            np.sin(2 * np.pi * d.dayofweek / 7), np.cos(2 * np.pi * d.dayofweek / 7),
            np.sin(2 * np.pi * d.dayofyear / 365), np.cos(2 * np.pi * d.dayofyear / 365),
        ]
        rows.append(feat); targets.append(y_z[i]); valid.append(i)
    return np.array(rows), np.array(targets), np.array(valid)


def prep_household(y: pd.Series) -> dict:
    y_arr = y.to_numpy()
    dates = y.index
    n_days = len(y_arr)
    n_tr = int(n_days * 0.7)
    n_va = int(n_days * 0.8)  # train+val cutoff
    mu = float(y_arr[:n_tr].mean()); sd = float(y_arr[:n_tr].std() + 1e-9)
    y_z = (y_arr - mu) / sd
    X, t, valid = build_features(y_z, dates)
    return dict(
        X=X, t=t,
        train_idx=np.where(valid < n_tr)[0],
        val_idx=np.where((valid >= n_tr) & (valid < n_va))[0],
        test_idx=np.where(valid >= n_va)[0],
        train_mean=mu, train_std=sd,
        y_arr=y_arr, dates=dates, n_tr=n_tr,
    )


def archetype_features(h: dict) -> np.ndarray:
    """Train-only features: NO val/test data used."""
    n_tr = h["n_tr"]
    y_train = h["y_arr"][:n_tr]
    dates_train = h["dates"][:n_tr]
    if y_train.std() < 1e-9:
        return np.zeros(5)
    cv = y_train.std() / (y_train.mean() + 1e-9)
    a = acf(y_train, nlags=14, fft=True)
    weekend = (dates_train.dayofweek >= 5)
    weekend_ratio = (y_train[weekend].mean() / (y_train[~weekend].mean() + 1e-9)
                     if weekend.any() else 1.0)
    monthly = pd.Series(y_train, index=dates_train).resample("ME").mean()
    peak_month = monthly.idxmax().month if len(monthly) else 1
    return np.array([cv, a[1] if len(a) > 1 else 0.0, a[7] if len(a) > 7 else 0.0,
                     weekend_ratio, np.cos(2 * np.pi * peak_month / 12)])


def cluster_pool_mae(target: int, apts: list[int], households: dict,
                      labels: dict, eval_idx_key: str) -> float:
    """Train ridge on cluster-mates' TRAIN, evaluate on target's val OR test."""
    h = households[target]
    cl = labels[target]
    Xs, ts = [], []
    for src in apts:
        if src == target:
            continue
        if labels[src] != cl:
            continue
        hs = households[src]
        Xs.append(hs["X"][hs["train_idx"]])
        ts.append(hs["t"][hs["train_idx"]])
    if not Xs:  # singleton cluster: fall back to B-only
        Xs.append(h["X"][h["train_idx"]]); ts.append(h["t"][h["train_idx"]])
    Xtr = np.vstack(Xs); ttr = np.concatenate(ts)
    Xev = h["X"][h[eval_idx_key]]; tev = h["t"][h[eval_idx_key]]
    pred_z = Ridge(alpha=1.0).fit(Xtr, ttr).predict(Xev)
    pred_kw = pred_z * h["train_std"] + h["train_mean"]
    true_kw = tev * h["train_std"] + h["train_mean"]
    return float(np.mean(np.abs(pred_kw - true_kw)))


def b_only_mae(target: int, households: dict, eval_idx_key: str) -> float:
    h = households[target]
    Xtr = h["X"][h["train_idx"]]; ttr = h["t"][h["train_idx"]]
    Xev = h["X"][h[eval_idx_key]]; tev = h["t"][h[eval_idx_key]]
    pred_z = Ridge(alpha=1.0).fit(Xtr, ttr).predict(Xev)
    pred_kw = pred_z * h["train_std"] + h["train_mean"]
    true_kw = tev * h["train_std"] + h["train_mean"]
    return float(np.mean(np.abs(pred_kw - true_kw)))


def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = list_household_ids()[: args.n]
    households = {a: prep_household(load_daily_peak(a)) for a in ids
                  if len(load_daily_peak(a)) >= 100}
    apts = sorted(households.keys()); N = len(apts)
    print(f"Households: {N}")

    # train-only archetype features ------------------------------------------
    feats = np.stack([archetype_features(households[a]) for a in apts])
    feats_z = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)

    rng = np.random.default_rng(args.seed)

    # 1) silhouette on train-only features (no MAE involved) -----------------
    sil = {}
    for K in range(args.kmin, args.kmax + 1):
        km = KMeans(n_clusters=K, random_state=args.seed, n_init=10).fit(feats_z)
        sil[K] = silhouette_score(feats_z, km.labels_) if K >= 2 else np.nan

    # 2) For each K: compute VAL MAE of cluster-pool, plus random-cluster control
    rows = []
    val_b_only = {a: b_only_mae(a, households, "val_idx") for a in apts}
    test_b_only = {a: b_only_mae(a, households, "test_idx") for a in apts}

    for K in range(args.kmin, args.kmax + 1):
        km = KMeans(n_clusters=K, random_state=args.seed, n_init=10).fit(feats_z)
        labels = {a: int(km.labels_[i]) for i, a in enumerate(apts)}
        # random control: same sizes, shuffled assignment
        rand_labels_arr = km.labels_.copy()
        rng.shuffle(rand_labels_arr)
        rand_labels = {a: int(rand_labels_arr[i]) for i, a in enumerate(apts)}

        for a in apts:
            mae_val = cluster_pool_mae(a, apts, households, labels, "val_idx")
            mae_val_rand = cluster_pool_mae(a, apts, households, rand_labels, "val_idx")
            mae_test = cluster_pool_mae(a, apts, households, labels, "test_idx")
            mae_test_rand = cluster_pool_mae(a, apts, households, rand_labels, "test_idx")
            rows.append(dict(
                K=K, apt=a, cluster=labels[a],
                val_b=val_b_only[a], val_cluster=mae_val, val_rand=mae_val_rand,
                test_b=test_b_only[a], test_cluster=mae_test, test_rand=mae_test_rand,
            ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"k_sweep_n{N}_seed{args.seed}.csv", index=False)

    # 3) aggregate ------------------------------------------------------------
    agg = df.groupby("K").agg(
        val_b_med=("val_b", "median"),
        val_cluster_med=("val_cluster", "median"),
        val_rand_med=("val_rand", "median"),
        test_b_med=("test_b", "median"),
        test_cluster_med=("test_cluster", "median"),
        test_rand_med=("test_rand", "median"),
    )
    agg["val_delta"]      = agg["val_cluster_med"] - agg["val_b_med"]
    agg["val_delta_rand"] = agg["val_rand_med"]    - agg["val_b_med"]
    agg["test_delta"]     = agg["test_cluster_med"]- agg["test_b_med"]
    agg["test_delta_rand"]= agg["test_rand_med"]   - agg["test_b_med"]
    agg["silhouette"]     = pd.Series(sil)
    agg = agg.round(4)
    print("\n=== K-sweep summary (median across households) ===")
    print(agg[["silhouette", "val_b_med", "val_cluster_med", "val_rand_med",
               "val_delta", "val_delta_rand"]])
    print("\n=== Test results (held-out -- DO NOT use to pick K) ===")
    print(agg[["test_b_med", "test_cluster_med", "test_rand_med",
               "test_delta", "test_delta_rand"]])

    # 4) pick K* by VAL ------------------------------------------------------
    K_star = int(agg["val_cluster_med"].idxmin())
    print(f"\n>>> K* selected by validation MAE: K={K_star} "
          f"(val_cluster={agg.loc[K_star,'val_cluster_med']:.3f}, "
          f"silhouette={agg.loc[K_star,'silhouette']:.3f})")
    print(f"   → test_cluster MAE @K*={K_star}: "
          f"{agg.loc[K_star,'test_cluster_med']:.3f}  "
          f"(B-only: {agg.loc[K_star,'test_b_med']:.3f}, "
          f"Δ={agg.loc[K_star,'test_delta']:+.3f}, "
          f"random Δ={agg.loc[K_star,'test_delta_rand']:+.3f})")

    # 5) figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    Ks = list(agg.index)

    axes[0].plot(Ks, agg["silhouette"], "o-", color="C0")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("silhouette (train features)")
    axes[0].set_title("Train-only clustering quality")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(Ks, agg["val_cluster_med"], "o-", color="C2", label="cluster-pool")
    axes[1].plot(Ks, agg["val_rand_med"],    "s--", color="C3", label="random-pool")
    axes[1].axhline(agg["val_b_med"].iloc[0], ls=":", color="k", lw=0.8, label="B-only")
    axes[1].axvline(K_star, ls=":", color="C1", lw=1.0, label=f"K*={K_star}")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("val MAE (kW)")
    axes[1].set_title("Val MAE — used to pick K*")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    axes[2].plot(Ks, agg["test_delta"],      "o-", color="C2", label="cluster − B-only")
    axes[2].plot(Ks, agg["test_delta_rand"], "s--", color="C3", label="random − B-only")
    axes[2].axhline(0, ls=":", color="k", lw=0.8)
    axes[2].axvline(K_star, ls=":", color="C1", lw=1.0)
    axes[2].set_xlabel("K"); axes[2].set_ylabel("Δ test MAE (kW)")
    axes[2].set_title("Test Δ vs B-only (informative if cluster < random)")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"K-sweep with random-cluster null, leakage-free K selection (N={N}, seed={args.seed})",
                 fontsize=11)
    fig.tight_layout()
    fig_path = OUT_DIR / f"k_sweep_n{N}_seed{args.seed}.png"
    fig.savefig(fig_path, dpi=120)
    print(f"\nSaved: {fig_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
