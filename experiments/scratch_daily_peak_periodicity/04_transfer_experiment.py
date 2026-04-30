"""Real A->B transfer experiment on daily peak forecasting.

Per-household 70/10/20 temporal split. Compare four training strategies on test MAE
in original kW units:
    B-only          : train ridge on B's train (personalization upper bound)
    Others-only     : train ridge on (all except B)'s train (pure transfer)
    Pool            : train ridge on everyone's train (naive FL)
    Cluster-pool    : train ridge on archetype-mates' train (pFL prior)

Archetype clustering uses ONLY train-portion features per household, so this is
honest: at deploy time, a new household reveals only its train; we infer its type
and route to the matching cluster model.

Run:
    uv run python experiments/scratch_daily_peak_periodicity/04_transfer_experiment.py --n 50 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
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


def build_features(y_z: np.ndarray, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, t, valid_idx). Skip rows where any lag is unavailable."""
    n = len(y_z)
    max_lag = max(LAGS)
    rows = []
    targets = []
    valid = []
    for i in range(max_lag, n):
        feat = [y_z[i - L] for L in LAGS]
        d = dates[i]
        dow = d.dayofweek
        doy = d.dayofyear
        feat += [
            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365),
        ]
        rows.append(feat)
        targets.append(y_z[i])
        valid.append(i)
    return np.array(rows), np.array(targets), np.array(valid)


def temporal_split(n_rows: int, train_ratio=0.7, val_ratio=0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(n_rows * train_ratio)
    n_val = int(n_rows * val_ratio)
    tr = np.arange(0, n_train)
    va = np.arange(n_train, n_train + n_val)
    te = np.arange(n_train + n_val, n_rows)
    return tr, va, te


def prep_household(y: pd.Series) -> dict:
    """Build per-household z-normalized features and splits using TRAIN stats only."""
    y_arr = y.to_numpy()
    dates = y.index
    # provisional split index in DAY space (before lag truncation)
    n_days = len(y_arr)
    n_tr_days = int(n_days * 0.7)
    train_mean = float(y_arr[:n_tr_days].mean())
    train_std = float(y_arr[:n_tr_days].std() + 1e-9)
    y_z = (y_arr - train_mean) / train_std
    X, t, valid = build_features(y_z, dates)
    # split rows whose anchor day is in train/val/test
    train_mask = valid < n_tr_days
    val_end = int(n_days * 0.8)
    val_mask = (valid >= n_tr_days) & (valid < val_end)
    test_mask = valid >= val_end
    return dict(
        X=X, t=t,
        train_idx=np.where(train_mask)[0],
        val_idx=np.where(val_mask)[0],
        test_idx=np.where(test_mask)[0],
        train_mean=train_mean, train_std=train_std,
        y_arr=y_arr, dates=dates,
    )


def archetype_features(h: dict) -> np.ndarray:
    """Compute train-only characterization features for clustering."""
    y_train = h["y_arr"][: int(len(h["y_arr"]) * 0.7)]
    dates_train = h["dates"][: int(len(h["y_arr"]) * 0.7)]
    if len(y_train) < 30 or y_train.std() < 1e-9:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    cv = y_train.std() / (y_train.mean() + 1e-9)
    a = acf(y_train, nlags=14, fft=True)
    ac1 = a[1] if len(a) > 1 else 0.0
    ac7 = a[7] if len(a) > 7 else 0.0
    weekend = (dates_train.dayofweek >= 5)
    weekend_ratio = (y_train[weekend].mean() / (y_train[~weekend].mean() + 1e-9)
                     if weekend.any() else 1.0)
    # peak season: argmax of monthly mean (1=jan, normalized to [0,1] cyclic via cos)
    monthly = pd.Series(y_train, index=dates_train).resample("ME").mean()
    if len(monthly) >= 2:
        peak_month = monthly.idxmax().month
    else:
        peak_month = 1
    peak_season_cos = np.cos(2 * np.pi * peak_month / 12)
    return np.array([cv, ac1, ac7, weekend_ratio, peak_season_cos])


def fit_predict(X_tr: np.ndarray, t_tr: np.ndarray, X_te: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return Ridge(alpha=alpha).fit(X_tr, t_tr).predict(X_te)


def mae_kw(t_pred_z: np.ndarray, t_true_z: np.ndarray, mu: float, sd: float) -> float:
    pred_kw = t_pred_z * sd + mu
    true_kw = t_true_z * sd + mu
    return float(np.mean(np.abs(pred_kw - true_kw)))


def baseline_mean(h: dict) -> float:
    """Test MAE of predicting the train mean (z=0)."""
    t_te = h["t"][h["test_idx"]]
    return mae_kw(np.zeros_like(t_te), t_te, h["train_mean"], h["train_std"])


def baseline_persistence(h: dict) -> float:
    """Test MAE of predicting lag-1 (already in feature column 0)."""
    X_te = h["X"][h["test_idx"]]
    t_te = h["t"][h["test_idx"]]
    return mae_kw(X_te[:, 0], t_te, h["train_mean"], h["train_std"])


def baseline_dow(h: dict) -> float:
    """Test MAE of predicting train-period mean by day-of-week."""
    y_train = h["y_arr"][: int(len(h["y_arr"]) * 0.7)]
    dates_train = h["dates"][: int(len(h["y_arr"]) * 0.7)]
    df_tr = pd.Series(y_train, index=dates_train)
    dow_means = df_tr.groupby(df_tr.index.dayofweek).mean()
    test_idx = h["test_idx"]
    test_dates = h["dates"][h["valid_anchor"][test_idx]] if "valid_anchor" in h else None
    # need anchor dates -- recompute
    max_lag = max(LAGS)
    valid_anchors = np.arange(max_lag, len(h["y_arr"]))
    test_anchors = valid_anchors[test_idx]
    pred_kw = np.array([dow_means.get(h["dates"][a].dayofweek, y_train.mean()) for a in test_anchors])
    true_kw = h["y_arr"][test_anchors]
    return float(np.mean(np.abs(pred_kw - true_kw)))


def run(args) -> None:
    rng = np.random.default_rng(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ids = list_household_ids()[: args.n]
    households = {}
    for apt in ids:
        try:
            y = load_daily_peak(apt)
            if len(y) < 100:
                continue
            households[apt] = prep_household(y)
        except Exception as e:
            print(f"  skipped Apt{apt}: {e}")
    apts = sorted(households.keys())
    N = len(apts)
    print(f"Households: {N} (target {args.n})")
    print(f"Per-HH days: {len(households[apts[0]]['y_arr'])}, "
          f"feature rows train/val/test = "
          f"{len(households[apts[0]]['train_idx'])}/"
          f"{len(households[apts[0]]['val_idx'])}/"
          f"{len(households[apts[0]]['test_idx'])}")

    # Archetype clustering on train-only features ----------------------------
    feats = np.stack([archetype_features(households[a]) for a in apts])
    feats_z = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)
    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10).fit(feats_z)
    cluster = {a: int(km.labels_[i]) for i, a in enumerate(apts)}
    print(f"\nArchetype clusters (K={args.k}): "
          + ", ".join(f"c{c}={sum(v == c for v in cluster.values())}" for c in range(args.k)))

    # Strategy evaluation per target household -------------------------------
    rows = []
    for tgt in apts:
        h = households[tgt]
        # baselines
        mae_meanB = baseline_mean(h)
        mae_persist = baseline_persistence(h)
        mae_dow = baseline_dow(h)

        X_te = h["X"][h["test_idx"]]
        t_te = h["t"][h["test_idx"]]

        # B-only
        pred_z = fit_predict(h["X"][h["train_idx"]], h["t"][h["train_idx"]], X_te)
        mae_B = mae_kw(pred_z, t_te, h["train_mean"], h["train_std"])

        # Others-only and Pool
        X_others, t_others, X_pool, t_pool, X_cluster, t_cluster = [], [], [], [], [], []
        cl_tgt = cluster[tgt]
        for src in apts:
            hs = households[src]
            X_src = hs["X"][hs["train_idx"]]
            t_src = hs["t"][hs["train_idx"]]
            if src != tgt:
                X_others.append(X_src); t_others.append(t_src)
                if cluster[src] == cl_tgt:
                    X_cluster.append(X_src); t_cluster.append(t_src)
            X_pool.append(X_src); t_pool.append(t_src)
        X_others = np.vstack(X_others); t_others = np.concatenate(t_others)
        X_pool   = np.vstack(X_pool);   t_pool   = np.concatenate(t_pool)
        if X_cluster:
            X_cluster = np.vstack(X_cluster); t_cluster = np.concatenate(t_cluster)
            pred_z = fit_predict(X_cluster, t_cluster, X_te)
            mae_cluster = mae_kw(pred_z, t_te, h["train_mean"], h["train_std"])
        else:
            mae_cluster = np.nan
        pred_z = fit_predict(X_others, t_others, X_te)
        mae_others = mae_kw(pred_z, t_te, h["train_mean"], h["train_std"])
        pred_z = fit_predict(X_pool, t_pool, X_te)
        mae_pool = mae_kw(pred_z, t_te, h["train_mean"], h["train_std"])

        rows.append(dict(
            apt=tgt, cluster=cl_tgt,
            mae_meanB=mae_meanB, mae_persist=mae_persist, mae_dow=mae_dow,
            mae_B=mae_B, mae_others=mae_others, mae_pool=mae_pool, mae_cluster=mae_cluster,
        ))
    df = pd.DataFrame(rows).set_index("apt")

    # ---- aggregation -------------------------------------------------------
    print("\n=== Test MAE (kW), median across households ===")
    cols = ["mae_meanB", "mae_persist", "mae_dow", "mae_B", "mae_others", "mae_pool", "mae_cluster"]
    for c in cols:
        q = df[c].quantile([0.25, 0.5, 0.75])
        print(f"  {c:<14s}: median={q[0.5]:5.3f}  IQR=[{q[0.25]:.3f}, {q[0.75]:.3f}]")

    # head-to-head deltas vs B-only
    df["delta_others"] = df["mae_others"] - df["mae_B"]   # >0 means others worse
    df["delta_pool"]   = df["mae_pool"]   - df["mae_B"]
    df["delta_cluster"]= df["mae_cluster"]- df["mae_B"]

    def boot_median(x, B=1000):
        x = np.asarray(x); ok = ~np.isnan(x); x = x[ok]
        if len(x) == 0: return (np.nan, np.nan, np.nan)
        meds = np.array([np.median(rng.choice(x, len(x), replace=True)) for _ in range(B)])
        return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))

    print("\n=== Δ MAE vs B-only (negative = transfer helps) ===")
    for c in ["delta_others", "delta_pool", "delta_cluster"]:
        m, lo, hi = boot_median(df[c].values)
        win = float((df[c] < 0).mean())
        print(f"  {c:<16s}: median={m:+.3f} kW  95%CI=[{lo:+.3f}, {hi:+.3f}]   "
              f"wins (Δ<0): {win:.1%}")

    # vs trivial baselines (sanity)
    print("\n=== Beat-baseline rates ===")
    for src, name in [("mae_B", "B-only"), ("mae_others", "Others-only"),
                       ("mae_pool", "Pool"), ("mae_cluster", "Cluster-pool")]:
        r = float((df[src] < df["mae_meanB"]).mean())
        print(f"  {name:<14s} beats B-mean : {r:.1%}")
        r = float((df[src] < df["mae_persist"]).mean())
        print(f"  {name:<14s} beats persist: {r:.1%}")

    # cluster-conditioned summary
    print("\n=== Per-archetype median MAE ===")
    print(df.groupby("cluster")[["mae_B", "mae_others", "mae_pool", "mae_cluster"]].median().round(3))
    print("\n=== Per-archetype size and Δ_pool median ===")
    g = df.groupby("cluster").agg(
        n=("mae_B", "size"),
        delta_pool_med=("delta_pool", "median"),
        delta_cluster_med=("delta_cluster", "median"),
    )
    print(g.round(3))

    df.to_csv(OUT_DIR / f"transfer_n{N}_seed{args.seed}.csv")

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    box = [df["mae_meanB"], df["mae_persist"], df["mae_dow"],
           df["mae_B"], df["mae_others"], df["mae_pool"], df["mae_cluster"].dropna()]
    labels = ["B-mean", "lag1", "DoW", "B-only", "Others", "Pool", "Cluster"]
    bp = axes[0].boxplot(box, tick_labels=labels, showmeans=True)
    axes[0].set_ylabel("Test MAE (kW)")
    axes[0].set_title("Per-household test MAE distribution")
    axes[0].tick_params(axis="x", rotation=20)

    for c, lbl, color in [("delta_others", "Others-only − B-only", "C0"),
                          ("delta_pool", "Pool − B-only", "C1"),
                          ("delta_cluster", "Cluster − B-only", "C2")]:
        x = df[c].dropna().values
        axes[1].hist(x, bins=20, alpha=0.5, label=lbl, color=color)
    axes[1].axvline(0, ls="--", color="k", lw=0.8)
    axes[1].set_xlabel("Δ MAE vs B-only (kW). <0 = transfer helps")
    axes[1].set_title("Where transfer wins / loses")
    axes[1].legend(fontsize=8)

    # scatter: B-only MAE vs delta_pool, colored by cluster
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        axes[2].scatter(sub["mae_B"], sub["delta_pool"], label=f"c{c} (n={len(sub)})", alpha=0.85)
    axes[2].axhline(0, ls="--", color="k", lw=0.8)
    axes[2].set_xlabel("B-only MAE (kW)"); axes[2].set_ylabel("Pool − B-only (kW)")
    axes[2].set_title("Who benefits from pooling?")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"A→B transfer experiment, N={N}, K={args.k}, seed={args.seed}", fontsize=11)
    fig.tight_layout()
    fig_path = OUT_DIR / f"transfer_n{N}_seed{args.seed}.png"
    fig.savefig(fig_path, dpi=120)
    print(f"\nSaved: {fig_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--k", type=int, default=3, help="number of archetype clusters")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args)
