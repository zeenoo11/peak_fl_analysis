"""Train/cold household splits for v01 (50:50) and v02 (80:20).

The v10 50:50 split lives in an external YAML produced by
Peak_Analysis/experiments/federated/v10_0425_cold_split.py and is read here
for backward compatibility. The v02 80:20 split is generated locally by
``make_v02_split`` using the same stratification recipe (4-feature
KMeans(k=2) + alternating extraction + KL gate); only the train/cold ratio
changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import OUTPUT_DIR
from dataloader.umass import (
    filter_valid_apartments,
    list_available_apartments,
    load_apartment_hourly,
)

V10_YAML = (
    Path(__file__).resolve().parents[3]
    / "Peak_Analysis"
    / "configs"
    / "v10_households.yaml"
)

V02_SPLITS_DIR = OUTPUT_DIR / "v02_fl_8020_ratio" / "splits"


def load_v10_split(yaml_path: Path = V10_YAML) -> dict[str, list[str]]:
    """Return {'train': [...50 apts...], 'cold': [...50 apts...]}."""
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"v10 split yaml missing: {yaml_path}. "
            "v11 depends on the v10 train/cold split for comparability."
        )
    with open(yaml_path) as fh:
        raw = yaml.safe_load(fh)
    return {"train": list(raw["train"]), "cold": list(raw["cold"])}


def _apt_int(name: str) -> int:
    return int(name.replace("Apt", ""))


def _extract_features(apts: list[str], year: str = "2016") -> pd.DataFrame:
    """Per-apt 4-d profile features, mirroring v10_0425_cold_split.py.

    Features:
        mean             — annual hourly-kW mean
        std              — annual hourly-kW std
        daily_peak_mean  — mean of daily-max
        weekday_ratio    — fraction of weekday hours
    """
    records = []
    for apt in apts:
        try:
            series = load_apartment_hourly(apt, year)
            values = series.values.astype(np.float64)
            mean_ = float(np.mean(values))
            std_ = float(np.std(values))
            daily_max = series.resample("D").max().values
            daily_peak_mean = float(np.mean(daily_max[np.isfinite(daily_max)]))
            weekday_mask = series.index.weekday < 5
            weekday_ratio = float(weekday_mask.mean())
            records.append(
                {
                    "apt": apt,
                    "mean": mean_,
                    "std": std_,
                    "daily_peak_mean": daily_peak_mean,
                    "weekday_ratio": weekday_ratio,
                }
            )
        except Exception:
            continue
    return pd.DataFrame(records).set_index("apt")


def _compute_kl_divergence(
    train: list[str],
    cold: list[str],
    feat_df: pd.DataFrame,
    n_bins: int = 10,
    eps: float = 1e-10,
) -> float:
    """KL(cold || train) on the 'mean' feature, histogram with shared bin edges."""
    train_vals = feat_df.loc[train, "mean"].values
    cold_vals = feat_df.loc[cold, "mean"].values
    all_vals = np.concatenate([train_vals, cold_vals])
    bins = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)
    train_hist, _ = np.histogram(train_vals, bins=bins, density=True)
    cold_hist, _ = np.histogram(cold_vals, bins=bins, density=True)
    train_hist = train_hist + eps
    cold_hist = cold_hist + eps
    train_hist = train_hist / train_hist.sum()
    cold_hist = cold_hist / cold_hist.sum()
    return float(entropy(cold_hist, train_hist))


def _stratified_split(
    apts: list[str],
    feat_df: pd.DataFrame,
    n_train: int,
    n_cold: int,
    seed: int,
) -> tuple[list[str], list[str], dict]:
    """4-feature StandardScaler -> KMeans(k=2) -> per-cluster proportional extraction."""
    feat_matrix = feat_df.loc[apts].values
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat_matrix)

    km = KMeans(n_clusters=2, random_state=seed, n_init=10)
    labels = km.fit_predict(feat_scaled)

    cluster_0 = [apts[i] for i, lab in enumerate(labels) if lab == 0]
    cluster_1 = [apts[i] for i, lab in enumerate(labels) if lab == 1]

    rng = np.random.default_rng(seed)
    rng.shuffle(cluster_0)
    rng.shuffle(cluster_1)

    n_total = n_train + n_cold
    ratio_train = n_train / n_total
    n_train_from_0 = int(round(len(cluster_0) * ratio_train))
    # clamp so cluster 1 contribution stays in [0, len(cluster_1)]
    n_train_from_0 = max(n_train - len(cluster_1), min(len(cluster_0), n_train_from_0))
    n_train_from_1 = n_train - n_train_from_0

    train_from_0 = cluster_0[:n_train_from_0]
    cold_from_0 = cluster_0[n_train_from_0:]
    train_from_1 = cluster_1[:n_train_from_1]
    cold_from_1 = cluster_1[n_train_from_1:]

    train_list = sorted(train_from_0 + train_from_1, key=_apt_int)
    cold_list = sorted(cold_from_0 + cold_from_1, key=_apt_int)

    if len(cold_list) > n_cold:
        cold_list = cold_list[:n_cold]

    metadata = {
        "cluster_sizes": {
            "cluster_0": int(len(cluster_0)),
            "cluster_1": int(len(cluster_1)),
        },
        "n_train_from_cluster_0": int(n_train_from_0),
        "n_train_from_cluster_1": int(n_train_from_1),
        "n_train": int(len(train_list)),
        "n_cold": int(len(cold_list)),
        "kmeans_seed": int(seed),
    }
    return train_list, cold_list, metadata


def make_v02_split(
    seed: int,
    n_train: int = 80,
    n_cold: int = 20,
    year: str = "2016",
    min_hours: int = 7000,
    kl_threshold: float = 0.5,
) -> tuple[list[str], list[str], dict]:
    """Generate the v02 stratified train/cold split for one seed.

    Recipe (identical to v10 except for the train/cold ratio):
        1. List & filter apartments by minimum coverage (size-based heuristic,
           same as Peak_Analysis/v10).
        2. Take the first ``n_train + n_cold`` valid apts (deterministic order).
        3. Extract 4-d profile features.
        4. StandardScaler -> KMeans(k=2, random_state=seed) -> per-cluster
           proportional alternating extraction.
        5. KL(cold || train) on the 'mean' feature; if > threshold, retry once
           with seed + 1.

    Args:
        seed:          Both the KMeans random_state and the shuffle seed.
        n_train:       Number of training apartments.
        n_cold:        Number of cold-evaluation apartments.
        year:          Data year string, default '2016'.
        min_hours:     Minimum coverage in hours for the size-based filter.
        kl_threshold:  KL gate; above this we retry once with seed + 1.

    Returns:
        (train_list, cold_list, metadata) where metadata records the
        stratification provenance (cluster sizes, KL, retry flag, ...).
    """
    n_total = n_train + n_cold
    all_apts = list_available_apartments(year)
    valid_apts = filter_valid_apartments(all_apts, year=year, min_hours=min_hours)
    if len(valid_apts) < n_total:
        raise RuntimeError(
            f"valid apartments ({len(valid_apts)}) < required ({n_total}); "
            "consider relaxing min_hours."
        )
    apts_for_split = valid_apts[:n_total]

    feat_df = _extract_features(apts_for_split, year=year)
    if len(feat_df) < n_total:
        raise RuntimeError(
            f"feature extraction produced {len(feat_df)} rows < {n_total}; "
            "some apartments could not be loaded."
        )

    train_list, cold_list, meta = _stratified_split(
        apts_for_split, feat_df, n_train=n_train, n_cold=n_cold, seed=seed
    )
    kl = _compute_kl_divergence(train_list, cold_list, feat_df)
    meta["kl_divergence"] = kl
    meta["retry_seed"] = None

    if kl > kl_threshold:
        retry_seed = seed + 1
        train_list, cold_list, meta_retry = _stratified_split(
            apts_for_split, feat_df, n_train=n_train, n_cold=n_cold, seed=retry_seed
        )
        kl_retry = _compute_kl_divergence(train_list, cold_list, feat_df)
        meta = {**meta_retry, "kl_divergence": kl_retry, "retry_seed": int(retry_seed)}

    meta.update(
        {
            "year": year,
            "min_hours": int(min_hours),
            "kl_threshold": float(kl_threshold),
            "split_version": "v02",
            "seed": int(seed),
            "n_total_pool": int(n_total),
        }
    )
    return train_list, cold_list, meta


def v02_yaml_path(seed: int) -> Path:
    """Canonical filesystem location for a v02 per-seed split YAML."""
    return V02_SPLITS_DIR / f"v02_8020_seed{seed}.yaml"


def load_v02_split(seed: int) -> dict[str, list[str]]:
    """Read the saved v02 split for a given seed.

    Returns:
        {'train': [...80...], 'cold': [...20...]}.
    Raises:
        FileNotFoundError: if the per-seed YAML has not been generated yet.
    """
    path = v02_yaml_path(seed)
    if not path.exists():
        raise FileNotFoundError(
            f"v02 split yaml missing: {path}. "
            f"Run experiments/v02_fl_8020_ratio/01_make_split.py --seed {seed} first."
        )
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return {"train": list(raw["train"]), "cold": list(raw["cold"])}
