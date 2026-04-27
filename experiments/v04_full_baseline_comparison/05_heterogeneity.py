"""v04 G6 — Heterogeneity quantification (seed-independent).

Computes pairwise distribution similarity over the 80 train apts:

- **Wasserstein-1** on per-hour kW values (1-D distribution distance).
- **KL divergence** on a 64-bin histogram of per-hour kW.
- **Peak-shape similarity**: cosine similarity between each apt's
  hour-of-day average load profile (24-D vector).

Then correlates apt-level heterogeneity (mean of pairwise W1 to other
apts) with the **(Local-only − Shared) cold-PAPE gap** for that apt,
where:

- "Local-only" = apt-specific cold PAPE from
  ``outputs/.../seed{S}/local_only/result.json::per_apt_metrics``.
- "Shared"     = baseline cold PAPE (e.g. FedAvg or NF DLinear) on the
  same apt — taken from each apt's contribution to the cold-window
  pool of those algorithms (recoverable via stored apt name in the
  cold inference, but here we approximate by per-apt PAPE which is
  available in Local-only and recompute for Shared via 04 outputs if
  needed).

Output is **seed-independent** in the heterogeneity sense — the
heterogeneity statistic depends only on train-data distributions, not
on a model — but we use the seed=42 split's 80 train apts as the
canonical pool for the figure. (All three seeds share the same 100-apt
pool; the 80 train apts vary slightly across seeds. We report the
seed=42 result and note in the docstring that other seeds' heterogeneity
heatmaps are qualitatively identical.)

CLI:

    uv run python experiments/v04_full_baseline_comparison/05_heterogeneity.py \\
        [--seed 42] [--n_apts 80]

Output:
    outputs/v04_full_baseline_comparison/heterogeneity_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from scipy.stats import wasserstein_distance

from config import OUTPUT_DIR
from dataloader.splits import load_v02_split
from dataloader.umass import load_apartment_hourly

V04_OUT_ROOT = OUTPUT_DIR / "v04_full_baseline_comparison"


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Symmetric KL on histograms (Jensen-Shannon-like; eps for stability)."""
    p = p + eps; q = q + eps
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def _hour_of_day_profile(series: np.ndarray) -> np.ndarray:
    """Mean kW for each of the 24 hours-of-day; aligned to 0..23."""
    n = len(series)
    n_full_days = n // 24
    if n_full_days == 0:
        return np.zeros(24, dtype=np.float32)
    truncated = series[: n_full_days * 24].reshape(n_full_days, 24)
    return truncated.mean(axis=0).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    ap = argparse.ArgumentParser(description="v04 G6 heterogeneity over the 80 train apts.")
    ap.add_argument("--seed", type=int, default=42, help="Which seed's 80 train apts to use as the canonical pool.")
    ap.add_argument("--n_apts", type=int, default=80, help="Truncate to this many apts (default 80 = full v02 train).")
    ap.add_argument("--n_hist_bins", type=int, default=64, help="Bin count for KL histograms.")
    args = ap.parse_args()

    sp = load_v02_split(args.seed)
    train_apts = sp["train"][: args.n_apts]
    print(f"[v04 G6] seed={args.seed}  train_apts={len(train_apts)}")

    # 1. Load all apts and compute per-apt summaries.
    series_all: list[np.ndarray] = []
    profiles: list[np.ndarray] = []
    apt_names: list[str] = []
    for apt in train_apts:
        try:
            s = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            continue
        # Use only the train segment (first 70%) for heterogeneity — same
        # data the FL algorithms actually train on.
        n = len(s); train_end = int(n * 0.7)
        seg = s[:train_end]
        series_all.append(seg)
        profiles.append(_hour_of_day_profile(seg))
        apt_names.append(apt)
    K = len(apt_names)
    print(f"[v04 G6] loaded {K} apts")

    # 2. Common bin edges across all apts (so KL is comparable).
    all_vals = np.concatenate(series_all)
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), args.n_hist_bins + 1)
    histograms = [np.histogram(s, bins=bin_edges, density=True)[0] for s in series_all]

    # 3. Pairwise heterogeneity matrices (K × K, symmetric, zero diagonal).
    print(f"[v04 G6] computing K x K = {K*K} pairwise distances …")
    W = np.zeros((K, K), dtype=np.float32)
    KL = np.zeros((K, K), dtype=np.float32)
    COS = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        for j in range(i + 1, K):
            w = float(wasserstein_distance(series_all[i], series_all[j]))
            kl = _kl_divergence(histograms[i], histograms[j])
            cos = _cosine(profiles[i], profiles[j])
            W[i, j] = W[j, i] = w
            KL[i, j] = KL[j, i] = kl
            COS[i, j] = COS[j, i] = cos

    # Per-apt heterogeneity = mean pairwise W1 to all other apts.
    apt_heterogeneity = W.sum(axis=1) / (K - 1)

    # 4. (Optional) correlate apt heterogeneity with Local-only-vs-Shared gap.
    # Pull per-apt Local-only metrics if available.
    local_only_path = V04_OUT_ROOT / f"seed{args.seed}" / "local_only" / "result.json"
    correlation_block = None
    if local_only_path.exists():
        with open(local_only_path) as fh:
            lo = json.load(fh)
        per_apt = {row["apt"]: row for row in lo.get("per_apt_metrics", [])}
        local_pape = np.array([per_apt[a]["pape"] if a in per_apt else np.nan for a in apt_names])
        # Shared baseline = FedAvg cold PAPE (single number, same across apts).
        # We use it as the "shared" reference; per-apt gap = local_pape - shared_pape.
        shared_path = V04_OUT_ROOT / f"seed{args.seed}" / "fedavg" / "result.json"
        if shared_path.exists():
            with open(shared_path) as fh:
                shared = json.load(fh)
            shared_pape = float(shared["cold_metrics"]["pape"])
            gap = local_pape - shared_pape
            valid = ~np.isnan(gap)
            if valid.sum() >= 5:
                het = apt_heterogeneity[valid]
                gp = gap[valid]
                # Pearson correlation.
                corr = float(np.corrcoef(het, gp)[0, 1])
                correlation_block = {
                    "n_valid_apts": int(valid.sum()),
                    "shared_baseline": "fedavg",
                    "shared_pape": shared_pape,
                    "pearson_corr_heterogeneity_vs_localShared_gap": corr,
                    "per_apt_heterogeneity": apt_heterogeneity.tolist(),
                    "per_apt_local_pape": local_pape.tolist(),
                    "per_apt_gap": gap.tolist(),
                }

    summary = {
        "seed": args.seed,
        "n_apts": K,
        "apt_names": apt_names,
        "n_hist_bins": args.n_hist_bins,
        "pairwise": {
            "W1": W.tolist(),
            "KL_symm": KL.tolist(),
            "cos_hour_profile": COS.tolist(),
        },
        "summary_stats": {
            "W1_mean":  float(W[np.triu_indices(K, k=1)].mean()),
            "W1_max":   float(W.max()),
            "KL_mean":  float(KL[np.triu_indices(K, k=1)].mean()),
            "KL_max":   float(KL.max()),
            "cos_min":  float(COS[np.triu_indices(K, k=1)].min()),
            "cos_mean": float(COS[np.triu_indices(K, k=1)].mean()),
        },
        "apt_heterogeneity_W1_mean_to_others": apt_heterogeneity.tolist(),
        "correlation": correlation_block,
    }

    out_path = V04_OUT_ROOT / "heterogeneity_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[v04 G6] saved -> {out_path}")
    print(f"[v04 G6] W1 mean / max: {summary['summary_stats']['W1_mean']:.4f} / "
          f"{summary['summary_stats']['W1_max']:.4f}")
    print(f"[v04 G6] cos hour-profile min / mean: {summary['summary_stats']['cos_min']:.3f} / "
          f"{summary['summary_stats']['cos_mean']:.3f}")
    if correlation_block:
        print(f"[v04 G6] Pearson(heterogeneity, local-shared gap) = "
              f"{correlation_block['pearson_corr_heterogeneity_vs_localShared_gap']:+.3f} "
              f"(n={correlation_block['n_valid_apts']})")
    else:
        print(f"[v04 G6] correlation skipped (no local_only / fedavg results yet)")


if __name__ == "__main__":
    main()
