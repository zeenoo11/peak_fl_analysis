"""E4: Per-cluster cold benefit map.

For T2 (peak_aux) + W5 best hyperparams:
    1. Build codebook (M=32 KMeans on training latents).
    2. Compute per-cluster offset.
    3. For each cold window: assign to cluster c via KEY-NN on training KEYs.
    4. Compute per-cluster Δ_PAPE = base_PAPE − corr_PAPE on cold gucha that
       routed to that cluster.
    5. Visualize: scatter of (cluster amp_mean, cluster hr_mean) sized by
       n_cold_routed and colored by Δ_PAPE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO
from dataloader.splits import load_v10_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.nbeatsx_aux import NBEATSxAux
from models.vq_kmeans import VectorQuantizerKMeans
from probes.peak_descriptor import extract_key
from utils.metrics import compute_pape

OUT = OUTPUT_DIR / "v01_peak_from_latent"
E4 = OUT / "E4"
FIG = E4 / "figures"
E4.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W5 = {"sigma": 3.0, "alpha_v0": 1.5, "alpha_w1": 0.5}


def gather(apts):
    m = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
    m.load_state_dict(torch.load(OUT / "T2" / "best.pt", map_location="cpu", weights_only=False))
    keys, lats, base_z, true_z, p_amp, p_hr, m_arr, s_arr = [], [], [], [], [], [], [], []
    for apt in apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            continue
        n = len(series); train_end = int(n * TRAIN_RATIO)
        seg = series[:train_end]
        mean = float(seg.mean()); std = float(seg.std()) if seg.std() > 1e-8 else 1.0
        ds = HouseholdDataset(seg, mean, std, stride=24)
        for x, y in DataLoader(ds, batch_size=256, shuffle=False):
            keys.append(extract_key(x.numpy()))
            with torch.no_grad():
                y_hat, hidd, (amp_p, hr_p) = m(x.to(DEVICE))
            lats.append(hidd["h_generic"].cpu().numpy())
            base_z.append(y_hat.cpu().numpy()); true_z.append(y.numpy())
            p_amp.append(amp_p.cpu().numpy()); p_hr.append(hr_p.argmax(dim=1).cpu().numpy())
            m_arr.append(np.full(len(y), mean)); s_arr.append(np.full(len(y), std))
    return {
        "key": np.concatenate(keys, 0), "lat": np.concatenate(lats, 0),
        "base_z": np.concatenate(base_z, 0), "true_z": np.concatenate(true_z, 0),
        "pred_amp": np.concatenate(p_amp, 0), "pred_hr": np.concatenate(p_hr, 0),
        "mean": np.concatenate(m_arr, 0), "std": np.concatenate(s_arr, 0),
    }


def main():
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    split = load_v10_split()
    tr = gather(split["train"]); co = gather(split["cold"])
    print(f"[data] train windows={tr['lat'].shape[0]}, cold windows={co['lat'].shape[0]}")

    # Build codebook + offsets
    vq = VectorQuantizerKMeans(num_embeddings=32, embedding_dim=tr["lat"].shape[1], random_state=RANDOM_SEED)
    diag = vq.fit(torch.from_numpy(tr["lat"]).float())
    cb = vq.codebook.cpu().numpy()
    d_tr = ((tr["lat"][:, None, :] - cb[None, :, :]) ** 2).sum(axis=2)
    idx_tr = d_tr.argmin(axis=1)
    M = cb.shape[0]
    offsets = np.zeros((M, 24), dtype=np.float32)
    cluster_amp_mean = np.full(M, np.nan)
    cluster_hr_mean = np.full(M, np.nan)
    cluster_n_train = np.zeros(M, dtype=np.int64)
    for c in range(M):
        mask = idx_tr == c
        cluster_n_train[c] = int(mask.sum())
        if mask.sum() > 0:
            offsets[c] = (tr["true_z"][mask] - tr["base_z"][mask]).mean(axis=0)
            peaks = tr["true_z"][mask].max(axis=1)
            hours = tr["true_z"][mask].argmax(axis=1)
            cluster_amp_mean[c] = float(peaks.mean())
            cluster_hr_mean[c] = float(hours.mean())

    # Cold cluster assignment via KEY-NN
    ks = StandardScaler().fit(tr["key"])
    nn = NearestNeighbors(n_neighbors=1).fit(ks.transform(tr["key"]))
    _, ni = nn.kneighbors(ks.transform(co["key"]))
    cold_cluster = idx_tr[ni[:, 0]]

    # Compute corrected
    sigma, av, aw = W5["sigma"], W5["alpha_v0"], W5["alpha_w1"]
    t = np.arange(24)[None, :]
    g = np.exp(-0.5 * ((t - co["pred_hr"][:, None]) / sigma) ** 2)
    g = g / g.max(axis=1, keepdims=True) * co["pred_amp"][:, None]
    corrected = co["base_z"] + av * offsets[cold_cluster] + aw * g

    true_kw = co["true_z"] * co["std"][:, None] + co["mean"][:, None]
    base_kw = co["base_z"] * co["std"][:, None] + co["mean"][:, None]
    corr_kw = corrected * co["std"][:, None] + co["mean"][:, None]

    # Per-cluster cold improvement
    cluster_n_cold = np.zeros(M, dtype=np.int64)
    cluster_base_pape = np.full(M, np.nan)
    cluster_corr_pape = np.full(M, np.nan)
    for c in range(M):
        mask = cold_cluster == c
        cluster_n_cold[c] = int(mask.sum())
        if mask.sum() >= 5:
            cluster_base_pape[c] = compute_pape(true_kw[mask], base_kw[mask])
            cluster_corr_pape[c] = compute_pape(true_kw[mask], corr_kw[mask])
    delta_pape = cluster_base_pape - cluster_corr_pape    # positive = cold improved
    rel_delta = (cluster_corr_pape - cluster_base_pape) / cluster_base_pape * 100   # negative = improved

    # Visualization
    valid = ~np.isnan(cluster_amp_mean) & ~np.isnan(delta_pape)
    print(f"[viz] {valid.sum()}/{M} clusters have valid (amp, hr, Δpape)")

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        cluster_hr_mean[valid],
        cluster_amp_mean[valid],
        s=cluster_n_cold[valid] / max(cluster_n_cold.max(), 1) * 600 + 30,
        c=delta_pape[valid], cmap="RdYlGn", alpha=0.85, edgecolors="k",
        vmin=-30, vmax=30,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Δ PAPE (kW)  positive = cold improved by KV-VQ")
    ax.set_xlabel("cluster mean peak HOUR (forecast horizon)")
    ax.set_ylabel("cluster mean peak AMP (z-space)")
    ax.set_title(f"E4: per-cluster cold benefit map (M=32, T2 + W5)\n"
                 f"size = #cold windows routed to cluster")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "E4_cluster_benefit_map.png", dpi=120); plt.close(fig)

    # Bar chart: top winners and losers
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_idx = np.argsort(delta_pape)[::-1]   # descending
    valid_sorted = [c for c in sorted_idx if not np.isnan(delta_pape[c])]
    top_n = min(15, len(valid_sorted))
    bars = ax.bar(
        range(len(valid_sorted)),
        [delta_pape[c] for c in valid_sorted],
        color=["green" if delta_pape[c] >= 0 else "red" for c in valid_sorted],
        edgecolor="k", alpha=0.7,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"cluster (sorted by Δ PAPE descending), M={M}")
    ax.set_ylabel("Δ PAPE (kW) — positive = cold improved")
    ax.set_title("E4: cluster-level Δ PAPE distribution")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "E4_cluster_delta_bars.png", dpi=120); plt.close(fig)

    # Save data
    out = {
        "M": int(M),
        "vq_diag": diag,
        "n_cold_unrouted_to_cluster": int((cluster_n_cold == 0).sum()),
        "cluster_data": [
            {"c": int(c), "amp_mean": None if np.isnan(cluster_amp_mean[c]) else float(cluster_amp_mean[c]),
             "hr_mean": None if np.isnan(cluster_hr_mean[c]) else float(cluster_hr_mean[c]),
             "n_train": int(cluster_n_train[c]), "n_cold": int(cluster_n_cold[c]),
             "base_pape": None if np.isnan(cluster_base_pape[c]) else float(cluster_base_pape[c]),
             "corr_pape": None if np.isnan(cluster_corr_pape[c]) else float(cluster_corr_pape[c]),
             "delta_pape": None if np.isnan(delta_pape[c]) else float(delta_pape[c]),
             "rel_delta_percent": None if np.isnan(rel_delta[c]) else float(rel_delta[c])}
            for c in range(M)
        ],
    }
    with open(E4 / "E4_results.json", "w") as fh:
        json.dump(out, fh, indent=2)

    # Print summary table
    print("\n========== E4 PER-CLUSTER COLD BENEFIT ==========")
    print(f"{'c':3s}  {'n_train':>8s}  {'n_cold':>7s}  {'amp_mean':>9s}  {'hr_mean':>8s}  "
          f"{'base_pape':>10s}  {'corr_pape':>10s}  {'Δ_pape':>8s}")
    print("-" * 80)
    for c in valid_sorted[:5]:
        print(f"  {c:2d}  {cluster_n_train[c]:8d}  {cluster_n_cold[c]:7d}  "
              f"{cluster_amp_mean[c]:9.2f}  {cluster_hr_mean[c]:8.1f}  "
              f"{cluster_base_pape[c]:10.2f}  {cluster_corr_pape[c]:10.2f}  {delta_pape[c]:+8.2f}")
    print("  ... (top 5 winners shown)")
    print("\n  ... (bottom 3 losers):")
    for c in valid_sorted[-3:]:
        print(f"  {c:2d}  {cluster_n_train[c]:8d}  {cluster_n_cold[c]:7d}  "
              f"{cluster_amp_mean[c]:9.2f}  {cluster_hr_mean[c]:8.1f}  "
              f"{cluster_base_pape[c]:10.2f}  {cluster_corr_pape[c]:10.2f}  {delta_pape[c]:+8.2f}")

    n_winner = sum(1 for c in valid_sorted if delta_pape[c] > 0)
    n_loser = sum(1 for c in valid_sorted if delta_pape[c] < 0)
    print(f"\nclusters: {n_winner} winners (cold improved), {n_loser} losers, "
          f"{len(valid_sorted)-n_winner-n_loser} flat")
    print(f"\n[done] wrote {E4 / 'E4_results.json'} + 2 PNGs")


if __name__ == "__main__":
    main()
