"""iter3: hour-aware codebook variants on T2 (h_generic + peak_aux).

Diagnoses:
    - α sweep extended to {1.75, 2.0}
    - peak_hour distribution histogram (train + cold)
    - cluster offset shape visualization (does codebook capture hour?)

Variants (all on T2, no retraining):
    V0  baseline               KMeans M=32 on h_generic, KEY weights flat
    V2  hour-stratified VQ     4 hour-bins × M=8 each (total 32)
    V3  weighted KEY-NN        KEY argmax_norm weighted {3x, 5x, 10x}
    V4  smaller M              M=8 flat (forces fewer broad clusters)

Metrics tracked: cold PAPE, HR@1, HR@2, peak position MAE.
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
from utils.metrics import compute_hr, compute_pape

OUT = OUTPUT_DIR / "v01_peak_from_latent"
ITER3 = OUT / "iter3"
FIG = ITER3 / "figures"
ITER3.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARM = "T2"
ALPHAS = [0.5, 1.0, 1.5, 1.75, 2.0]
HOUR_BINS = [(0, 6), (6, 12), (12, 18), (18, 24)]


def gather(apts):
    m = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
    m.load_state_dict(torch.load(OUT / ARM / "best.pt", map_location="cpu", weights_only=False))
    keys, lats, base_z, true_z, m_arr, s_arr, apt_arr = [], [], [], [], [], [], []
    for apt in apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            continue
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        seg = series[:train_end]
        mean = float(seg.mean()); std = float(seg.std()) if seg.std() > 1e-8 else 1.0
        ds = HouseholdDataset(seg, mean, std, stride=24)
        for x, y in DataLoader(ds, batch_size=256, shuffle=False):
            keys.append(extract_key(x.numpy()))
            with torch.no_grad():
                y_hat, hidd, _ = m(x.to(DEVICE))
            lats.append(hidd["h_generic"].cpu().numpy())
            base_z.append(y_hat.cpu().numpy())
            true_z.append(y.numpy())
            m_arr.append(np.full(len(y), mean)); s_arr.append(np.full(len(y), std))
            apt_arr.append([apt] * len(y))
    return {
        "key": np.concatenate(keys, axis=0),
        "lat": np.concatenate(lats, axis=0),
        "base_z": np.concatenate(base_z, axis=0),
        "true_z": np.concatenate(true_z, axis=0),
        "mean": np.concatenate(m_arr, axis=0),
        "std": np.concatenate(s_arr, axis=0),
        "apt": np.array([a for chunk in apt_arr for a in chunk]),
    }


def assign(lat, codebook):
    d = ((lat[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)


def build_offsets(lat_tr, base_tr, true_tr, codebook):
    idx = assign(lat_tr, codebook)
    M = codebook.shape[0]
    offsets = np.zeros((M, 24), dtype=np.float32)
    counts = np.zeros(M, dtype=np.int64)
    for c in range(M):
        mask = idx == c
        counts[c] = int(mask.sum())
        if counts[c] > 0:
            offsets[c] = (true_tr[mask] - base_tr[mask]).mean(axis=0)
    return offsets, counts, idx


def cold_assign_via_key(K_tr, K_co, idx_tr, key_weights=None):
    """KEY-NN assignment, optionally weighting KEY dimensions."""
    K_tr = K_tr.copy(); K_co = K_co.copy()
    if key_weights is not None:
        w = np.asarray(key_weights, dtype=np.float32)
        K_tr = K_tr * w[None, :]; K_co = K_co * w[None, :]
    ks = StandardScaler().fit(K_tr)
    nn = NearestNeighbors(n_neighbors=1).fit(ks.transform(K_tr))
    _, ni = nn.kneighbors(ks.transform(K_co))
    return idx_tr[ni[:, 0]]


def metrics(true_z, base_z, corrected, mean_arr, std_arr) -> dict:
    true_kw = true_z * std_arr[:, None] + mean_arr[:, None]
    base_kw = base_z * std_arr[:, None] + mean_arr[:, None]
    corr_kw = corrected * std_arr[:, None] + mean_arr[:, None]
    base_argmax = base_kw.argmax(axis=1)
    corr_argmax = corr_kw.argmax(axis=1)
    true_argmax = true_kw.argmax(axis=1)
    return {
        "base_pape": compute_pape(true_kw, base_kw),
        "corr_pape": compute_pape(true_kw, corr_kw),
        "base_hr@1": compute_hr(true_kw, base_kw, tol=1),
        "corr_hr@1": compute_hr(true_kw, corr_kw, tol=1),
        "base_hr@2": compute_hr(true_kw, base_kw, tol=2),
        "corr_hr@2": compute_hr(true_kw, corr_kw, tol=2),
        "base_argmax_mae": float(np.abs(true_argmax - base_argmax).mean()),
        "corr_argmax_mae": float(np.abs(true_argmax - corr_argmax).mean()),
    }


# ── V0: baseline (M=32, flat KEY) ──
def variant_baseline(tr, co, M=32, alpha=1.0, key_weights=None):
    vq = VectorQuantizerKMeans(num_embeddings=M, embedding_dim=tr["lat"].shape[1],
                                random_state=RANDOM_SEED)
    vq.fit(torch.from_numpy(tr["lat"]).float())
    cb = vq.codebook.cpu().numpy()
    offsets, counts, idx_tr = build_offsets(tr["lat"], tr["base_z"], tr["true_z"], cb)
    cold_cluster = cold_assign_via_key(tr["key"], co["key"], idx_tr, key_weights)
    corrected = co["base_z"] + alpha * offsets[cold_cluster]
    m = metrics(co["true_z"], co["base_z"], corrected, co["mean"], co["std"])
    m.update({"M": int(M), "alpha": alpha, "k_min": int(counts.min()),
              "k_max": int(counts.max()), "n_empty": int((counts == 0).sum())})
    return m, offsets, counts, cb


# ── V2: hour-stratified codebook ──
def variant_hour_stratified(tr, co, M_per_bin=8, alpha=1.0, n_bins=4):
    """Split train windows into n_bins by peak hour; fit M_per_bin codes per bin.
    Cold assignment: predict cold's hour bin from its KEY's argmax_norm * 24,
    then NN within that bin."""
    bin_edges = np.linspace(0, 24, n_bins + 1)
    train_peak_hr = tr["true_z"].argmax(axis=1)
    cold_pred_hr = (co["key"][:, 1] * 24).clip(0, 23.99)   # KEY[:,1] = argmax_norm

    all_codebooks = []
    all_idx_tr = np.full(len(tr["lat"]), -1, dtype=np.int64)
    all_offsets_list = []
    all_counts_list = []
    bin_to_global_offsets = []
    global_M = M_per_bin * n_bins

    for bi in range(n_bins):
        lo, hi = bin_edges[bi], bin_edges[bi + 1]
        bin_mask_tr = (train_peak_hr >= lo) & (train_peak_hr < hi)
        if bin_mask_tr.sum() < M_per_bin:
            print(f"    [hour bin {bi}] only {int(bin_mask_tr.sum())} samples — skipping")
            continue
        bin_lats = tr["lat"][bin_mask_tr]
        vq = VectorQuantizerKMeans(num_embeddings=M_per_bin, embedding_dim=bin_lats.shape[1],
                                    random_state=RANDOM_SEED + bi)
        diag = vq.fit(torch.from_numpy(bin_lats).float())
        cb = vq.codebook.cpu().numpy()
        local_idx = assign(bin_lats, cb)
        global_offsets = np.zeros((M_per_bin, 24), dtype=np.float32)
        local_counts = np.zeros(M_per_bin, dtype=np.int64)
        bin_indices = np.where(bin_mask_tr)[0]
        for c in range(M_per_bin):
            local_mask = local_idx == c
            local_counts[c] = int(local_mask.sum())
            if local_counts[c] > 0:
                idxs = bin_indices[local_mask]
                global_offsets[c] = (tr["true_z"][idxs] - tr["base_z"][idxs]).mean(axis=0)
        all_codebooks.append((bi, cb, global_offsets, local_counts))

    cold_cluster_offsets = np.zeros((len(co["lat"]), 24), dtype=np.float32)
    bin_count_used = np.zeros(n_bins, dtype=np.int64)
    for bi in range(n_bins):
        lo, hi = bin_edges[bi], bin_edges[bi + 1]
        cold_in_bin = (cold_pred_hr >= lo) & (cold_pred_hr < hi)
        if cold_in_bin.sum() == 0:
            continue
        match = next((x for x in all_codebooks if x[0] == bi), None)
        if match is None:
            continue
        _, cb_b, off_b, _ = match
        d = ((co["lat"][cold_in_bin][:, None, :] - cb_b[None, :, :]) ** 2).sum(axis=2)
        nearest = d.argmin(axis=1)
        cold_cluster_offsets[cold_in_bin] = off_b[nearest]
        bin_count_used[bi] = int(cold_in_bin.sum())

    corrected = co["base_z"] + alpha * cold_cluster_offsets
    m = metrics(co["true_z"], co["base_z"], corrected, co["mean"], co["std"])
    m.update({"M_per_bin": M_per_bin, "n_bins": n_bins, "alpha": alpha,
              "M_total": global_M, "cold_in_bins": bin_count_used.tolist()})
    return m, all_codebooks


# ── plotting ──
def plot_alpha(rows, fname):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["alpha"] for r in rows], [r["corr_pape"] / r["base_pape"] for r in rows],
            "o-", label="PAPE ratio", color="C0")
    ax2 = ax.twinx()
    ax2.plot([r["alpha"] for r in rows], [r["corr_hr@1"] for r in rows],
             "s--", label="HR@1 (KV)", color="C1")
    ax2.plot([r["alpha"] for r in rows], [r["base_hr@1"] for r in rows],
             "s:", label="HR@1 (base)", color="C1", alpha=0.5)
    ax.axhline(0.95, color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("PAPE ratio", color="C0")
    ax2.set_ylabel("HR@1 (%)", color="C1")
    ax.set_title("V1: alpha extended sweep on T2 (M=32)")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=120); plt.close(fig)


def plot_hour_distribution(tr, co, fname):
    tr_hr = tr["true_z"].argmax(axis=1)
    co_hr = co["true_z"].argmax(axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(25)
    ax.hist(tr_hr, bins=bins, alpha=0.6, label=f"train (n={len(tr_hr)})", color="C0", density=True)
    ax.hist(co_hr, bins=bins, alpha=0.6, label=f"cold  (n={len(co_hr)})", color="C2", density=True)
    ax.set_xlabel("peak hour of forecast horizon"); ax.set_ylabel("density")
    ax.set_title("True peak hour distribution (forecast 24h)")
    ax.set_xticks(np.arange(0, 25, 2))
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=120); plt.close(fig)


def plot_offset_shapes(offsets, counts, cb_codebook, fname, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    M = offsets.shape[0]
    valid = counts > 0
    cmap = plt.cm.viridis
    norms = np.array([np.abs(o).max() for o in offsets])
    for c in range(M):
        if not valid[c]:
            continue
        ax.plot(np.arange(24), offsets[c], color=cmap(c / M), alpha=0.6, lw=1.0)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("hour of forecast horizon")
    ax.set_ylabel("offset (z-units)")
    ax.set_title(f"{title}\noffset shape per cluster (color = cluster idx)")
    ax.grid(alpha=0.3); ax.set_xticks(np.arange(0, 25, 2))
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=120); plt.close(fig)


def main():
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    split = load_v10_split()
    print(f"[setup] T2 arm, train={len(split['train'])}, cold={len(split['cold'])}")
    tr = gather(split["train"]); co = gather(split["cold"])
    print(f"[data] train windows: {tr['lat'].shape[0]}, cold: {co['lat'].shape[0]}")

    plot_hour_distribution(tr, co, "hour_distribution.png")
    print(f"[diag] hour distribution saved")

    results = {"variants": {}, "alpha_sweep": [], "hour_distribution": {}}

    # V1: extended alpha sweep on M=32 baseline
    print("\n========== V1: α extended sweep (M=32 flat) ==========")
    alpha_rows = []
    for a in ALPHAS:
        m, off, cnt, cb = variant_baseline(tr, co, M=32, alpha=a)
        alpha_rows.append({**m})
        tag = "PASS" if m["corr_pape"] / m["base_pape"] <= 0.95 else "    "
        print(f"  α={a:.2f}  PAPE {m['corr_pape']:.2f}/{m['base_pape']:.2f}={m['corr_pape']/m['base_pape']:.3f}  "
              f"HR@1 {m['corr_hr@1']:.1f}/{m['base_hr@1']:.1f}  "
              f"argmax_MAE {m['corr_argmax_mae']:.2f}/{m['base_argmax_mae']:.2f}  {tag}")
    results["alpha_sweep"] = alpha_rows
    plot_alpha(alpha_rows, "V1_alpha_extended.png")

    # save baseline offsets at α=1.0 for later
    m_v0, off_v0, cnt_v0, cb_v0 = variant_baseline(tr, co, M=32, alpha=1.0)
    plot_offset_shapes(off_v0, cnt_v0, cb_v0, "offsets_V0_baseline_M32.png",
                       "V0 baseline (M=32 flat KMeans on h_generic)")
    results["variants"]["V0_M32_flat_alpha1.0"] = m_v0
    print(f"\n[V0] saved baseline offsets")

    # V2: hour-stratified
    print("\n========== V2: hour-stratified codebook ==========")
    for n_bins, M_pb in [(4, 8), (4, 16), (6, 6)]:
        m, _ = variant_hour_stratified(tr, co, M_per_bin=M_pb, alpha=1.0, n_bins=n_bins)
        ratio = m["corr_pape"] / m["base_pape"]
        tag = "PASS" if ratio <= 0.95 else "    "
        print(f"  bins={n_bins} M/bin={M_pb} (total={n_bins*M_pb})  "
              f"PAPE {m['corr_pape']:.2f}/{m['base_pape']:.2f}={ratio:.3f}  "
              f"HR@1 {m['corr_hr@1']:.1f}/{m['base_hr@1']:.1f}  "
              f"argmax_MAE {m['corr_argmax_mae']:.2f}  {tag}")
        results["variants"][f"V2_hourbins{n_bins}_Mpb{M_pb}"] = m

    # V3: weighted KEY-NN (argmax_norm at index 1)
    print("\n========== V3: weighted KEY-NN (boost argmax_norm) ==========")
    for w in [3.0, 5.0, 10.0, 20.0]:
        weights = np.array([1.0, w, 1.0, 1.0, 1.0])
        m, _, _, _ = variant_baseline(tr, co, M=32, alpha=1.0, key_weights=weights)
        ratio = m["corr_pape"] / m["base_pape"]
        tag = "PASS" if ratio <= 0.95 else "    "
        print(f"  argmax_w={w:5.1f}  "
              f"PAPE {m['corr_pape']:.2f}/{m['base_pape']:.2f}={ratio:.3f}  "
              f"HR@1 {m['corr_hr@1']:.1f}/{m['base_hr@1']:.1f}  "
              f"argmax_MAE {m['corr_argmax_mae']:.2f}  {tag}")
        results["variants"][f"V3_keyargmax_w{w}"] = m

    # V4: smaller M
    print("\n========== V4: smaller M (force broader clusters) ==========")
    for M in [4, 8, 16]:
        m, _, _, _ = variant_baseline(tr, co, M=M, alpha=1.0)
        ratio = m["corr_pape"] / m["base_pape"]
        tag = "PASS" if ratio <= 0.95 else "    "
        print(f"  M={M:3d}  k_min={m['k_min']}  "
              f"PAPE {m['corr_pape']:.2f}/{m['base_pape']:.2f}={ratio:.3f}  "
              f"HR@1 {m['corr_hr@1']:.1f}/{m['base_hr@1']:.1f}  "
              f"argmax_MAE {m['corr_argmax_mae']:.2f}  {tag}")
        results["variants"][f"V4_M{M}"] = m

    with open(ITER3 / "iter3_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    # combined summary table
    print("\n========== COMBINED SUMMARY ==========")
    print(f"{'variant':40s}  PAPE ratio  HR@1     argmax_MAE")
    print("-" * 80)
    print(f"{'baseline NBEATSx (no KV)':40s}  1.000        {m_v0['base_hr@1']:.1f}     {m_v0['base_argmax_mae']:.2f}")
    print(f"{'V0 M=32 flat α=1.0':40s}  "
          f"{m_v0['corr_pape']/m_v0['base_pape']:.3f}        {m_v0['corr_hr@1']:.1f}     {m_v0['corr_argmax_mae']:.2f}")
    for k, v in results["variants"].items():
        if k == "V0_M32_flat_alpha1.0":
            continue
        ratio = v["corr_pape"] / v["base_pape"]
        print(f"{k:40s}  {ratio:.3f}        {v['corr_hr@1']:.1f}     {v['corr_argmax_mae']:.2f}")

    print(f"\n[done] wrote {ITER3 / 'iter3_results.json'} + figures in {FIG}")


if __name__ == "__main__":
    main()
