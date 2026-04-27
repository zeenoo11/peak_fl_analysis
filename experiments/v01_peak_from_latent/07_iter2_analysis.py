"""iter2: 5-perspective deep dive on T2 (PASS) and T3 (near-miss) arms.

Single-pass extraction of all needed arrays per arm; then run all analyses on
top of those arrays.

Analyses:
    A. alpha sweep        — KV correction strength {0.0..1.5}, no refit
    B. M sweep            — codebook size {8, 16, 32, 64}, refit KMeans each
    C. per-household      — winners/losers in cold-PAPE delta
    D. cluster semantics  — what each cluster covers in (peak_amp, peak_hour)
    E. stronger baselines — global mean offset, stats2-based offset, KEY-only

Outputs JSON + 4 PNGs to outputs/v01_peak_from_latent/iter2/.
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
from models.nbeatsx import MinimalNBEATSx
from models.nbeatsx_aux import NBEATSxAux
from models.vq_kmeans import VectorQuantizerKMeans
from probes.peak_descriptor import extract_key
from utils.metrics import compute_pape

OUT = OUTPUT_DIR / "v01_peak_from_latent"
ITER2 = OUT / "iter2"
FIG = ITER2 / "figures"
ITER2.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARMS = ["T2", "T3"]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
M_VALUES = [8, 16, 32, 64]


def load_arm_model(arm: str):
    if arm == "T2":
        m = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
    elif arm == "T3":
        m = NBEATSxAux(latent_source="h_concat").to(DEVICE).eval()
    m.load_state_dict(torch.load(OUT / arm / "best.pt", map_location="cpu", weights_only=False))
    return m


def latent_for(arm: str, hidd: dict) -> torch.Tensor:
    if arm == "T2":
        return hidd["h_generic"]
    return torch.cat([hidd["h_trend"], hidd["h_seasonal"], hidd["h_generic"]], dim=1)


def gather(arm: str, apts: list[str]):
    """Collect everything needed: KEY, latent, baseline forecast, true forecast,
    per-apt mean/std, per-apt label."""
    m = load_arm_model(arm)
    keys, lats, base_z, true_z, m_arr, s_arr, apt_arr = [], [], [], [], [], [], []
    for ai, apt in enumerate(apts):
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
            lats.append(latent_for(arm, hidd).cpu().numpy())
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


def assign_clusters(lat: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    d = ((lat[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)


def offsets_from(lat_tr, base_tr, true_tr, codebook):
    idx_tr = assign_clusters(lat_tr, codebook)
    residuals = true_tr - base_tr
    M = codebook.shape[0]
    offsets = np.zeros((M, 24), dtype=np.float32)
    counts = np.zeros(M, dtype=np.int64)
    for c in range(M):
        mask = idx_tr == c
        counts[c] = int(mask.sum())
        if counts[c] > 0:
            offsets[c] = residuals[mask].mean(axis=0)
    return offsets, counts, idx_tr


def cold_assign_via_key(K_tr, K_co, idx_tr):
    ks = StandardScaler().fit(K_tr)
    nn = NearestNeighbors(n_neighbors=1).fit(ks.transform(K_tr))
    _, ni = nn.kneighbors(ks.transform(K_co))
    return idx_tr[ni[:, 0]]


def pape_from(true_z, base_z, mean_arr, std_arr, alpha, offsets, cold_cluster):
    corrected = base_z + alpha * offsets[cold_cluster]
    true_kw = true_z * std_arr[:, None] + mean_arr[:, None]
    base_kw = base_z * std_arr[:, None] + mean_arr[:, None]
    corr_kw = corrected * std_arr[:, None] + mean_arr[:, None]
    return {
        "base_pape": compute_pape(true_kw, base_kw),
        "corr_pape": compute_pape(true_kw, corr_kw),
    }


# ── Analysis A: α sweep ──────────────────────────────────────────────────
def analysis_alpha(tr, co, codebook):
    offsets, counts, idx_tr = offsets_from(tr["lat"], tr["base_z"], tr["true_z"], codebook)
    cold_cluster = cold_assign_via_key(tr["key"], co["key"], idx_tr)
    rows = []
    for a in ALPHAS:
        r = pape_from(co["true_z"], co["base_z"], co["mean"], co["std"], a, offsets, cold_cluster)
        rows.append({"alpha": a, "base_pape": r["base_pape"], "corr_pape": r["corr_pape"],
                     "ratio": r["corr_pape"] / r["base_pape"]})
    return rows


# ── Analysis B: M sweep ──────────────────────────────────────────────────
def analysis_M(tr, co, latent_dim):
    rows = []
    for M in M_VALUES:
        vq = VectorQuantizerKMeans(num_embeddings=M, embedding_dim=latent_dim,
                                    random_state=RANDOM_SEED)
        diag = vq.fit(torch.from_numpy(tr["lat"]).float())
        cb = vq.codebook.cpu().numpy()
        offsets, counts, idx_tr = offsets_from(tr["lat"], tr["base_z"], tr["true_z"], cb)
        cold_cluster = cold_assign_via_key(tr["key"], co["key"], idx_tr)
        r = pape_from(co["true_z"], co["base_z"], co["mean"], co["std"], 0.5,
                      offsets, cold_cluster)
        rows.append({"M": M, "k_min": diag["k_min"], "util": diag["utilization"],
                     "perplexity": diag["perplexity"],
                     "base_pape": r["base_pape"], "corr_pape": r["corr_pape"],
                     "ratio": r["corr_pape"] / r["base_pape"]})
    return rows


# ── Analysis C: per-household winners/losers ────────────────────────────
def analysis_per_apt(tr, co, codebook, alpha=0.5):
    offsets, _, idx_tr = offsets_from(tr["lat"], tr["base_z"], tr["true_z"], codebook)
    cold_cluster = cold_assign_via_key(tr["key"], co["key"], idx_tr)
    corrected = co["base_z"] + alpha * offsets[cold_cluster]
    true_kw = co["true_z"] * co["std"][:, None] + co["mean"][:, None]
    base_kw = co["base_z"] * co["std"][:, None] + co["mean"][:, None]
    corr_kw = corrected * co["std"][:, None] + co["mean"][:, None]
    rows = []
    for apt in np.unique(co["apt"]):
        mask = co["apt"] == apt
        rows.append({
            "apt": str(apt),
            "n_windows": int(mask.sum()),
            "base_pape": compute_pape(true_kw[mask], base_kw[mask]),
            "corr_pape": compute_pape(true_kw[mask], corr_kw[mask]),
        })
    for r in rows:
        r["delta_pape"] = r["corr_pape"] - r["base_pape"]
        r["rel_delta"] = r["delta_pape"] / r["base_pape"] if r["base_pape"] > 0 else 0.0
    return rows


# ── Analysis D: cluster semantics ────────────────────────────────────────
def analysis_clusters(tr, codebook):
    idx_tr = assign_clusters(tr["lat"], codebook)
    M = codebook.shape[0]
    rows = []
    for c in range(M):
        mask = idx_tr == c
        if mask.sum() == 0:
            rows.append({"c": c, "n": 0, "amp_mean": None, "hr_mean": None})
            continue
        peaks = tr["true_z"][mask].max(axis=1)
        hours = tr["true_z"][mask].argmax(axis=1)
        offset_norm = float(np.abs(tr["true_z"][mask] - tr["base_z"][mask]).mean())
        rows.append({
            "c": c,
            "n": int(mask.sum()),
            "amp_mean": float(peaks.mean()),
            "amp_std": float(peaks.std()),
            "hr_mean": float(hours.mean()),
            "hr_std": float(hours.std()),
            "offset_l1": offset_norm,
        })
    return rows


# ── Analysis E: stronger baselines ───────────────────────────────────────
def analysis_baselines(tr, co):
    """3 dumber baselines vs KV-VQ.
    B1: global mean residual offset (single 24-d vector for all cold windows)
    B2: stats2 cluster (KMeans on stats2 instead of latent)
    B3: KEY cluster (KMeans on KEY itself, no latent at all)
    """
    out = {}
    # B1 — global single offset
    global_offset = (tr["true_z"] - tr["base_z"]).mean(axis=0)   # [24]
    corrected = co["base_z"] + 0.5 * global_offset[None, :]
    true_kw = co["true_z"] * co["std"][:, None] + co["mean"][:, None]
    base_kw = co["base_z"] * co["std"][:, None] + co["mean"][:, None]
    corr_kw = corrected * co["std"][:, None] + co["mean"][:, None]
    out["B1_global_offset"] = {
        "base_pape": compute_pape(true_kw, base_kw),
        "corr_pape": compute_pape(true_kw, corr_kw),
    }
    # B2 — stats2 cluster
    stats_tr = np.stack([tr["base_z"].mean(axis=1), tr["base_z"].std(axis=1)], axis=1)
    stats_co = np.stack([co["base_z"].mean(axis=1), co["base_z"].std(axis=1)], axis=1)
    vq = VectorQuantizerKMeans(num_embeddings=32, embedding_dim=2, random_state=RANDOM_SEED)
    vq.fit(torch.from_numpy(stats_tr).float())
    cb = vq.codebook.cpu().numpy()
    offsets, _, idx_tr = offsets_from(stats_tr, tr["base_z"], tr["true_z"], cb)
    cold_cluster = cold_assign_via_key(tr["key"], co["key"], idx_tr)
    corrected = co["base_z"] + 0.5 * offsets[cold_cluster]
    corr_kw = corrected * co["std"][:, None] + co["mean"][:, None]
    out["B2_stats2_cluster"] = {
        "base_pape": compute_pape(true_kw, base_kw),
        "corr_pape": compute_pape(true_kw, corr_kw),
    }
    # B3 — KEY cluster (no latent at all)
    vq = VectorQuantizerKMeans(num_embeddings=32, embedding_dim=tr["key"].shape[1],
                                random_state=RANDOM_SEED)
    vq.fit(torch.from_numpy(tr["key"]).float())
    cb = vq.codebook.cpu().numpy()
    offsets, _, idx_tr = offsets_from(tr["key"], tr["base_z"], tr["true_z"], cb)
    cold_cluster = cold_assign_via_key(tr["key"], co["key"], idx_tr)
    corrected = co["base_z"] + 0.5 * offsets[cold_cluster]
    corr_kw = corrected * co["std"][:, None] + co["mean"][:, None]
    out["B3_key_cluster"] = {
        "base_pape": compute_pape(true_kw, base_kw),
        "corr_pape": compute_pape(true_kw, corr_kw),
    }
    for k in out:
        out[k]["ratio"] = out[k]["corr_pape"] / out[k]["base_pape"]
    return out


# ── plotting helpers ────────────────────────────────────────────────────
def plot_alpha(results: dict):
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, rows in results.items():
        ax.plot([r["alpha"] for r in rows], [r["ratio"] for r in rows],
                "o-", label=arm)
    ax.axhline(0.95, color="r", linestyle="--", label="H1c PASS line (0.95)")
    ax.axhline(1.0, color="gray", linestyle=":", label="no improvement")
    ax.set_xlabel(r"$\alpha$ (correction strength)")
    ax.set_ylabel("cold PAPE ratio (corr / base)")
    ax.set_title("A. alpha sweep — KV-VQ correction strength")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "A_alpha_sweep.png", dpi=120); plt.close(fig)


def plot_M(results: dict):
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, rows in results.items():
        ax.plot([r["M"] for r in rows], [r["ratio"] for r in rows], "o-", label=arm)
    ax.axhline(0.95, color="r", linestyle="--", label="H1c PASS line")
    ax.axhline(1.0, color="gray", linestyle=":")
    ax.set_xlabel("M (codebook size)"); ax.set_ylabel("cold PAPE ratio")
    ax.set_xscale("log", base=2); ax.set_xticks(M_VALUES); ax.set_xticklabels(M_VALUES)
    ax.set_title("B. M sweep — codebook size sensitivity")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "B_M_sweep.png", dpi=120); plt.close(fig)


def plot_per_apt(results: dict):
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (arm, rows) in zip(axes, results.items()):
        deltas = [r["rel_delta"] * 100 for r in rows]
        ax.hist(deltas, bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="r", linestyle="--", label="no change")
        ax.axvline(np.mean(deltas), color="green", linestyle="-",
                   label=f"mean={np.mean(deltas):.1f}%")
        ax.set_xlabel("relative ΔPAPE per apt (%)  [<0 means KV better]")
        ax.set_ylabel("# apts")
        ax.set_title(f"C. per-cold-apt KV vs base ({arm})")
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "C_per_apt.png", dpi=120); plt.close(fig)


def plot_cluster_semantics(results: dict):
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, (arm, rows) in zip(axes, results.items()):
        valid = [r for r in rows if r["amp_mean"] is not None]
        amps = [r["amp_mean"] for r in valid]
        hrs = [r["hr_mean"] for r in valid]
        sizes = np.array([r["n"] for r in valid])
        offsets = np.array([r["offset_l1"] for r in valid])
        sc = ax.scatter(hrs, amps, s=sizes / sizes.max() * 400 + 20,
                        c=offsets, cmap="viridis", alpha=0.7, edgecolors="k")
        cbar = plt.colorbar(sc, ax=ax); cbar.set_label("L1(true - base) z-units")
        ax.set_xlabel("mean peak hour (forecast)")
        ax.set_ylabel("mean peak amp (z-space)")
        ax.set_title(f"D. cluster semantics ({arm})  size = n_train_windows")
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "D_cluster_semantics.png", dpi=120); plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    split = load_v10_split()
    train_apts, cold_apts = split["train"], split["cold"]
    print(f"[setup] train={len(train_apts)}  cold={len(cold_apts)}")

    summary = {}
    a_results, b_results, c_results, d_results, e_results = {}, {}, {}, {}, {}

    for arm in ARMS:
        print(f"\n========== {arm} ==========")
        tr = gather(arm, train_apts); co = gather(arm, cold_apts)
        print(f"  train windows: {tr['lat'].shape[0]}  cold: {co['lat'].shape[0]}  "
              f"latent dim: {tr['lat'].shape[1]}")

        cb = np.load(OUT / arm / "codebook.npz")["codebook"]

        print("  [A] alpha sweep ...")
        a_results[arm] = analysis_alpha(tr, co, cb)
        for r in a_results[arm]:
            tag = "PASS" if r["ratio"] <= 0.95 else "    "
            print(f"    α={r['alpha']:.2f}  base={r['base_pape']:.2f}  "
                  f"corr={r['corr_pape']:.2f}  ratio={r['ratio']:.3f}  {tag}")

        print("  [B] M sweep ...")
        b_results[arm] = analysis_M(tr, co, latent_dim=tr['lat'].shape[1])
        for r in b_results[arm]:
            tag = "PASS" if r["ratio"] <= 0.95 else "    "
            print(f"    M={r['M']:3d}  k_min={r['k_min']:5d}  ppl={r['perplexity']:.2f}  "
                  f"base={r['base_pape']:.2f}  corr={r['corr_pape']:.2f}  ratio={r['ratio']:.3f}  {tag}")

        print("  [C] per-apt analysis ...")
        c_results[arm] = analysis_per_apt(tr, co, cb)
        deltas = np.array([r["rel_delta"] * 100 for r in c_results[arm]])
        winners = sum(1 for d in deltas if d < -1.0)
        losers = sum(1 for d in deltas if d > 1.0)
        print(f"    {winners} apts improved >1%, {losers} apts degraded >1%, "
              f"{len(deltas) - winners - losers} flat. mean Δ={deltas.mean():+.2f}%")
        sorted_apts = sorted(c_results[arm], key=lambda r: r["delta_pape"])
        print(f"    top-3 winners: " + ", ".join(f"{r['apt']}({r['delta_pape']:+.1f})" for r in sorted_apts[:3]))
        print(f"    top-3 losers : " + ", ".join(f"{r['apt']}({r['delta_pape']:+.1f})" for r in sorted_apts[-3:]))

        print("  [D] cluster semantics ...")
        d_results[arm] = analysis_clusters(tr, cb)
        valid = [r for r in d_results[arm] if r["amp_mean"] is not None]
        print(f"    M={len(d_results[arm])} clusters, {len(valid)} non-empty")
        amp_range = (min(r["amp_mean"] for r in valid), max(r["amp_mean"] for r in valid))
        hr_range = (min(r["hr_mean"] for r in valid), max(r["hr_mean"] for r in valid))
        print(f"    cluster amp_mean range: [{amp_range[0]:.2f}, {amp_range[1]:.2f}]")
        print(f"    cluster hr_mean range : [{hr_range[0]:.1f}, {hr_range[1]:.1f}]")

        print("  [E] stronger baselines ...")
        e_results[arm] = analysis_baselines(tr, co)
        kv_pape = a_results[arm][2]["corr_pape"]   # alpha=0.5
        for name, m in e_results[arm].items():
            tag = "BEAT" if m["corr_pape"] < kv_pape else "    "
            print(f"    {name}  base={m['base_pape']:.2f}  corr={m['corr_pape']:.2f}  "
                  f"ratio={m['ratio']:.3f}  vs KV-VQ {kv_pape:.2f}  {tag}")

        summary[arm] = {
            "alpha_sweep": a_results[arm], "M_sweep": b_results[arm],
            "per_apt": c_results[arm], "clusters": d_results[arm],
            "baselines": e_results[arm],
        }

    plot_alpha(a_results)
    plot_M(b_results)
    plot_per_apt(c_results)
    plot_cluster_semantics(d_results)

    with open(ITER2 / "iter2_results.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[done] wrote {ITER2 / 'iter2_results.json'} + 4 PNGs in {FIG}")


if __name__ == "__main__":
    main()
