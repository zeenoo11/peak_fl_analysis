"""iter4: 6 mechanisms for hour-aware cold-start correction.

Single shared extraction (T2 model + aux head outputs); then variants:

    V0   baseline KV-VQ (M=32 flat, α=2.0)        — best of iter3
    W1a  sharp-Gaussian additive boost            — corrected = base + α·gauss
    W1b  sharp-Gaussian blend                      — corrected = (1-α)·base + α·template
    W3   2D codebook (cluster × hour_bin)         — offsets differentiated by both
    W4   K-NN in latent (K=10, inv-dist weighted) — sharpness recovery
    W5   Hybrid V0+W1                              — amp from V0, hour from W1
    W6   KEY extended with hour-of-day             — KEY[5] = argmax % 24

Each variant: PAPE, HR@1, HR@2, argmax_MAE.
Output: Pareto plot (PAPE × HR@1) + JSON.
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
ITER4 = OUT / "iter4"
FIG = ITER4 / "figures"
ITER4.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_key_extended(x_np: np.ndarray) -> np.ndarray:
    """6-d KEY: original 5 + argmax_hour_of_day."""
    if x_np.ndim == 1:
        x_np = x_np[None, :]
    return np.stack([
        x_np.max(axis=1),
        x_np.argmax(axis=1) / 96.0,
        x_np.mean(axis=1),
        x_np.std(axis=1),
        x_np[:, -24:].max(axis=1),
        (x_np.argmax(axis=1) % 24) / 23.0,    # hour-of-day, normalized
    ], axis=1)


def gather(apts, key_extended=False):
    m = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
    m.load_state_dict(torch.load(OUT / "T2" / "best.pt", map_location="cpu", weights_only=False))
    keys, lats, base_z, true_z = [], [], [], []
    pred_amp_list, pred_hr_list = [], []
    m_arr, s_arr, apt_arr = [], [], []
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
            x_np = x.numpy()
            if key_extended:
                keys.append(extract_key_extended(x_np))
            else:
                keys.append(extract_key(x_np))
            with torch.no_grad():
                y_hat, hidd, (amp_p, hr_p) = m(x.to(DEVICE))
            lats.append(hidd["h_generic"].cpu().numpy())
            base_z.append(y_hat.cpu().numpy())
            true_z.append(y.numpy())
            pred_amp_list.append(amp_p.cpu().numpy())
            pred_hr_list.append(hr_p.argmax(dim=1).cpu().numpy())
            m_arr.append(np.full(len(y), mean)); s_arr.append(np.full(len(y), std))
            apt_arr.append([apt] * len(y))
    return {
        "key": np.concatenate(keys, axis=0),
        "lat": np.concatenate(lats, axis=0),
        "base_z": np.concatenate(base_z, axis=0),
        "true_z": np.concatenate(true_z, axis=0),
        "pred_amp": np.concatenate(pred_amp_list, axis=0),
        "pred_hr": np.concatenate(pred_hr_list, axis=0),
        "mean": np.concatenate(m_arr, axis=0),
        "std": np.concatenate(s_arr, axis=0),
        "apt": np.array([a for chunk in apt_arr for a in chunk]),
    }


def metrics(true_z, base_z, corrected, mean_arr, std_arr) -> dict:
    true_kw = true_z * std_arr[:, None] + mean_arr[:, None]
    base_kw = base_z * std_arr[:, None] + mean_arr[:, None]
    corr_kw = corrected * std_arr[:, None] + mean_arr[:, None]
    return {
        "base_pape": compute_pape(true_kw, base_kw),
        "corr_pape": compute_pape(true_kw, corr_kw),
        "base_hr@1": compute_hr(true_kw, base_kw, tol=1),
        "corr_hr@1": compute_hr(true_kw, corr_kw, tol=1),
        "base_hr@2": compute_hr(true_kw, base_kw, tol=2),
        "corr_hr@2": compute_hr(true_kw, corr_kw, tol=2),
        "base_argmax_mae": float(np.abs(true_kw.argmax(1) - base_kw.argmax(1)).mean()),
        "corr_argmax_mae": float(np.abs(true_kw.argmax(1) - corr_kw.argmax(1)).mean()),
    }


def assign(lat, codebook):
    d = ((lat[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)


def cold_assign_via_key(K_tr, K_co, idx_tr):
    ks = StandardScaler().fit(K_tr)
    nn = NearestNeighbors(n_neighbors=1).fit(ks.transform(K_tr))
    _, ni = nn.kneighbors(ks.transform(K_co))
    return idx_tr[ni[:, 0]]


# ── V0 baseline ──
def run_V0(tr, co, M=32, alpha=2.0):
    vq = VectorQuantizerKMeans(num_embeddings=M, embedding_dim=tr["lat"].shape[1],
                                random_state=RANDOM_SEED)
    vq.fit(torch.from_numpy(tr["lat"]).float())
    cb = vq.codebook.cpu().numpy()
    idx_tr = assign(tr["lat"], cb)
    M = cb.shape[0]
    offsets = np.zeros((M, 24), dtype=np.float32)
    for c in range(M):
        mask = idx_tr == c
        if mask.sum() > 0:
            offsets[c] = (tr["true_z"][mask] - tr["base_z"][mask]).mean(axis=0)
    cold_cluster = cold_assign_via_key(tr["key"], co["key"], idx_tr)
    corrected = co["base_z"] + alpha * offsets[cold_cluster]
    return metrics(co["true_z"], co["base_z"], corrected, co["mean"], co["std"]), idx_tr, cb, offsets


# ── W1: sharp Gaussian from aux head ──
def gauss_template(pred_hr, pred_amp, sigma=1.5, length=24):
    t = np.arange(length)[None, :]
    g = np.exp(-0.5 * ((t - pred_hr[:, None]) / sigma) ** 2)
    g = g / g.max(axis=1, keepdims=True)
    return g * pred_amp[:, None]


def run_W1a(co, alpha, sigma=1.5):
    """Additive: corrected = base + α * gauss(pred_hr, pred_amp)."""
    g = gauss_template(co["pred_hr"], co["pred_amp"], sigma=sigma)
    corrected = co["base_z"] + alpha * g
    return metrics(co["true_z"], co["base_z"], corrected, co["mean"], co["std"])


def run_W1b(co, alpha, sigma=1.5):
    """Blend: corrected = (1-α)·base + α·template (template = gauss)."""
    g = gauss_template(co["pred_hr"], co["pred_amp"], sigma=sigma)
    corrected = (1 - alpha) * co["base_z"] + alpha * g
    return metrics(co["true_z"], co["base_z"], corrected, co["mean"], co["std"])


# ── W3: 2D codebook (cluster × hour_bin) ──
def run_W3(tr, co, M=32, n_hbin=4, alpha=2.0):
    vq = VectorQuantizerKMeans(num_embeddings=M, embedding_dim=tr["lat"].shape[1],
                                random_state=RANDOM_SEED)
    vq.fit(torch.from_numpy(tr["lat"]).float())
    cb = vq.codebook.cpu().numpy()
    idx_tr = assign(tr["lat"], cb)
    train_hr = tr["true_z"].argmax(axis=1)
    bin_edges = np.linspace(0, 24, n_hbin + 1)
    train_hbin = np.searchsorted(bin_edges, train_hr, side="right") - 1
    train_hbin = train_hbin.clip(0, n_hbin - 1)

    offsets = np.zeros((M, n_hbin, 24), dtype=np.float32)
    counts = np.zeros((M, n_hbin), dtype=np.int64)
    for c in range(M):
        for h in range(n_hbin):
            mask = (idx_tr == c) & (train_hbin == h)
            counts[c, h] = int(mask.sum())
            if counts[c, h] > 0:
                offsets[c, h] = (tr["true_z"][mask] - tr["base_z"][mask]).mean(axis=0)

    cold_cluster = cold_assign_via_key(tr["key"], co["key"], idx_tr)
    cold_hbin = (np.searchsorted(bin_edges, co["pred_hr"], side="right") - 1).clip(0, n_hbin - 1)
    chosen_offsets = offsets[cold_cluster, cold_hbin]   # [N_co, 24]
    # fallback: if counts[c, h] == 0, use cluster-marginal mean
    no_data = counts[cold_cluster, cold_hbin] == 0
    if no_data.any():
        cluster_marginal = offsets.mean(axis=1)   # [M, 24]
        chosen_offsets[no_data] = cluster_marginal[cold_cluster[no_data]]
    corrected = co["base_z"] + alpha * chosen_offsets
    res = metrics(co["true_z"], co["base_z"], corrected, co["mean"], co["std"])
    res["empty_cells"] = int((counts == 0).sum())
    res["n_fallback"] = int(no_data.sum())
    return res


# ── W4: K-NN in latent ──
def run_W4(tr, co, K=10, alpha=2.0):
    nn = NearestNeighbors(n_neighbors=K).fit(tr["lat"])
    dists, idxs = nn.kneighbors(co["lat"])     # [N_co, K]
    # inverse distance weighting (eps for stability)
    w = 1.0 / (dists + 1e-6)
    w = w / w.sum(axis=1, keepdims=True)        # [N_co, K]
    train_resid = tr["true_z"] - tr["base_z"]   # [N_tr, 24]
    cold_offset = (w[:, :, None] * train_resid[idxs]).sum(axis=1)
    corrected = co["base_z"] + alpha * cold_offset
    res = metrics(co["true_z"], co["base_z"], corrected, co["mean"], co["std"])
    res["K"] = K
    return res


# ── W5: Hybrid V0 + W1 ──
def run_W5(tr, co, alpha_v0=2.0, alpha_w1=0.3, sigma=1.5):
    """V0 amp offset added; on top, blend in W1's hour-shaped boost."""
    _, idx_tr, cb, offsets_v0 = run_V0(tr, co, M=32, alpha=alpha_v0)   # discard metrics
    cold_cluster = cold_assign_via_key(tr["key"], co["key"], idx_tr)
    g = gauss_template(co["pred_hr"], co["pred_amp"], sigma=sigma)
    corrected = co["base_z"] + alpha_v0 * offsets_v0[cold_cluster] + alpha_w1 * g
    return metrics(co["true_z"], co["base_z"], corrected, co["mean"], co["std"])


# ── W6: extended KEY ──
def run_W6(apts_tr, apts_co, alpha=2.0):
    """V0 logic but with 6-d KEY (adds hour-of-day)."""
    tr6 = gather(apts_tr, key_extended=True)
    co6 = gather(apts_co, key_extended=True)
    return run_V0(tr6, co6, M=32, alpha=alpha)[0]


# ── plotting ──
def plot_pareto(rows: list[dict], fname: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in rows:
        marker = "o" if r["name"].startswith("V0") else "s"
        ax.scatter(r["corr_pape"], r["corr_hr@1"], s=120, label=r["name"],
                   marker=marker, alpha=0.85)
        ax.annotate(r["name"], (r["corr_pape"], r["corr_hr@1"]),
                    fontsize=8, xytext=(5, 5), textcoords="offset points")
    base_pape = rows[0]["base_pape"]; base_hr = rows[0]["base_hr@1"]
    ax.scatter(base_pape, base_hr, s=200, c="red", marker="*", label="NBEATSx (no KV)", zorder=5)
    ax.annotate("NBEATSx", (base_pape, base_hr), fontsize=9, xytext=(5, 5),
                textcoords="offset points", color="red", fontweight="bold")
    ax.set_xlabel("cold PAPE (kW)  ← lower is better")
    ax.set_ylabel("cold HR@1 (%)  ↑ higher is better")
    ax.set_title("iter4 Pareto: cold PAPE vs HR@1 across 6 mechanisms")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=120); plt.close(fig)


def main():
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    split = load_v10_split()
    print(f"[setup] T2 arm; train={len(split['train'])}, cold={len(split['cold'])}")

    tr = gather(split["train"]); co = gather(split["cold"])
    print(f"[data] train={tr['lat'].shape[0]} windows, cold={co['lat'].shape[0]} windows")

    # diagnostic: aux head's own hour prediction accuracy on cold data
    cold_true_hr = co["true_z"].argmax(axis=1)
    aux_acc = float((co["pred_hr"] == cold_true_hr).mean())
    aux_within1 = float((np.abs(co["pred_hr"] - cold_true_hr) <= 1).mean())
    aux_within2 = float((np.abs(co["pred_hr"] - cold_true_hr) <= 2).mean())
    print(f"[diag] aux head on cold: top-1={aux_acc*100:.1f}%  within-1h={aux_within1*100:.1f}%  within-2h={aux_within2*100:.1f}%")

    rows = []

    # V0 baseline (α=2.0)
    print("\n========== V0: baseline KV (M=32, α=2.0) ==========")
    m_v0, _, _, _ = run_V0(tr, co, alpha=2.0)
    print(f"  PAPE {m_v0['corr_pape']:.2f}/{m_v0['base_pape']:.2f}={m_v0['corr_pape']/m_v0['base_pape']:.3f}  "
          f"HR@1 {m_v0['corr_hr@1']:.1f}/{m_v0['base_hr@1']:.1f}  argmax_MAE {m_v0['corr_argmax_mae']:.2f}")
    rows.append({**m_v0, "name": "V0 (M=32 α=2.0)"})

    # W1a additive
    print("\n========== W1a: sharp Gaussian additive ==========")
    for a in [0.3, 0.5, 1.0, 1.5]:
        m = run_W1a(co, alpha=a, sigma=1.5)
        print(f"  α={a}  PAPE {m['corr_pape']:.2f}/{m['base_pape']:.2f}={m['corr_pape']/m['base_pape']:.3f}  "
              f"HR@1 {m['corr_hr@1']:.1f}  argmax_MAE {m['corr_argmax_mae']:.2f}")
        rows.append({**m, "name": f"W1a α={a}"})

    # W1b blend
    print("\n========== W1b: sharp Gaussian blend ==========")
    for a in [0.1, 0.3, 0.5]:
        m = run_W1b(co, alpha=a, sigma=1.5)
        print(f"  α={a}  PAPE {m['corr_pape']:.2f}/{m['base_pape']:.2f}={m['corr_pape']/m['base_pape']:.3f}  "
              f"HR@1 {m['corr_hr@1']:.1f}  argmax_MAE {m['corr_argmax_mae']:.2f}")
        rows.append({**m, "name": f"W1b α={a}"})

    # W3 2D codebook
    print("\n========== W3: 2D codebook (cluster × hour_bin) ==========")
    for n_hbin in [4, 6]:
        m = run_W3(tr, co, M=32, n_hbin=n_hbin, alpha=2.0)
        print(f"  bins={n_hbin}  empty_cells={m['empty_cells']}  fallback={m['n_fallback']}  "
              f"PAPE {m['corr_pape']:.2f}/{m['base_pape']:.2f}={m['corr_pape']/m['base_pape']:.3f}  "
              f"HR@1 {m['corr_hr@1']:.1f}  argmax_MAE {m['corr_argmax_mae']:.2f}")
        rows.append({**m, "name": f"W3 bins={n_hbin}"})

    # W4 K-NN in latent
    print("\n========== W4: K-NN in latent ==========")
    for K in [5, 10, 50]:
        m = run_W4(tr, co, K=K, alpha=2.0)
        print(f"  K={K}  PAPE {m['corr_pape']:.2f}/{m['base_pape']:.2f}={m['corr_pape']/m['base_pape']:.3f}  "
              f"HR@1 {m['corr_hr@1']:.1f}  argmax_MAE {m['corr_argmax_mae']:.2f}")
        rows.append({**m, "name": f"W4 K={K}"})

    # W5 Hybrid V0 + W1
    print("\n========== W5: Hybrid V0+W1 ==========")
    for a_w1 in [0.1, 0.3, 0.5]:
        m = run_W5(tr, co, alpha_v0=2.0, alpha_w1=a_w1, sigma=1.5)
        print(f"  α_w1={a_w1}  PAPE {m['corr_pape']:.2f}/{m['base_pape']:.2f}={m['corr_pape']/m['base_pape']:.3f}  "
              f"HR@1 {m['corr_hr@1']:.1f}  argmax_MAE {m['corr_argmax_mae']:.2f}")
        rows.append({**m, "name": f"W5 α_w1={a_w1}"})

    # W6 extended KEY
    print("\n========== W6: extended KEY (+hour_of_day) ==========")
    m = run_W6(split["train"], split["cold"], alpha=2.0)
    print(f"  PAPE {m['corr_pape']:.2f}/{m['base_pape']:.2f}={m['corr_pape']/m['base_pape']:.3f}  "
          f"HR@1 {m['corr_hr@1']:.1f}  argmax_MAE {m['corr_argmax_mae']:.2f}")
    rows.append({**m, "name": "W6 ext KEY"})

    plot_pareto(rows, "iter4_pareto.png")

    # Pareto-optimal selection
    print("\n========== Pareto-optimal variants ==========")
    print(f"  {'name':30s}  PAPE      HR@1   argmax_MAE")
    base_pape = rows[0]["base_pape"]; base_hr = rows[0]["base_hr@1"]
    print(f"  {'NBEATSx baseline':30s}  {base_pape:.2f}    {base_hr:.1f}    {rows[0]['base_argmax_mae']:.2f}")
    for r in rows:
        # Pareto-optimal: no other row weakly dominates (≤ PAPE AND ≥ HR@1, with ≥1 strict)
        dominated = any(
            (other["corr_pape"] <= r["corr_pape"] and other["corr_hr@1"] > r["corr_hr@1"])
            or (other["corr_pape"] < r["corr_pape"] and other["corr_hr@1"] >= r["corr_hr@1"])
            for other in rows if other["name"] != r["name"]
        )
        marker = "*" if not dominated else " "
        print(f"  {marker} {r['name']:28s}  {r['corr_pape']:.2f}    {r['corr_hr@1']:.1f}    {r['corr_argmax_mae']:.2f}")

    with open(ITER4 / "iter4_results.json", "w") as fh:
        json.dump({"rows": rows, "aux_diag": {
            "top1": aux_acc, "within1": aux_within1, "within2": aux_within2
        }}, fh, indent=2)
    print(f"\n[done] wrote {ITER4 / 'iter4_results.json'} + Pareto plot")


if __name__ == "__main__":
    main()
