"""E1: peak_aux ON/OFF clean ablation.

Holds mechanism (V0 cluster offset, W5 hybrid) constant; varies only the
backbone training:
    T0  = MinimalNBEATSx, MAE only          (no peak_aux)
    T2  = NBEATSxAux,    MAE + peak_aux     (with peak_aux)

For W5 hybrid we need (pred_amp, pred_hr). T2 uses its aux head; T0 has none,
so we use the model's own forecast (base_z.max / base_z.argmax) as a proxy
("self-derived aux"). This makes the comparison fair.

Output: outputs/v01_peak_from_latent/E1/E1_results.json + ablation table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
from utils.metrics import compute_hr, compute_pape

OUT = OUTPUT_DIR / "v01_peak_from_latent"
E1 = OUT / "E1"
E1.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W5_BEST = {"sigma": 3.0, "alpha_v0": 1.5, "alpha_w1": 0.5}


def gather(arm: str, apts: list[str]):
    """Gather (KEY, latent, base_z, true_z, pred_amp, pred_hr, mean, std)."""
    if arm == "T0":
        m = MinimalNBEATSx().to(DEVICE).eval()
        m.load_state_dict(torch.load(OUT / "T0" / "best.pt", map_location="cpu", weights_only=False))
        is_aux = False
    elif arm == "T2":
        m = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
        m.load_state_dict(torch.load(OUT / "T2" / "best.pt", map_location="cpu", weights_only=False))
        is_aux = True
    else:
        raise ValueError(arm)

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
                if is_aux:
                    y_hat, hidd, (amp_p, hr_p) = m(x.to(DEVICE))
                    p_amp.append(amp_p.cpu().numpy())
                    p_hr.append(hr_p.argmax(dim=1).cpu().numpy())
                else:
                    y_hat, hidd = m(x.to(DEVICE))
                    # Self-derived "aux" from forecast itself
                    p_amp.append(y_hat.max(dim=1).values.cpu().numpy())
                    p_hr.append(y_hat.argmax(dim=1).cpu().numpy())
            lats.append(hidd["h_generic"].cpu().numpy())
            base_z.append(y_hat.cpu().numpy())
            true_z.append(y.numpy())
            m_arr.append(np.full(len(y), mean)); s_arr.append(np.full(len(y), std))
    return {
        "key": np.concatenate(keys, 0), "lat": np.concatenate(lats, 0),
        "base_z": np.concatenate(base_z, 0), "true_z": np.concatenate(true_z, 0),
        "pred_amp": np.concatenate(p_amp, 0), "pred_hr": np.concatenate(p_hr, 0),
        "mean": np.concatenate(m_arr, 0), "std": np.concatenate(s_arr, 0),
    }


def fit_vq_and_offsets(tr):
    vq = VectorQuantizerKMeans(num_embeddings=32, embedding_dim=tr["lat"].shape[1],
                                random_state=RANDOM_SEED)
    diag = vq.fit(torch.from_numpy(tr["lat"]).float())
    cb = vq.codebook.cpu().numpy()
    d = ((tr["lat"][:, None, :] - cb[None, :, :]) ** 2).sum(axis=2)
    idx_tr = d.argmin(axis=1)
    M = cb.shape[0]
    offsets = np.zeros((M, 24), dtype=np.float32)
    for c in range(M):
        mask = idx_tr == c
        if mask.sum() > 0:
            offsets[c] = (tr["true_z"][mask] - tr["base_z"][mask]).mean(axis=0)
    return cb, offsets, idx_tr, diag


def cold_assign(tr_key, co_key, idx_tr):
    ks = StandardScaler().fit(tr_key)
    nn = NearestNeighbors(n_neighbors=1).fit(ks.transform(tr_key))
    _, ni = nn.kneighbors(ks.transform(co_key))
    return idx_tr[ni[:, 0]]


def metrics(true_z, base_z, corrected, mean_arr, std_arr):
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
    }


def eval_arm(arm: str, train_apts, cold_apts):
    print(f"\n========== {arm} ==========")
    tr = gather(arm, train_apts)
    co = gather(arm, cold_apts)
    cb, offsets, idx_tr, diag = fit_vq_and_offsets(tr)
    print(f"  vq diag: util={diag['utilization']:.3f}  ppl={diag['perplexity']:.2f}  k_min={diag['k_min']}")
    cold_cluster = cold_assign(tr["key"], co["key"], idx_tr)

    # V0 (cluster offset only, α=2.0)
    v0_corrected = co["base_z"] + 2.0 * offsets[cold_cluster]
    v0_m = metrics(co["true_z"], co["base_z"], v0_corrected, co["mean"], co["std"])

    # W5 hybrid (V0 offset + sharp Gaussian template)
    sigma, av, aw = W5_BEST["sigma"], W5_BEST["alpha_v0"], W5_BEST["alpha_w1"]
    t = np.arange(24)[None, :]
    g = np.exp(-0.5 * ((t - co["pred_hr"][:, None]) / sigma) ** 2)
    g = g / g.max(axis=1, keepdims=True) * co["pred_amp"][:, None]
    w5_corrected = co["base_z"] + av * offsets[cold_cluster] + aw * g
    w5_m = metrics(co["true_z"], co["base_z"], w5_corrected, co["mean"], co["std"])

    return {"vq_diag": diag, "V0": v0_m, "W5": w5_m,
            "n_train_windows": int(tr["lat"].shape[0]),
            "n_cold_windows": int(co["lat"].shape[0])}


def main():
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    split = load_v10_split()
    print(f"[setup] train={len(split['train'])}, cold={len(split['cold'])}")

    results = {}
    for arm in ["T0", "T2"]:
        results[arm] = eval_arm(arm, split["train"], split["cold"])

    print("\n========== E1 ABLATION TABLE ==========")
    print(f"{'metric':25s}  T0 (no peak_aux)  T2 (peak_aux)  Δ (peak_aux 효과)")
    print("-" * 90)
    for mech in ["V0", "W5"]:
        for k in ["base_pape", "corr_pape", "base_hr@1", "corr_hr@1", "base_hr@2", "corr_hr@2"]:
            t0_v = results["T0"][mech][k]
            t2_v = results["T2"][mech][k]
            delta = t2_v - t0_v
            arrow = "↓" if "pape" in k and delta < 0 else ("↑" if "hr" in k and delta > 0 else "")
            print(f"  {mech} {k:18s}  {t0_v:8.2f}            {t2_v:8.2f}     {delta:+7.2f} {arrow}")
        print()

    # Compute relative cold improvement
    print("Relative cold improvements (corr/base):")
    for arm in ["T0", "T2"]:
        for mech in ["V0", "W5"]:
            r = results[arm][mech]
            ratio = r["corr_pape"] / r["base_pape"]
            print(f"  {arm} {mech}: cold PAPE {r['corr_pape']:.2f}/{r['base_pape']:.2f} = {ratio:.3f} "
                  f"({(1-ratio)*100:+.1f}%)")

    with open(E1 / "E1_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[done] wrote {E1 / 'E1_results.json'}")


if __name__ == "__main__":
    main()
