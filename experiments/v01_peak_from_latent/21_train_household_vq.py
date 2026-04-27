"""Train-household VQ check: PAPE change on TRAIN apts when V0 / W5 corrections
are applied to the same data the cluster offsets were learned from.

Complements E1 (cold-apt evaluation):
    - E1 measures cold-start transfer (KEY-NN cluster assignment).
    - This script measures in-distribution effect (latent-NN assignment, train apts).

For each arm in {T0, T2}:
    1. Gather train apts data (50 apts, stride=24, series[:train_end]).
    2. Fit M=32 KMeans on train latents -> codebook C, per-cluster mean
       residual offsets o_c.
    3. Two assignment protocols:
         (a) latent-NN -> direct in-distribution
         (b) KEY-NN    -> matches the cold-start protocol for fairness
    4. Apply V0 (corrected = base + 2.0 * o_{c}) and W5 hybrid (V0 + Gaussian
       template from aux_head's predictions; T0 uses self-derived peak).
    5. Report base / corrected PAPE, HR@1, HR@2.
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
TRAIN_VQ = OUT / "train_vq_check"
TRAIN_VQ.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W5_BEST = {"sigma": 3.0, "alpha_v0": 1.5, "alpha_w1": 0.5}


def gather(arm: str, apts: list[str]):
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


def keynn_self_assign(tr_key, idx_tr):
    """Re-assign each train window to a cluster via 1-NN in KEY space among
    OTHER train windows (leave-one-out). Mirrors the cold-start protocol
    where assignment is forced through the KEY descriptor."""
    ks = StandardScaler().fit(tr_key)
    Kn = ks.transform(tr_key)
    nn = NearestNeighbors(n_neighbors=2).fit(Kn)
    _, ni = nn.kneighbors(Kn)
    # nearest neighbor is self; use second-nearest
    return idx_tr[ni[:, 1]]


def eval_arm(arm: str, train_apts):
    print(f"\n========== {arm} ==========")
    tr = gather(arm, train_apts)
    cb, offsets, idx_tr, diag = fit_vq_and_offsets(tr)
    print(f"  vq diag: util={diag['utilization']:.3f}  ppl={diag['perplexity']:.2f}  k_min={diag['k_min']}")
    print(f"  windows: {tr['lat'].shape[0]}")

    # ── Protocol (a): latent-NN (in-distribution direct) ──
    v0_lat = tr["base_z"] + 2.0 * offsets[idx_tr]
    v0_lat_m = metrics(tr["true_z"], tr["base_z"], v0_lat, tr["mean"], tr["std"])

    sigma, av, aw = W5_BEST["sigma"], W5_BEST["alpha_v0"], W5_BEST["alpha_w1"]
    t = np.arange(24)[None, :]
    g = np.exp(-0.5 * ((t - tr["pred_hr"][:, None]) / sigma) ** 2)
    g = g / g.max(axis=1, keepdims=True) * tr["pred_amp"][:, None]
    w5_lat = tr["base_z"] + av * offsets[idx_tr] + aw * g
    w5_lat_m = metrics(tr["true_z"], tr["base_z"], w5_lat, tr["mean"], tr["std"])

    # ── Protocol (b): KEY-NN (cold-protocol mirror via leave-one-out) ──
    idx_key = keynn_self_assign(tr["key"], idx_tr)
    v0_key = tr["base_z"] + 2.0 * offsets[idx_key]
    v0_key_m = metrics(tr["true_z"], tr["base_z"], v0_key, tr["mean"], tr["std"])
    w5_key = tr["base_z"] + av * offsets[idx_key] + aw * g
    w5_key_m = metrics(tr["true_z"], tr["base_z"], w5_key, tr["mean"], tr["std"])

    # consistency between protocols
    same_cluster = float((idx_tr == idx_key).mean())

    return {
        "vq_diag": diag,
        "n_train_windows": int(tr["lat"].shape[0]),
        "latent_nn": {"V0": v0_lat_m, "W5": w5_lat_m},
        "key_nn_loo": {"V0": v0_key_m, "W5": w5_key_m},
        "key_eq_latent_assign_rate": same_cluster,
    }


def main():
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    split = load_v10_split()
    print(f"[setup] train_apts={len(split['train'])}; W5_BEST={W5_BEST}")

    results = {}
    for arm in ["T0", "T2"]:
        results[arm] = eval_arm(arm, split["train"])

    # Print summary table comparable to E1
    print("\n========== TRAIN-HOUSEHOLD VQ TABLE ==========")
    print(f"{'arm':5s}  {'mech':4s}  {'assign':10s}  {'base_PAPE':>10s}  {'corr_PAPE':>10s}  "
          f"{'rel_Δ %':>8s}  {'base_HR1':>8s}  {'corr_HR1':>8s}")
    print("-" * 95)
    for arm in ["T0", "T2"]:
        for proto in ["latent_nn", "key_nn_loo"]:
            for mech in ["V0", "W5"]:
                m = results[arm][proto][mech]
                rel = (1 - m["corr_pape"] / m["base_pape"]) * 100
                print(f"{arm:5s}  {mech:4s}  {proto:10s}  "
                      f"{m['base_pape']:10.2f}  {m['corr_pape']:10.2f}  "
                      f"{rel:+8.2f}  {m['base_hr@1']:8.2f}  {m['corr_hr@1']:8.2f}")
        print()

    print(f"key↔latent agreement rate (within train, leave-one-out KEY-NN):")
    for arm in ["T0", "T2"]:
        print(f"  {arm}: {results[arm]['key_eq_latent_assign_rate']*100:.1f}%")

    out_path = TRAIN_VQ / "train_vq_check.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
