"""H1c gate: KV-VQ residual correction on 50 cold households.

For each H1b-PASS arm:
    1. Build per-cluster forecast offset from training data:
       offset_c = mean(true_z - baseline_z) over windows assigned to cluster c
       (where assignment = NN to codebook in latent space).
    2. For each cold input window:
       a. compute KEY (5-d peak descriptor from input).
       b. find c* via 1-NN in KEY space across training KEYs.
       c. baseline ŷ = NBEATSx forecast on cold input.
       d. corrected ŷ = baseline + alpha * offset_{c*}.
    3. Compare cold-PAPE(baseline) vs cold-PAPE(corrected).

PASS criterion (H1c): cold-PAPE(KV-VQ) <= 0.95 * cold-PAPE(baseline).
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
from probes.peak_descriptor import extract_key
from utils.metrics import seven_axis_metrics

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
probe_mod = import_module("03_probe_h1a")
load_arm_extractor = probe_mod.load_arm_extractor

OUT = OUTPUT_DIR / "v01_peak_from_latent"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = 0.5
PASS_RATIO = 0.95


def load_baseline_model(arm: str):
    """Returns the model used to produce the baseline forecast.
    For T2/T3 we use the aux model itself (since its forecast head was trained
    with peak_aux). For others, T0 backbone."""
    if arm in ("T0", "T1", "T4", "T5", "T6"):
        m = MinimalNBEATSx().to(DEVICE).eval()
        m.load_state_dict(torch.load(OUT / "T0" / "best.pt", map_location="cpu", weights_only=False))
        is_aux = False
    elif arm == "T2":
        m = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
        m.load_state_dict(torch.load(OUT / arm / "best.pt", map_location="cpu", weights_only=False))
        is_aux = True
    elif arm == "T3":
        m = NBEATSxAux(latent_source="h_concat").to(DEVICE).eval()
        m.load_state_dict(torch.load(OUT / arm / "best.pt", map_location="cpu", weights_only=False))
        is_aux = True
    return m, is_aux


def gather_test_segment(apts, model, is_aux, extract_fn):
    """Returns dicts of arrays: {key, lat, base_z, true_z, mean, std}."""
    keys, lats, base_z, true_z, m_arr, s_arr = [], [], [], [], [], []
    for apt in apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            continue
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        seg = series[:train_end]
        m = float(seg.mean()); s = float(seg.std()) if seg.std() > 1e-8 else 1.0
        ds = HouseholdDataset(seg, m, s, stride=24)
        for x, y in DataLoader(ds, batch_size=256, shuffle=False):
            keys.append(extract_key(x.numpy()))
            lats.append(extract_fn(x))
            with torch.no_grad():
                if is_aux:
                    y_hat, _, _ = model(x.to(DEVICE))
                else:
                    y_hat, _ = model(x.to(DEVICE))
            base_z.append(y_hat.cpu().numpy())
            true_z.append(y.numpy())
            m_arr.append(np.full(len(y), m)); s_arr.append(np.full(len(y), s))
    return {
        "key": np.concatenate(keys, axis=0),
        "lat": np.concatenate(lats, axis=0),
        "base_z": np.concatenate(base_z, axis=0),
        "true_z": np.concatenate(true_z, axis=0),
        "mean": np.concatenate(m_arr, axis=0),
        "std": np.concatenate(s_arr, axis=0),
    }


def main():
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    h1b = json.load(open(OUT / "quantize_h1b.json"))
    pass_arms = h1b.get("pass_arms_h1b", [])
    if not pass_arms:
        print("[H1c] no PASS arms from H1b — skipping.")
        with open(OUT / "coldstart_h1c.json", "w") as fh:
            json.dump({"pass_arms_h1c": [], "results": {}, "skipped_reason": "h1b empty"}, fh)
        return

    split = load_v10_split()
    train_apts = split["train"]; cold_apts = split["cold"]

    results = {}
    for arm in pass_arms:
        print(f"\n========== H1c {arm} ==========")
        m, is_aux = load_baseline_model(arm)
        extract_fn = load_arm_extractor(arm)

        tr = gather_test_segment(train_apts, m, is_aux, extract_fn)
        co = gather_test_segment(cold_apts, m, is_aux, extract_fn)
        print(f"  train windows: {tr['lat'].shape[0]}, cold windows: {co['lat'].shape[0]}")

        cb = np.load(OUT / arm / "codebook.npz")
        codebook = cb["codebook"]

        d_tr = ((tr["lat"][:, None, :] - codebook[None, :, :]) ** 2).sum(axis=2)
        idx_tr = d_tr.argmin(axis=1)
        residuals = tr["true_z"] - tr["base_z"]
        offsets = np.zeros((codebook.shape[0], 24), dtype=np.float32)
        counts = np.zeros(codebook.shape[0], dtype=np.int64)
        for c in range(codebook.shape[0]):
            mask = idx_tr == c
            counts[c] = int(mask.sum())
            if counts[c] > 0:
                offsets[c] = residuals[mask].mean(axis=0)
        print(f"  cluster counts: min={counts.min()}  max={counts.max()}  empty={(counts==0).sum()}")

        ks = StandardScaler().fit(tr["key"])
        Kt, Kc = ks.transform(tr["key"]), ks.transform(co["key"])
        nn = NearestNeighbors(n_neighbors=1).fit(Kt)
        _, neigh_idx = nn.kneighbors(Kc)
        cold_cluster = idx_tr[neigh_idx[:, 0]]
        cold_offset = offsets[cold_cluster]

        corrected = co["base_z"] + ALPHA * cold_offset
        true_kw = co["true_z"] * co["std"][:, None] + co["mean"][:, None]
        base_kw = co["base_z"] * co["std"][:, None] + co["mean"][:, None]
        corr_kw = corrected * co["std"][:, None] + co["mean"][:, None]
        m_base = seven_axis_metrics(true_kw, base_kw)
        m_corr = seven_axis_metrics(true_kw, corr_kw)
        ratio = m_corr["pape"] / m_base["pape"] if m_base["pape"] > 0 else 1.0
        gate = "PASS" if ratio <= PASS_RATIO else "FAIL"
        print(f"  cold PAPE: base={m_base['pape']:.2f}  KV={m_corr['pape']:.2f}  "
              f"ratio={ratio:.3f}  [{gate} H1c]")
        print(f"  cold HR@1: base={m_base['hr@1']:.1f}  KV={m_corr['hr@1']:.1f}")

        results[arm] = {
            "alpha": ALPHA, "n_cold_windows": int(co["true_z"].shape[0]),
            "baseline": m_base, "kv_vq": m_corr, "pape_ratio": ratio,
            "gate_h1c": gate, "cluster_counts_train": counts.tolist(),
        }

    pass_h1c = [a for a, r in results.items() if r["gate_h1c"] == "PASS"]
    print(f"\n[H1c] PASS: {pass_h1c}")
    with open(OUT / "coldstart_h1c.json", "w") as fh:
        json.dump({"alpha": ALPHA, "pass_threshold_ratio": PASS_RATIO,
                   "pass_arms_h1c": pass_h1c, "results": results}, fh, indent=2)


if __name__ == "__main__":
    main()
