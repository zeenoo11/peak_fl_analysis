"""H1b gate: quantize PASS arms with KMeans++, verify peak info preserved.

Procedure per PASS arm:
    1. Extract latent from 40 train_probe apts.
    2. Fit VectorQuantizerKMeans(M=32) on those latents.
    3. Quantize train+across latents, re-fit Ridge probe on z_q.
    4. Ratio R²(z_q) / R²(z_raw) >= 0.90 PASS.

Codebook saved to outputs/v01_peak_from_latent/{arm}/codebook.npz.
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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from config import OUTPUT_DIR, RANDOM_SEED
from dataloader.splits import load_v10_split
from models.vq_kmeans import VectorQuantizerKMeans
from utils.metrics import compute_pape

# Reuse extractors / gather_features from 03_probe_h1a
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
probe_mod = import_module("03_probe_h1a")
load_arm_extractor = probe_mod.load_arm_extractor
gather_features = probe_mod.gather_features

OUT = OUTPUT_DIR / "v01_peak_from_latent"
M = 32
PASS_RATIO = 0.90


def main():
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    h1a = json.load(open(OUT / "probe_h1a.json"))
    pass_arms = h1a["pass_arms"]
    if not pass_arms:
        print("[H1b] no PASS arms from H1a — skipping.")
        with open(OUT / "quantize_h1b.json", "w") as fh:
            json.dump({"pass_arms_h1b": [], "results": {}, "skipped_reason": "h1a empty"}, fh)
        return

    apts = load_v10_split()["train"]
    train_apts = apts[:40]; cold_probe_apts = apts[40:]

    results = {}
    for arm in pass_arms:
        print(f"\n========== quantize {arm} ==========")
        extract_fn = load_arm_extractor(arm)
        X_tr, amp_tr, _, _ = gather_features(extract_fn, train_apts)
        X_te, amp_te, _, _ = gather_features(extract_fn, cold_probe_apts)

        D = X_tr.shape[1]
        vq = VectorQuantizerKMeans(num_embeddings=M, embedding_dim=D, random_state=RANDOM_SEED)
        diag = vq.fit(torch.from_numpy(X_tr).float())
        print(f"  vq diag: util={diag['utilization']:.3f}  ppl={diag['perplexity']:.2f}  "
              f"k_min={diag['k_min']}  k_max={diag['k_max']}  inertia={diag['kmeans_inertia']:.1f}")

        sc = StandardScaler().fit(X_tr)
        ridge_raw = Ridge(alpha=1.0).fit(sc.transform(X_tr), amp_tr)
        pred_raw = ridge_raw.predict(sc.transform(X_te))
        r2_raw = float(r2_score(amp_te, pred_raw))
        pape_raw = probe_mod.peak_mape(amp_te, pred_raw)

        with torch.no_grad():
            Zq_tr, _ = vq(torch.from_numpy(X_tr).float())
            Zq_te, _ = vq(torch.from_numpy(X_te).float())
        Zq_tr, Zq_te = Zq_tr.numpy(), Zq_te.numpy()
        sc_q = StandardScaler().fit(Zq_tr)
        ridge_q = Ridge(alpha=1.0).fit(sc_q.transform(Zq_tr), amp_tr)
        pred_q = ridge_q.predict(sc_q.transform(Zq_te))
        r2_q = float(r2_score(amp_te, pred_q))
        pape_q = probe_mod.peak_mape(amp_te, pred_q)

        ratio = r2_q / r2_raw if r2_raw > 0 else 0.0
        gate = "PASS" if ratio >= PASS_RATIO else "FAIL"
        print(f"  R²(raw)={r2_raw:.3f}  R²(q)={r2_q:.3f}  ratio={ratio:.3f}  [{gate} H1b]")
        print(f"  PAPE(raw)={pape_raw:.1f}%  PAPE(q)={pape_q:.1f}%")

        np.savez(OUT / arm / "codebook.npz",
                 codebook=vq.codebook.cpu().numpy(),
                 counts=vq.counts.cpu().numpy())
        results[arm] = {
            "vq_diagnostics": diag,
            "r2_raw": r2_raw, "r2_quantized": r2_q, "ratio": ratio,
            "pape_raw": pape_raw, "pape_quantized": pape_q,
            "gate_h1b": gate,
        }

    pass_h1b = [a for a, r in results.items() if r["gate_h1b"] == "PASS"]
    print(f"\n[H1b] PASS: {pass_h1b}")
    with open(OUT / "quantize_h1b.json", "w") as fh:
        json.dump({"M": M, "pass_threshold_ratio": PASS_RATIO,
                   "pass_arms_h1b": pass_h1b, "results": results}, fh, indent=2)


if __name__ == "__main__":
    main()
