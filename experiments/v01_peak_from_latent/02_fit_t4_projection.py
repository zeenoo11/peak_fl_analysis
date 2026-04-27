"""T4: post-hoc Ridge projection of h_concat to 32-d peak-aware subspace.

Backbone (T0 ckpt) is untouched. We extract h_concat from 50 train apts,
fit a multi-output Ridge on (peak_amp_z, one-hot peak_hr), then take the
top-32 rows by row norm as our projection W ∈ R^{32, 192}.

latent_T4(x) = StandardScaler(h_concat(x)) @ W^T
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO
from dataloader.splits import load_v10_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.nbeatsx import MinimalNBEATSx

OUT_DIR = OUTPUT_DIR / "v01_peak_from_latent" / "T4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
T0_CKPT = OUTPUT_DIR / "v01_peak_from_latent" / "T0" / "best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJ_DIM = 32


def extract_h_concat_and_targets(model, apts: list[str]):
    h_chunks, amp_chunks, hr_chunks = [], [], []
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
            with torch.no_grad():
                _, hidd = model(x.to(DEVICE))
            h_concat = torch.cat(
                [hidd["h_trend"], hidd["h_seasonal"], hidd["h_generic"]], dim=1
            )
            h_chunks.append(h_concat.cpu().numpy())
            amp_chunks.append(y.numpy().max(axis=1))
            hr_chunks.append(y.numpy().argmax(axis=1))
    return (
        np.concatenate(h_chunks, axis=0),
        np.concatenate(amp_chunks, axis=0),
        np.concatenate(hr_chunks, axis=0),
    )


def main():
    np.random.seed(RANDOM_SEED)
    print(f"[T4] loading T0: {T0_CKPT}")
    state = torch.load(T0_CKPT, map_location="cpu", weights_only=False)
    model = MinimalNBEATSx().to(DEVICE).eval()
    model.load_state_dict(state, strict=True)

    apts = load_v10_split()["train"][:40]   # train_probe split — exclude held-out cold_probe apts (last 10) used in 03_probe_h1a.py
    print(f"[T4] extracting h_concat from {len(apts)} train_probe apts (stride=24)")
    H, amp, hr = extract_h_concat_and_targets(model, apts)
    print(f"[T4] H={H.shape}  amp range [{amp.min():.2f}, {amp.max():.2f}]")

    Y = np.concatenate([amp[:, None], np.eye(24)[hr]], axis=1)
    sc = StandardScaler().fit(H)
    Hs = sc.transform(H)

    ridge = Ridge(alpha=1.0).fit(Hs, Y)
    coef = ridge.coef_  # [25, 192]
    if PROJ_DIM <= coef.shape[0]:
        norms = np.linalg.norm(coef, axis=1)
        top = np.argsort(-norms)[:PROJ_DIM]
        W = coef[top]
    else:
        rng = np.random.RandomState(RANDOM_SEED)
        extra = rng.randn(PROJ_DIM - coef.shape[0], coef.shape[1])
        W = np.concatenate([coef, extra], axis=0)
    print(f"[T4] W={W.shape}")

    np.savez(OUT_DIR / "W.npz", W=W, scaler_mean=sc.mean_, scaler_scale=sc.scale_)
    with open(OUT_DIR / "training_log.json", "w") as fh:
        json.dump({
            "arm": "T4",
            "method": "Ridge multi-output, top-rows-by-norm",
            "n_train_windows": int(H.shape[0]),
            "n_train_apts": len(apts),
            "proj_dim": PROJ_DIM,
            "ridge_alpha": 1.0,
        }, fh, indent=2)
    print(f"[T4] saved {OUT_DIR / 'W.npz'}")


if __name__ == "__main__":
    main()
