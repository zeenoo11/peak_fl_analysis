"""Fit the post-hoc Peak-VQ codebook on the v02 80-train-apt T2 latents (per-seed).

For one seed:
    1. Load the frozen T2 backbone produced by ``02_train_arms.py``.
    2. Forward all train apts' train-segment windows (stride=24, matching v01)
       through the frozen backbone; collect (h_g, y_hat_z, y_true_z, key).
    3. Fit KMeans++ with M=32 on h_g — the codebook is **post-hoc 1-shot**
       (CLAUDE.md: iterative federated KMeans is out of scope through v03).
    4. Compute per-cluster residual offsets in z-norm space:
           offset_c = mean over {windows i: c*(i) = c} of (y_true_z[i] - y_hat_z[i]).
    5. Build the KEY pool for R0 routing: 5-d KEY for every train window plus
       the StandardScaler params; cold side will reproduce the scaler exactly.

The codebook bundle is saved to
``outputs/v02_fl_8020_ratio/seed{S}/codebook.npz`` and a separate
``codebook_diagnostics.json`` records utilisation / perplexity / k_min so the
v02 G1 health-metric check (k_min ≥ 113 at M=32) is reproducible.

Per-seed invocation:
    uv run python experiments/v02_fl_8020_ratio/03_fit_codebook.py --seed 42
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
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO
from dataloader.splits import load_v02_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.nbeatsx_aux import NBEATSxAux
from models.vq_kmeans import VectorQuantizerKMeans
from probes.peak_descriptor import extract_key

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V02_OUT_ROOT = OUTPUT_DIR / "v02_fl_8020_ratio"


def gather_train_segment(
    apts: list[str],
    model: NBEATSxAux,
    batch: int = 256,
    stride: int = 24,
) -> dict[str, np.ndarray]:
    """Collect (h_g, y_hat_z, y_true_z, key) on the train segment of each apt.

    Stride matches v01's gather_features (= horizon, non-overlapping) so
    codebook fit statistics stay comparable across versions.
    """
    h_chunks, yhat_chunks, ytrue_chunks, key_chunks = [], [], [], []
    n_windows_per_apt = []
    for apt in apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            print(f"  [skip] {apt}: missing")
            continue
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        seg = series[:train_end]
        m_ = float(seg.mean())
        s_ = float(seg.std()) if seg.std() > 1e-8 else 1.0
        ds = HouseholdDataset(seg, m_, s_, stride=stride)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=batch, shuffle=False)
        per_apt = 0
        for x, y in loader:
            x_dev = x.to(DEVICE)
            with torch.no_grad():
                y_hat, hiddens, _ = model(x_dev)
            h_g = hiddens["h_generic"].cpu().numpy()
            h_chunks.append(h_g)
            yhat_chunks.append(y_hat.cpu().numpy())
            ytrue_chunks.append(y.numpy())
            key_chunks.append(extract_key(x.numpy()))
            per_apt += len(x)
        n_windows_per_apt.append(per_apt)
    return {
        "h_g": np.concatenate(h_chunks, axis=0),
        "y_hat_z": np.concatenate(yhat_chunks, axis=0),
        "y_true_z": np.concatenate(ytrue_chunks, axis=0),
        "key": np.concatenate(key_chunks, axis=0),
        "n_windows_per_apt": np.asarray(n_windows_per_apt, dtype=np.int64),
    }


def fit_codebook(seed: int, M: int, arm: str, batch: int, stride: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)

    apts = load_v02_split(seed)["train"]
    seed_root = V02_OUT_ROOT / f"seed{seed}"
    arm_dir = seed_root / arm
    ckpt = arm_dir / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"backbone checkpoint missing: {ckpt}. "
            f"Run 02_train_arms.py --seed {seed} --arms {arm} first."
        )

    model = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    print(f"[fit] seed={seed} arm={arm} backbone loaded ({sum(p.numel() for p in model.parameters())} params)")

    feats = gather_train_segment(apts, model, batch=batch, stride=stride)
    h_g = feats["h_g"]
    print(f"[fit] {len(apts)} apts, {h_g.shape[0]} train windows; stride={stride}")

    vq = VectorQuantizerKMeans(num_embeddings=M, embedding_dim=h_g.shape[1], random_state=seed)
    diag = vq.fit(torch.from_numpy(h_g).float())
    print(
        f"[fit] M={M}  util={diag['utilization']:.3f}  ppl={diag['perplexity']:.2f}  "
        f"k_min={diag['k_min']}  k_max={diag['k_max']}  inertia={diag['kmeans_inertia']:.1f}"
    )

    # Per-window cluster assignment + per-cluster residual offset.
    centroids = vq.codebook.cpu().numpy()
    counts = vq.counts.cpu().numpy()
    h_t = torch.from_numpy(h_g).float()
    with torch.no_grad():
        _, cluster_idx_t = vq(h_t)
    cluster_idx = cluster_idx_t.cpu().numpy().astype(np.int64)
    residuals = feats["y_true_z"] - feats["y_hat_z"]
    horizon = residuals.shape[1]
    offsets = np.zeros((M, horizon), dtype=np.float32)
    for c in range(M):
        mask = cluster_idx == c
        if mask.any():
            offsets[c] = residuals[mask].mean(axis=0)

    # KEY pool for R0 routing.
    key_pool = feats["key"].astype(np.float32)
    key_scaler = StandardScaler().fit(key_pool)
    key_pool_scaled = key_scaler.transform(key_pool).astype(np.float32)

    out_path = seed_root / "codebook.npz"
    np.savez(
        out_path,
        codebook=centroids.astype(np.float32),
        counts=counts.astype(np.int64),
        offsets=offsets,
        cluster_idx=cluster_idx.astype(np.int32),
        key_pool=key_pool,
        key_pool_scaled=key_pool_scaled,
        key_scaler_mean=key_scaler.mean_.astype(np.float32),
        key_scaler_scale=key_scaler.scale_.astype(np.float32),
        n_windows_per_apt=feats["n_windows_per_apt"],
    )

    diagnostics = {
        "seed": int(seed),
        "arm": arm,
        "split_version": "v02",
        "M": int(M),
        "embedding_dim": int(h_g.shape[1]),
        "n_train_apts": int(len(apts)),
        "n_train_windows": int(h_g.shape[0]),
        "stride": int(stride),
        "horizon": int(horizon),
        "vq_utilization": float(diag["utilization"]),
        "vq_perplexity": float(diag["perplexity"]),
        "vq_k_min": int(diag["k_min"]),
        "vq_k_max": int(diag["k_max"]),
        "vq_kmeans_inertia": float(diag["kmeans_inertia"]),
        "n_empty_clusters": int((counts == 0).sum()),
        "k_min_health_threshold_v01": 113,
        "k_min_health_pass": bool(int(diag["k_min"]) >= 113),
        "key_dim": int(key_pool.shape[1]),
    }
    with open(seed_root / "codebook_diagnostics.json", "w") as fh:
        json.dump(diagnostics, fh, indent=2)

    print(f"[fit] saved {out_path.name} + codebook_diagnostics.json")
    return diagnostics


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit M=32 KMeans codebook on v02 T2 latents (per-seed).")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--M", type=int, default=32, help="Codebook size.")
    ap.add_argument("--arm", type=str, default="T2", choices=["T2"], help="Backbone arm; v02 uses T2 only.")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--stride", type=int, default=24, help="Window stride; v01 uses 24 (= horizon).")
    args = ap.parse_args()

    fit_codebook(seed=args.seed, M=args.M, arm=args.arm, batch=args.batch, stride=args.stride)


if __name__ == "__main__":
    main()
