"""v04 G5 cross-cell — Peak-VQ + W5 Hybrid correction on top of an FL backbone.

For each (seed, FL algorithm) pair:
    1. Load the FL algorithm's final state into a MinimalNBEATSx backbone.
    2. Forward all train apts' train-segment windows (stride=24) -> h_g
       and self-derived aux (â = ŷ.max, ĥ = ŷ.argmax). Self-derived because
       the FL backbone has no peak_aux head — same construction v01 §4.3
       used for the T0 row of the E1 ablation.
    3. Fit KMeans++ M=32 on the collected h_g; compute per-cluster residual
       offsets and the KEY pool / scaler (mirrors v02 03_fit_codebook).
    4. Forward cold apts (warm-start z-norm, stride=24); route via Key-Route
       (R0) and apply the W5 Hybrid correction at both v01 operating points.
    5. Save baseline + per-op-point metrics.

This isolates the *complementarity* of Peak-VQ on top of any FL pattern:
the FL backbone alone vs FL + Peak-VQ on the same cold apts and same seed.

CLI:

    uv run python experiments/v04_full_baseline_comparison/04_peakvq_on_fl.py \\
        --seed 42 --backbone_algorithm fedavg
    uv run python experiments/v04_full_baseline_comparison/04_peakvq_on_fl.py \\
        --seed 42 --backbone_algorithm fedrep

Output:
    outputs/v04_full_baseline_comparison/seed{S}/peakvq_on_{algorithm}/result.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import HORIZON, OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO
from dataloader.splits import load_v02_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from eval.cold_helpers import OPERATING_POINTS, gauss_template, metrics_z_to_kw, route_R0
from models.nbeatsx import MinimalNBEATSx
from models.vq_kmeans import VectorQuantizerKMeans
from probes.peak_descriptor import extract_key

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V04_OUT_ROOT = OUTPUT_DIR / "v04_full_baseline_comparison"

SUPPORTED = ["fedavg", "fedrep"]


def _gather_segment_self_aux(apts: list[str], model, batch: int = 512, stride: int = HORIZON) -> dict:
    """Forward each apt's train segment; collect h_g + self-derived aux + KEY.

    self-derived aux (v01 §4.3 E1 pattern for T0 backbones):
        pred_amp = y_hat.max(dim=1).values   in z-norm space
        pred_hr  = y_hat.argmax(dim=1)        as integer hour
    """
    model.eval()
    h_chunks, yhat_chunks, ytrue_chunks = [], [], []
    amp_chunks, hr_chunks, key_chunks = [], [], []
    mean_chunks, std_chunks = [], []
    for apt in apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            continue
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        seg = series[:train_end]
        m_ = float(seg.mean())
        s_ = float(seg.std()) if seg.std() > 1e-8 else 1.0
        ds = HouseholdDataset(seg, m_, s_, stride=stride)
        if len(ds) == 0:
            continue
        for x, y in DataLoader(ds, batch_size=batch, shuffle=False):
            x_dev = x.to(DEVICE, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if DEVICE.type == "cuda" else _NullCtx():
                y_hat, hiddens = model(x_dev)
            y_hat_np = y_hat.float().cpu().numpy()
            h_chunks.append(hiddens["h_generic"].float().cpu().numpy())
            yhat_chunks.append(y_hat_np)
            ytrue_chunks.append(y.numpy())
            amp_chunks.append(y_hat_np.max(axis=1))                    # self-derived â
            hr_chunks.append(y_hat_np.argmax(axis=1).astype(np.int64))  # self-derived ĥ
            key_chunks.append(extract_key(x.numpy()))
            mean_chunks.append(np.full(len(y), m_, dtype=np.float32))
            std_chunks.append(np.full(len(y), s_, dtype=np.float32))
    return {
        "h_g": np.concatenate(h_chunks, 0).astype(np.float32),
        "y_hat_z": np.concatenate(yhat_chunks, 0).astype(np.float32),
        "y_true_z": np.concatenate(ytrue_chunks, 0).astype(np.float32),
        "pred_amp": np.concatenate(amp_chunks, 0).astype(np.float32),
        "pred_hr": np.concatenate(hr_chunks, 0).astype(np.int64),
        "key": np.concatenate(key_chunks, 0).astype(np.float32),
        "mean": np.concatenate(mean_chunks, 0),
        "std": np.concatenate(std_chunks, 0),
    }


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main() -> None:
    ap = argparse.ArgumentParser(description="v04 G5: Peak-VQ + W5 Hybrid on top of an FL backbone.")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--backbone_algorithm", required=True, choices=SUPPORTED,
                    help="Which FL run's final_state_dict.pt to use as backbone.")
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=HORIZON)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    seed_root = V04_OUT_ROOT / f"seed{args.seed}"

    fl_dir = seed_root / args.backbone_algorithm
    sd_path = fl_dir / "final_state_dict.pt"
    if not sd_path.exists():
        raise FileNotFoundError(
            f"missing {sd_path}; run 01_fl_train.py --seed {args.seed} --algorithm {args.backbone_algorithm} first."
        )

    sp = load_v02_split(args.seed)
    train_apts, cold_apts = sp["train"], sp["cold"]
    out_dir = seed_root / f"peakvq_on_{args.backbone_algorithm}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v04 G5] seed={args.seed}  backbone={args.backbone_algorithm}  M={args.M}")
    backbone = MinimalNBEATSx().to(DEVICE).eval()
    backbone.load_state_dict(torch.load(sd_path, map_location="cpu", weights_only=False))
    print(f"[v04 G5] backbone loaded ({sum(p.numel() for p in backbone.parameters())} params)")

    # ---- Train side: collect h_g + self-derived aux, fit codebook + offsets + KEY pool ----
    t0 = time.time()
    tr = _gather_segment_self_aux(train_apts, backbone, batch=args.batch_size, stride=args.stride)
    print(f"[v04 G5] train: {tr['h_g'].shape[0]} windows; gather elapsed: {time.time()-t0:.1f}s")

    vq = VectorQuantizerKMeans(num_embeddings=args.M, embedding_dim=tr["h_g"].shape[1], random_state=args.seed)
    diag = vq.fit(torch.from_numpy(tr["h_g"]).float())
    print(f"[v04 G5] vq diag: util={diag['utilization']:.3f}  ppl={diag['perplexity']:.2f}  "
          f"k_min={diag['k_min']}  k_max={diag['k_max']}")
    centroids = vq.codebook.cpu().numpy()
    counts = vq.counts.cpu().numpy()
    h_t = torch.from_numpy(tr["h_g"]).float()
    with torch.no_grad():
        _, idx_t = vq(h_t)
    cluster_idx = idx_t.cpu().numpy().astype(np.int64)
    residuals = tr["y_true_z"] - tr["y_hat_z"]
    offsets = np.zeros((args.M, residuals.shape[1]), dtype=np.float32)
    for c in range(args.M):
        mask = cluster_idx == c
        if mask.any():
            offsets[c] = residuals[mask].mean(axis=0)

    # KEY pool + scaler (R0 routing).
    key_pool = tr["key"].astype(np.float32)
    key_scaler = StandardScaler().fit(key_pool)
    key_pool_scaled = key_scaler.transform(key_pool).astype(np.float32)

    # ---- Cold side: forward + Key-Route + W5 Hybrid at both op-points ----
    co = _gather_segment_self_aux(cold_apts, backbone, batch=args.batch_size, stride=args.stride)
    print(f"[v04 G5] cold: {co['y_hat_z'].shape[0]} windows")

    cold_cluster = route_R0(
        co["key"],
        key_scaler.mean_.astype(np.float32),
        key_scaler.scale_.astype(np.float32),
        key_pool_scaled,
        cluster_idx,
    )
    cluster_offset = offsets[cold_cluster]

    base = metrics_z_to_kw(co["y_true_z"], co["y_hat_z"], co["mean"], co["std"])
    print(f"[v04 G5] baseline (FL backbone, no correction): "
          f"PAPE={base['pape']:.2f}  HR@1={base['hr@1']:.1f}  HR@2={base['hr@2']:.1f}")

    out_per_op = {}
    for op_name, op in OPERATING_POINTS.items():
        g = gauss_template(co["pred_hr"], co["pred_amp"], sigma=op["sigma"])
        corrected_z = (co["y_hat_z"] + op["alpha_v0"] * cluster_offset + op["alpha_w1"] * g).astype(np.float32)
        m = metrics_z_to_kw(co["y_true_z"], corrected_z, co["mean"], co["std"])
        ratio = m["pape"] / base["pape"] if base["pape"] > 0 else float("nan")
        delta = base["pape"] - m["pape"]
        print(f"[v04 G5] {op_name}: PAPE={m['pape']:.2f} (Δ={delta:+.2f} kW vs baseline; ratio={ratio:.3f})")
        out_per_op[op_name] = {
            "sigma": op["sigma"], "alpha_v0": op["alpha_v0"], "alpha_w1": op["alpha_w1"],
            "metrics": m,
            "pape_ratio_vs_baseline": ratio,
            "pape_delta_kw": delta,
        }

    elapsed = time.time() - t0
    out = {
        "algorithm": f"peakvq_on_{args.backbone_algorithm}",
        "backbone_algorithm": args.backbone_algorithm,
        "seed": int(args.seed),
        "M": int(args.M),
        "vq_diagnostics": {
            "utilization": float(diag["utilization"]),
            "perplexity": float(diag["perplexity"]),
            "k_min": int(diag["k_min"]),
            "k_max": int(diag["k_max"]),
            "n_empty_clusters": int((counts == 0).sum()),
        },
        "n_train_windows": int(tr["h_g"].shape[0]),
        "n_cold_windows": int(co["y_hat_z"].shape[0]),
        "n_cold_apts": int(np.unique([0]).size if not cold_apts else len(cold_apts)),
        "baseline": base,
        "operating_points": out_per_op,
        "elapsed_seconds": elapsed,
        "comment": (
            "Peak-VQ + W5 Hybrid layered on top of an FL backbone. The FL "
            "backbone is MinimalNBEATSx (no peak_aux head), so W5's (amp, hr) "
            "are taken self-derived from the forecast itself (y_hat.max / "
            "y_hat.argmax) — same construction v01 §4.3 E1 used for the T0 row."
        ),
    }
    with open(out_dir / "result.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[v04 G5] saved -> {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
