"""Priority 1B — Peak-VQ codebook size (M) sensitivity sweep.

Why
---
The unified pFL paper fixes the codebook size at M=32 (v01/v02 default,
chosen for "M=32 codebook fits in ~5 KB and gives the best v01 G3
healthwise k_min ≥ 113 floor"). M=32 is therefore *one* design point —
the headline gap between v02's 35.7 PAPE and the v04 baselines could in
principle be sensitive to this choice. This sweep gives the codebook-size
sensitivity curve to put the M=32 choice in context.

What this script does
---------------------
For each ``M ∈ {8, 16, 64}``:

    1. Reload the **existing** v02 T2 backbone for this seed
       (``outputs/v02_fl_8020_ratio/seed{S}/T2/best.pt``).
       The backbone is NOT retrained — only the codebook varies.
    2. Forward all 80 train apts' train-segment windows (stride=24)
       through the frozen T2 backbone -> collect (h_g, y_hat_z, y_true_z, KEY).
    3. Fit ``KMeans(n_clusters=M, init="k-means++", n_init=10)`` on h_g;
       compute per-cluster residual offsets and the KEY pool / scaler
       — same recipe as ``v02 03_fit_codebook.py``, only ``n_clusters`` changes.
    4. Forward the 20 cold apts (warm-start z-norm, stride=24); route via
       R0 (Key-Route 1-NN); apply W5 hybrid at both v01 op-points.
    5. Save metrics for all three M values in a single result.json.

The M=32 baseline already lives in
``outputs/v02_fl_8020_ratio/seed{S}/coldstart_R0.json`` from v02; an
external aggregator can splice the four points {8, 16, 32, 64} together
to draw the sensitivity curve.

Operating points are carried over from v01 §4.2 unchanged; cold-side
α tuning is forbidden (CLAUDE.md / v01 §5.4.1).

CLI
---
    uv run python experiments/v04_full_baseline_comparison/09_fix_rerun/03_m_sensitivity.py --seed 42

Output
------
    outputs/v04_full_baseline_comparison/09_fix_rerun/seed{S}/m_sensitivity/
        result.json
            ├── seed
            ├── M_sweep: [8, 16, 64]
            ├── metrics_by_M:
            │       M_8  : {HR-preserving: {...}, PAPE-aggressive: {...},
            │               vq_diagnostics: {util, ppl, k_min, k_max, n_empty},
            │               baseline: {pape, hr@1, hr@2, mae},
            │               routing_diagnostics: {n_clusters_used, usage_min, usage_max, usage_mean}}
            │       M_16 : {...}
            │       M_64 : {...}
            ├── elapsed_seconds, n_train_windows, n_cold_windows, n_cold_apts

Wall-clock per seed: ~5 min on a 5070 Ti
    (1 forward pass on train apts + 3 KMeans fits + 1 forward pass on cold
     apts + 3 cold corrections; the train-side forward pass dominates).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import HORIZON, OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO
from dataloader.splits import load_v02_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from eval.cold_helpers import (
    OPERATING_POINTS,
    gather_cold,
    gauss_template,
    metrics_z_to_kw,
    route_R0,
)
from models.nbeatsx_aux import NBEATSxAux
from models.vq_kmeans import VectorQuantizerKMeans
from probes.peak_descriptor import extract_key

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V02_OUT_ROOT = OUTPUT_DIR / "v02_fl_8020_ratio"
V04_FIX_OUT_ROOT = OUTPUT_DIR / "v04_full_baseline_comparison" / "09_fix_rerun"

DEFAULT_M_SWEEP = [8, 16, 64]


def _gather_train_segment(
    apts: list[str], model: NBEATSxAux, batch: int, stride: int,
) -> dict[str, np.ndarray]:
    """Same shape as v02 03_fit_codebook.gather_train_segment, kept local."""
    h_chunks, yhat_chunks, ytrue_chunks, key_chunks = [], [], [], []
    n_per_apt: list[int] = []
    model.eval()
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
        loader = DataLoader(ds, batch_size=batch, shuffle=False)
        per_apt = 0
        for x, y in loader:
            x_dev = x.to(DEVICE)
            with torch.no_grad():
                y_hat, hiddens, _aux = model(x_dev)
            h_chunks.append(hiddens["h_generic"].cpu().numpy())
            yhat_chunks.append(y_hat.cpu().numpy())
            ytrue_chunks.append(y.numpy())
            key_chunks.append(extract_key(x.numpy()))
            per_apt += len(x)
        n_per_apt.append(per_apt)
    return {
        "h_g": np.concatenate(h_chunks, axis=0).astype(np.float32),
        "y_hat_z": np.concatenate(yhat_chunks, axis=0).astype(np.float32),
        "y_true_z": np.concatenate(ytrue_chunks, axis=0).astype(np.float32),
        "key": np.concatenate(key_chunks, axis=0).astype(np.float32),
        "n_windows_per_apt": np.asarray(n_per_apt, dtype=np.int64),
    }


def evaluate_one_M(
    M: int,
    seed: int,
    tr: dict[str, np.ndarray],
    co: dict[str, np.ndarray],
) -> dict:
    """Fit a codebook of size M on tr['h_g']; evaluate cold at both op-points.

    Returns the same shape v02 04_coldstart_eval.evaluate_routing returns,
    with an added vq_diagnostics block (utilization / perplexity / k_min / k_max).
    Routing is fixed to R0 (Key-Route, 1-NN on the train KEY pool) for the
    sensitivity sweep — adding R1 would double the cells with no extra
    information about M's effect.
    """
    vq = VectorQuantizerKMeans(num_embeddings=M, embedding_dim=tr["h_g"].shape[1], random_state=seed)
    diag = vq.fit(torch.from_numpy(tr["h_g"]).float())

    centroids = vq.codebook.cpu().numpy()
    counts = vq.counts.cpu().numpy()
    h_t = torch.from_numpy(tr["h_g"]).float()
    with torch.no_grad():
        _, idx_t = vq(h_t)
    cluster_idx = idx_t.cpu().numpy().astype(np.int64)
    residuals = tr["y_true_z"] - tr["y_hat_z"]
    offsets = np.zeros((M, residuals.shape[1]), dtype=np.float32)
    for c in range(M):
        mask = cluster_idx == c
        if mask.any():
            offsets[c] = residuals[mask].mean(axis=0)

    key_pool = tr["key"].astype(np.float32)
    key_scaler = StandardScaler().fit(key_pool)
    key_pool_scaled = key_scaler.transform(key_pool).astype(np.float32)

    cold_cluster = route_R0(
        co["key"],
        key_scaler.mean_.astype(np.float32),
        key_scaler.scale_.astype(np.float32),
        key_pool_scaled,
        cluster_idx,
    )
    cluster_offset = offsets[cold_cluster]

    base = metrics_z_to_kw(co["y_true_z"], co["y_hat_z"], co["mean"], co["std"])

    op_results: dict = {}
    for op_name, op in OPERATING_POINTS.items():
        g = gauss_template(co["pred_hr"], co["pred_amp"], sigma=op["sigma"])
        corrected_z = (
            co["y_hat_z"]
            + op["alpha_v0"] * cluster_offset
            + op["alpha_w1"] * g
        ).astype(np.float32)
        m = metrics_z_to_kw(co["y_true_z"], corrected_z, co["mean"], co["std"])
        ratio = m["pape"] / base["pape"] if base["pape"] > 0 else float("nan")
        delta = base["pape"] - m["pape"]
        op_results[op_name] = {
            "sigma": op["sigma"], "alpha_v0": op["alpha_v0"], "alpha_w1": op["alpha_w1"],
            "metrics": m,
            "pape_delta_kw_vs_baseline": delta,
            "pape_ratio_vs_baseline": ratio,
        }

    usage_counts = np.bincount(cold_cluster, minlength=M)
    return {
        "M": int(M),
        "vq_diagnostics": {
            "utilization": float(diag["utilization"]),
            "perplexity": float(diag["perplexity"]),
            "k_min": int(diag["k_min"]),
            "k_max": int(diag["k_max"]),
            "n_empty_clusters": int((counts == 0).sum()),
            "kmeans_inertia": float(diag["kmeans_inertia"]),
        },
        "baseline": base,
        "HR-preserving":   op_results["HR-preserving"],
        "PAPE-aggressive": op_results["PAPE-aggressive"],
        "routing_diagnostics": {
            "n_clusters_used": int((usage_counts > 0).sum()),
            "usage_min": int(usage_counts.min()),
            "usage_max": int(usage_counts.max()),
            "usage_mean": float(usage_counts.mean()),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="v04 09_fix_rerun: M-sensitivity sweep on v02 T2 backbone.")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--M_sweep", nargs="+", type=int, default=DEFAULT_M_SWEEP,
                    help="Codebook sizes to evaluate (M=32 baseline lives in v02; default sweep is {8, 16, 64}).")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--stride", type=int, default=HORIZON)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    sp = load_v02_split(args.seed)
    train_apts, cold_apts = sp["train"], sp["cold"]
    out_dir = V04_FIX_OUT_ROOT / f"seed{args.seed}" / "m_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the EXISTING v02 T2 backbone — do NOT retrain.
    ckpt = V02_OUT_ROOT / f"seed{args.seed}" / "T2" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"v02 T2 backbone missing: {ckpt}. "
            f"Run experiments/v02_fl_8020_ratio/02_train_arms.py --seed {args.seed} --arms T2 first."
        )
    print(f"[v04 1B] seed={args.seed}  M_sweep={args.M_sweep}  ckpt={ckpt}")

    model = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    print(f"[v04 1B] backbone loaded ({sum(p.numel() for p in model.parameters())} params)")

    # ---- Single train-side forward pass; reused across all M values ----
    t0 = time.time()
    tr = _gather_train_segment(train_apts, model, batch=args.batch, stride=args.stride)
    print(f"[v04 1B] train: {tr['h_g'].shape[0]} windows, gather elapsed: {time.time() - t0:.1f}s")

    # ---- Single cold-side forward pass via the shared gather_cold helper ----
    co = gather_cold(cold_apts, model, batch=args.batch, stride=args.stride, verbose_skips=False)
    print(f"[v04 1B] cold: {co['y_true_z'].shape[0]} windows from "
          f"{len(np.unique(co['apt']))} apts")

    # ---- Sweep M ----
    metrics_by_M: dict = {}
    for M in args.M_sweep:
        t_m = time.time()
        res = evaluate_one_M(M, args.seed, tr, co)
        m_key = f"M_{M}"
        metrics_by_M[m_key] = res
        b = res["baseline"]; hp = res["HR-preserving"]["metrics"]; pa = res["PAPE-aggressive"]["metrics"]
        print(
            f"[v04 1B] M={M:3d} "
            f"baseline={b['pape']:.2f}  HR-preserving={hp['pape']:.2f}  "
            f"PAPE-aggressive={pa['pape']:.2f}  "
            f"util={res['vq_diagnostics']['utilization']:.3f}  "
            f"k_min={res['vq_diagnostics']['k_min']}  "
            f"({time.time() - t_m:.1f}s)"
        )

    elapsed = time.time() - t0

    out = {
        "algorithm": "m_sensitivity",
        "seed": int(args.seed),
        "M_sweep": list(args.M_sweep),
        "config": {"batch": args.batch, "stride": args.stride,
                   "backbone_ckpt": str(ckpt), "routing": "R0",
                   "operating_points": OPERATING_POINTS},
        "metrics_by_M": metrics_by_M,
        "n_train_windows": int(tr["h_g"].shape[0]),
        "n_cold_windows":  int(co["y_true_z"].shape[0]),
        "n_cold_apts":     int(len(np.unique(co["apt"]))),
        "elapsed_seconds": elapsed,
        "comment": (
            "Codebook-size sensitivity sweep on the EXISTING v02 T2 backbone — "
            "backbone is NOT retrained, only the codebook varies. Default sweep "
            "{8, 16, 64} brackets the v02 default M=32; the M=32 cell lives in "
            "v02's coldstart_R0.json and can be spliced in at aggregation. "
            "Routing fixed to R0 (Key-Route); op-points carried over from v01 §4.2 "
            "unchanged. Diagnostics include k_min so the v01 health threshold "
            "(k_min ≥ 113 at M=32) interpretation can be reproduced for each M."
        ),
    }

    with open(out_dir / "result.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[v04 1B] saved -> {out_dir}")
    print(f"[v04 1B] total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()


# Expected output (seed=42, GTX 5070 Ti):
#   - One forward pass on 80 train apts (~30-60 s) + KMeans fits (small;
#     a few seconds each) + one forward pass on 20 cold apts + corrections.
#   - Total ~3-5 min per seed.
#   - Expected curve direction: very small M (8) collapses sparse routing
#     (k_min may approach 0); larger M (64) gives finer-grained clusters
#     but typical k_min drops below v01's 113 floor — both are interesting
#     data points for the headline "M=32 is well-chosen" claim.
