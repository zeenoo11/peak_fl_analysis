"""Cold-side zero-shot inference under {R0, R1} routing × {HR-preserving, PAPE-aggressive}.

Inputs (per seed):
    outputs/v02_fl_8020_ratio/seed{S}/T2/best.pt       — frozen backbone + aux head
    outputs/v02_fl_8020_ratio/seed{S}/codebook.npz     — centroids, offsets, KEY pool, cluster_idx
    outputs/v02_fl_8020_ratio/splits/v02_8020_seed{S}.yaml  — train/cold apartments

For each cold apt:
    1. warm-start z-norm on its OWN first 70% (mirrors v01 cold protocol).
    2. sliding windows on the train-segment (stride=24, matching v01).
    3. frozen forward -> (y_hat_z, h_g, amp_pred, hr_pred_int, key).

Routing:
    R0 — KEY(x) -> StandardScaler (params from codebook.npz) -> 1-NN on
         train KEY pool -> cluster_idx of that train window.
    R1 — argmin_c ||h_g - codebook[c]||_2 directly (×12 info, no extra fwd).

Correction (W5 hybrid, both v01 operating points):
    g(t; h_hat, a_hat, sigma) = a_hat * exp(-(t - h_hat)^2 / (2*sigma^2))
                                normalised so g.max(axis=1) == a_hat.
    y_corr_z = y_hat_z + alpha_v0 * offsets[c*] + alpha_w1 * g

Outputs:
    outputs/v02_fl_8020_ratio/seed{S}/coldstart_R0.json
    outputs/v02_fl_8020_ratio/seed{S}/coldstart_R1.json
        — baseline + per-op-point metrics + aux diagnostics + cluster-usage stats.
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
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from config import OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO
from dataloader.splits import load_v02_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.nbeatsx_aux import NBEATSxAux
from probes.peak_descriptor import extract_key
from utils.metrics import compute_hr, compute_mae, compute_pape

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V02_OUT_ROOT = OUTPUT_DIR / "v02_fl_8020_ratio"

# v01 carry-over operating points (plans/v02-01_fl_8020_ratio.md "Non-goals").
OPERATING_POINTS = {
    "HR-preserving": {"sigma": 3.0, "alpha_v0": 1.0, "alpha_w1": 0.1},
    "PAPE-aggressive": {"sigma": 3.0, "alpha_v0": 1.5, "alpha_w1": 0.5},
}


def gather_cold(
    apts: list[str],
    model: NBEATSxAux,
    batch: int = 256,
    stride: int = 24,
) -> dict[str, np.ndarray]:
    """Forward pass on every cold apt's train-segment (warm-start z-norm)."""
    h_chunks, yhat_chunks, ytrue_chunks = [], [], []
    amp_chunks, hr_chunks, key_chunks = [], [], []
    mean_chunks, std_chunks, apt_chunks = [], [], []
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
        for x, y in loader:
            x_dev = x.to(DEVICE)
            with torch.no_grad():
                y_hat, hiddens, (amp_p, hr_p) = model(x_dev)
            h_chunks.append(hiddens["h_generic"].cpu().numpy())
            yhat_chunks.append(y_hat.cpu().numpy())
            ytrue_chunks.append(y.numpy())
            amp_chunks.append(amp_p.cpu().numpy().reshape(-1))
            hr_chunks.append(hr_p.argmax(dim=1).cpu().numpy())
            key_chunks.append(extract_key(x.numpy()))
            mean_chunks.append(np.full(len(y), m_, dtype=np.float32))
            std_chunks.append(np.full(len(y), s_, dtype=np.float32))
            apt_chunks.append(np.array([apt] * len(y)))
    return {
        "h_g": np.concatenate(h_chunks, axis=0).astype(np.float32),
        "y_hat_z": np.concatenate(yhat_chunks, axis=0).astype(np.float32),
        "y_true_z": np.concatenate(ytrue_chunks, axis=0).astype(np.float32),
        "pred_amp": np.concatenate(amp_chunks, axis=0).astype(np.float32),
        "pred_hr": np.concatenate(hr_chunks, axis=0).astype(np.int64),
        "key": np.concatenate(key_chunks, axis=0).astype(np.float32),
        "mean": np.concatenate(mean_chunks, axis=0),
        "std": np.concatenate(std_chunks, axis=0),
        "apt": np.concatenate(apt_chunks, axis=0),
    }


def gauss_template(
    pred_hr: np.ndarray,
    pred_amp: np.ndarray,
    sigma: float,
    length: int = 24,
) -> np.ndarray:
    """Gaussian peak template, normalised so g.max(axis=1) == pred_amp.

    Mirrors experiments/v01_peak_from_latent/09_iter4_mechanisms.py:gauss_template.
    """
    t = np.arange(length, dtype=np.float32)[None, :]
    g = np.exp(-0.5 * ((t - pred_hr.astype(np.float32)[:, None]) / sigma) ** 2)
    g = g / g.max(axis=1, keepdims=True)
    return (g * pred_amp[:, None]).astype(np.float32)


def metrics_z_to_kw(
    true_z: np.ndarray,
    pred_z: np.ndarray,
    mean_arr: np.ndarray,
    std_arr: np.ndarray,
) -> dict:
    true_kw = true_z * std_arr[:, None] + mean_arr[:, None]
    pred_kw = pred_z * std_arr[:, None] + mean_arr[:, None]
    return {
        "pape": float(compute_pape(true_kw, pred_kw)),
        "hr@1": float(compute_hr(true_kw, pred_kw, tol=1)),
        "hr@2": float(compute_hr(true_kw, pred_kw, tol=2)),
        "mae": float(compute_mae(true_kw, pred_kw)),
    }


def route_R0(
    co_key: np.ndarray,
    key_scaler_mean: np.ndarray,
    key_scaler_scale: np.ndarray,
    key_pool_scaled: np.ndarray,
    train_cluster_idx: np.ndarray,
) -> np.ndarray:
    """Cold KEY -> 1-NN on scaled train KEY pool -> train window's cluster_idx."""
    co_key_scaled = (co_key - key_scaler_mean) / key_scaler_scale
    nn = NearestNeighbors(n_neighbors=1).fit(key_pool_scaled)
    _, neigh_idx = nn.kneighbors(co_key_scaled)
    return train_cluster_idx[neigh_idx[:, 0]]


def route_R1(co_h_g: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Cold h_g_cold -> argmin_c ||h_g - centroid_c||_2."""
    d = ((co_h_g[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1).astype(np.int64)


def run_routing(
    routing: str,
    co: dict,
    cb: dict,
) -> tuple[np.ndarray, dict]:
    """Returns (cold_cluster [N], routing_diag)."""
    if routing == "R0":
        cold_cluster = route_R0(
            co["key"],
            cb["key_scaler_mean"],
            cb["key_scaler_scale"],
            cb["key_pool_scaled"],
            cb["cluster_idx"].astype(np.int64),
        )
    elif routing == "R1":
        cold_cluster = route_R1(co["h_g"], cb["codebook"])
    else:
        raise ValueError(f"unknown routing: {routing}")
    M = cb["codebook"].shape[0]
    usage_counts = np.bincount(cold_cluster, minlength=M)
    diag = {
        "n_clusters_used": int((usage_counts > 0).sum()),
        "usage_min": int(usage_counts.min()),
        "usage_max": int(usage_counts.max()),
        "usage_mean": float(usage_counts.mean()),
    }
    return cold_cluster, diag


def evaluate_routing(
    routing: str,
    co: dict,
    cb: dict,
) -> dict:
    cold_cluster, route_diag = run_routing(routing, co, cb)
    offsets = cb["offsets"]  # [M, 24] z-norm
    cluster_offset = offsets[cold_cluster]  # [N, 24]

    base = metrics_z_to_kw(co["y_true_z"], co["y_hat_z"], co["mean"], co["std"])

    op_results = {}
    for op_name, op in OPERATING_POINTS.items():
        g = gauss_template(co["pred_hr"], co["pred_amp"], sigma=op["sigma"])
        corrected_z = (
            co["y_hat_z"]
            + op["alpha_v0"] * cluster_offset
            + op["alpha_w1"] * g
        ).astype(np.float32)
        op_results[op_name] = {
            "sigma": op["sigma"],
            "alpha_v0": op["alpha_v0"],
            "alpha_w1": op["alpha_w1"],
            "metrics": metrics_z_to_kw(co["y_true_z"], corrected_z, co["mean"], co["std"]),
        }

    cold_true_hr = co["y_true_z"].argmax(axis=1)
    aux_diag = {
        "top1": float((co["pred_hr"] == cold_true_hr).mean()),
        "within_1h": float((np.abs(co["pred_hr"] - cold_true_hr) <= 1).mean()),
        "within_2h": float((np.abs(co["pred_hr"] - cold_true_hr) <= 2).mean()),
    }
    return {
        "routing": routing,
        "n_cold_windows": int(co["y_true_z"].shape[0]),
        "n_cold_apts": int(len(np.unique(co["apt"]))),
        "baseline": base,
        "operating_points": op_results,
        "routing_diagnostics": route_diag,
        "aux_diagnostics": aux_diag,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Cold zero-shot evaluation: R0/R1 × HR-pres/PAPE-aggr.")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--routings", nargs="+", default=["R0", "R1"], choices=["R0", "R1"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--stride", type=int, default=24)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    seed_root = V02_OUT_ROOT / f"seed{args.seed}"
    ckpt = seed_root / "T2" / "best.pt"
    cb_path = seed_root / "codebook.npz"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing {ckpt}; run 02_train_arms.py --seed {args.seed} --arms T2 first.")
    if not cb_path.exists():
        raise FileNotFoundError(f"missing {cb_path}; run 03_fit_codebook.py --seed {args.seed} first.")

    cold_apts = load_v02_split(args.seed)["cold"]
    print(f"[setup] seed={args.seed}; {len(cold_apts)} cold apts; routings={args.routings}")
    print(f"[setup] device={DEVICE}; seed_root={seed_root}")

    model = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))

    co = gather_cold(cold_apts, model, batch=args.batch, stride=args.stride)
    print(
        f"[data] {len(np.unique(co['apt']))} cold apts present, "
        f"{co['y_true_z'].shape[0]} cold windows"
    )

    cb_npz = np.load(cb_path)
    cb = {k: cb_npz[k] for k in cb_npz.files}

    for routing in args.routings:
        print(f"\n========== {routing} ==========")
        result = evaluate_routing(routing, co, cb)
        result["seed"] = int(args.seed)
        result["split_version"] = "v02"
        out_path = seed_root / f"coldstart_{routing}.json"
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
        base = result["baseline"]
        print(
            f"  baseline    PAPE={base['pape']:.2f}  HR@1={base['hr@1']:.1f}  "
            f"HR@2={base['hr@2']:.1f}  MAE={base['mae']:.4f}"
        )
        for op_name, op in result["operating_points"].items():
            ops_m = op["metrics"]
            ratio = ops_m["pape"] / base["pape"] if base["pape"] > 0 else float("nan")
            print(
                f"  {op_name:<16} PAPE={ops_m['pape']:.2f}  HR@1={ops_m['hr@1']:.1f}  "
                f"HR@2={ops_m['hr@2']:.1f}  MAE={ops_m['mae']:.4f}  (ratio={ratio:.3f})"
            )
        rd = result["routing_diagnostics"]
        print(
            f"  routing_diag  used={rd['n_clusters_used']}/32  "
            f"usage min/max={rd['usage_min']}/{rd['usage_max']}  mean={rd['usage_mean']:.1f}"
        )
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
