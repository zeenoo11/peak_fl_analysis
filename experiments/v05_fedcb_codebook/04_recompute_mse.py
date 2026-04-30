"""Recompute MSE / RMSE for all v04 + v05 methods that have saved state dicts.

Why
---
The existing multiseed_summary.json files store PAPE / HR@1 / HR@2 / MAE
but not MSE / RMSE. MAE is similar across methods (0.40-0.48 kW) so it
does not differentiate well; MSE penalises peak-amplitude outliers more
heavily and is what we want for a presentation table.

Re-forwarding is required because raw cold predictions were not persisted.
This script loads each method's saved state and recomputes all six metrics
(including MSE/RMSE) on the same v02 80:20 cold pool, 3 seeds.

Skipped (no saved state):
    - FM baselines (Chronos-Bolt / Chronos-T5 / TimesFM): zero-shot,
      no state to reload.
    - Local-only NBEATSx: per-cold-apt training, no central state.
    - peakvq_on_fedavg / peakvq_on_fedrep: codebook is not persisted by
      v04 04_peakvq_on_fl.py, would require re-fitting.
    - V5-FedCB-0 (v02 anchor): exists in v02 W_component_results.json
      but those JSONs do not store MSE either; the row stays MAE-only
      with a 'mse: n/a' marker in the summary.

Output:
    outputs/v05_fedcb_codebook/mse_recompute_summary.json
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
from torch.utils.data import DataLoader

from config import HORIZON, OUTPUT_DIR, TRAIN_RATIO
from dataloader.splits import load_v02_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.crossformer import Crossformer
from models.dlinear import DLinear
from models.nbeatsx import MinimalNBEATSx
from models.nbeatsx_aux import NBEATSxAux
from models.nhits import NHITS
from utils.metrics import compute_hr, compute_mae, compute_pape

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V04_OUT = OUTPUT_DIR / "v04_full_baseline_comparison"
V04_FIX = V04_OUT / "09_fix_rerun"
V05_OUT = OUTPUT_DIR / "v05_fedcb_codebook"

SEEDS = [42, 123, 7]


def _gather_cold_arrays(model, cold_apts, model_kind, batch=512, stride=HORIZON):
    """Forward cold apts; return (y_true_z, y_hat_z, mean_arr, std_arr, h_g_or_None)."""
    model.eval()
    t_chunks, p_chunks, m_chunks, s_chunks, h_chunks = [], [], [], [], []
    for apt in cold_apts:
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
        for x, y in loader:
            x_dev = x.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                if model_kind == "nf":
                    y_hat = model(x_dev)
                    h_g = None
                elif model_kind == "minimal":
                    y_hat, hiddens = model(x_dev)
                    h_g = hiddens["h_generic"]
                elif model_kind == "aux":
                    y_hat, hiddens, _ = model(x_dev)
                    h_g = hiddens["h_generic"]
                else:
                    raise ValueError(model_kind)
            y_hat_np = y_hat.float().cpu().numpy()
            t_chunks.append(y.numpy())
            p_chunks.append(y_hat_np)
            m_chunks.append(np.full(len(y), m_, dtype=np.float32))
            s_chunks.append(np.full(len(y), s_, dtype=np.float32))
            if h_g is not None:
                h_chunks.append(h_g.float().cpu().numpy())
    t_z = np.concatenate(t_chunks, axis=0).astype(np.float32)
    p_z = np.concatenate(p_chunks, axis=0).astype(np.float32)
    m_arr = np.concatenate(m_chunks, axis=0).astype(np.float32)
    s_arr = np.concatenate(s_chunks, axis=0).astype(np.float32)
    h_g = np.concatenate(h_chunks, axis=0).astype(np.float32) if h_chunks else None
    return t_z, p_z, m_arr, s_arr, h_g


def _compute_all(t_z, p_z, m_arr, s_arr):
    t_kw = t_z * s_arr[:, None] + m_arr[:, None]
    p_kw = p_z * s_arr[:, None] + m_arr[:, None]
    err = t_kw - p_kw
    mse = float((err ** 2).mean())
    rmse = float(np.sqrt(mse))
    return {
        "pape": float(compute_pape(t_kw, p_kw)),
        "hr@1": float(compute_hr(t_kw, p_kw, tol=1)),
        "hr@2": float(compute_hr(t_kw, p_kw, tol=2)),
        "mae": float(compute_mae(t_kw, p_kw)),
        "mse": mse,
        "rmse": rmse,
        "n_windows": int(t_z.shape[0]),
    }


def run_method(method: str, seed: int, cold_apts):
    if method in ("fedavg", "fedprox", "fedrep", "ditto"):
        sd_path = V04_OUT / f"seed{seed}" / method / "final_state_dict.pt"
        if not sd_path.exists():
            return None, f"missing {sd_path}"
        m = MinimalNBEATSx().to(DEVICE).eval()
        m.load_state_dict(torch.load(sd_path, map_location="cpu", weights_only=False), strict=True)
        t_z, p_z, m_arr, s_arr, _ = _gather_cold_arrays(m, cold_apts, "minimal")
        return _compute_all(t_z, p_z, m_arr, s_arr), "ok"

    if method == "fedproto":
        sd_path = V04_FIX / f"seed{seed}" / "fedproto" / "final_state_dict.pt"
        if not sd_path.exists():
            return None, f"missing {sd_path}"
        m = MinimalNBEATSx().to(DEVICE).eval()
        m.load_state_dict(torch.load(sd_path, map_location="cpu", weights_only=False), strict=True)
        t_z, p_z, m_arr, s_arr, _ = _gather_cold_arrays(m, cold_apts, "minimal")
        return _compute_all(t_z, p_z, m_arr, s_arr), "ok"

    if method == "nf_dlinear":
        sd_path = V04_OUT / f"seed{seed}" / "nf_dlinear" / "best.pt"
        if not sd_path.exists():
            return None, f"missing {sd_path}"
        m = DLinear().to(DEVICE).eval()
        m.load_state_dict(torch.load(sd_path, map_location="cpu", weights_only=False), strict=True)
        t_z, p_z, m_arr, s_arr, _ = _gather_cold_arrays(m, cold_apts, "nf")
        return _compute_all(t_z, p_z, m_arr, s_arr), "ok"

    if method == "nf_nhits_fixed":
        sd_path = V04_FIX / f"seed{seed}" / "nf_nhits_fixed" / "best.pt"
        if not sd_path.exists():
            return None, f"missing {sd_path}"
        m = NHITS().to(DEVICE).eval()
        m.load_state_dict(torch.load(sd_path, map_location="cpu", weights_only=False), strict=True)
        t_z, p_z, m_arr, s_arr, _ = _gather_cold_arrays(m, cold_apts, "nf")
        return _compute_all(t_z, p_z, m_arr, s_arr), "ok"

    if method == "nf_crossformer":
        sd_path = V04_OUT / f"seed{seed}" / "nf_crossformer" / "best.pt"
        if not sd_path.exists():
            return None, f"missing {sd_path}"
        m = Crossformer().to(DEVICE).eval()
        m.load_state_dict(torch.load(sd_path, map_location="cpu", weights_only=False), strict=True)
        t_z, p_z, m_arr, s_arr, _ = _gather_cold_arrays(m, cold_apts, "nf")
        return _compute_all(t_z, p_z, m_arr, s_arr), "ok"

    if method == "fedavg_nbeatsx_aux_fl_only":
        sd_path = V04_FIX / f"seed{seed}" / "fedavg_nbeatsx_aux" / "final_state_dict.pt"
        if not sd_path.exists():
            return None, f"missing {sd_path}"
        m = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
        m.load_state_dict(torch.load(sd_path, map_location="cpu", weights_only=False), strict=True)
        t_z, p_z, m_arr, s_arr, _ = _gather_cold_arrays(m, cold_apts, "aux")
        return _compute_all(t_z, p_z, m_arr, s_arr), "ok"

    if method.startswith("v5_fedcb_K"):
        k = int(method.split("K")[-1])
        sd_path = V04_FIX / f"seed{seed}" / "fedavg_nbeatsx_aux" / "final_state_dict.pt"
        cb_path = V05_OUT / f"seed{seed}" / f"fedcb_K{k}" / "codebook.npz"
        if not sd_path.exists():
            return None, f"missing {sd_path}"
        if not cb_path.exists():
            return None, f"missing {cb_path}"
        m = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
        m.load_state_dict(torch.load(sd_path, map_location="cpu", weights_only=False), strict=True)
        cb_data = np.load(cb_path)
        codebook = cb_data["codebook"]
        offsets = cb_data["offsets"]
        t_z, p_z, m_arr, s_arr, h_g = _gather_cold_arrays(m, cold_apts, "aux")
        d = ((h_g[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=2)
        cluster = d.argmin(axis=1)
        corrected_z = (p_z + 1.0 * offsets[cluster]).astype(np.float32)
        return _compute_all(t_z, corrected_z, m_arr, s_arr), "ok"

    return None, f"unknown method '{method}'"


METHODS = [
    "fedavg", "fedprox", "fedrep", "ditto", "fedproto",
    "nf_dlinear", "nf_nhits_fixed", "nf_crossformer",
    "fedavg_nbeatsx_aux_fl_only",
    "v5_fedcb_K2", "v5_fedcb_K4", "v5_fedcb_K8",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=V05_OUT / "mse_recompute_summary.json")
    args = ap.parse_args()

    print(f"[mse-recompute] device={DEVICE}")
    cold_cache = {s: load_v02_split(s)["cold"] for s in SEEDS}

    results = {}
    skipped = []
    for method in METHODS:
        print(f"=== {method} ===")
        per_seed = {}
        for s in SEEDS:
            t0 = time.time()
            res, status = run_method(method, s, cold_cache[s])
            if res is None:
                print(f"  seed {s}: SKIP — {status}")
                skipped.append((method, s, status))
                continue
            print(f"  seed {s}: PAPE={res['pape']:6.3f}  MAE={res['mae']:.4f}  "
                  f"MSE={res['mse']:.4f}  RMSE={res['rmse']:.4f}  ({time.time() - t0:.1f}s)")
            per_seed[str(s)] = res
        if per_seed:
            agg = {}
            for k in ("pape", "hr@1", "hr@2", "mae", "mse", "rmse"):
                vals = [per_seed[str(s)][k] for s in SEEDS if str(s) in per_seed]
                agg[k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "values": vals,
                }
            results[method] = {"per_seed": per_seed, "agg": agg, "n_seeds": len(per_seed)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "methods": results,
        "skipped": [{"method": m, "seed": s, "reason": r} for (m, s, r) in skipped],
        "seeds": SEEDS,
        "stride": HORIZON,
        "comment": "Recomputed cold metrics including MSE/RMSE (kW^2 / kW). FM baselines and local_only have no saved state and are not in this summary.",
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[done] saved -> {args.out}")


if __name__ == "__main__":
    main()
