"""Cold-side evaluation helpers shared across v02 and v04.

Extracted verbatim from ``experiments/v02_fl_8020_ratio/04_coldstart_eval.py``
to remove the copy-paste duplication that v02's 04, 05, and 06 scripts
were carrying. Behaviour MUST stay bit-identical to that v02 source —
the v02 paper numbers were already produced by it.

Public surface
--------------
- ``OPERATING_POINTS`` : dict, the two v01 §4.2 carry-over op-points.
- ``gather_cold(apts, model, ...)`` : forward NBEATSxAux on every cold
  apt's train segment with warm-start z-norm, return a unified dict.
- ``gauss_template(pred_hr, pred_amp, sigma, length=24)`` : Gaussian peak
  template normalised so the maximum equals ``pred_amp``.
- ``metrics_z_to_kw(true_z, pred_z, mean, std)`` : denormalise z-norm
  forecasts to kW and report PAPE / HR@1 / HR@2 / MAE.
- ``route_R0(...)`` : Key-Route, KEY 1-NN on scaled train pool.
- ``route_R1(co_h_g, codebook)`` : Latent-Route, h_g 1-NN on centroids.

NOTE
----
Other cold-eval scripts (NF / FL / FM baselines in v04) will need
slightly different ``gather_cold`` shapes (e.g. no ``h_g`` for DLinear,
no ``aux`` for non-NBEATSxAux backbones). Those wrappers will live in
``src/eval/`` next to this file; this module is the *NBEATSxAux-specific*
helper that v02 already depends on.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from config import TRAIN_RATIO
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.nbeatsx_aux import NBEATSxAux
from probes.peak_descriptor import extract_key
from utils.metrics import compute_hr, compute_mae, compute_pape

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# v01 §4.2 carry-over operating points.
# Both points keep σ=3.0; cold-side α tuning is forbidden by design
# (would re-introduce v01 §5.4.1 selection bias).
OPERATING_POINTS: dict[str, dict[str, float]] = {
    "HR-preserving":   {"sigma": 3.0, "alpha_v0": 1.0, "alpha_w1": 0.1},
    "PAPE-aggressive": {"sigma": 3.0, "alpha_v0": 1.5, "alpha_w1": 0.5},
}


def gather_cold(
    apts: Iterable[str],
    model: NBEATSxAux,
    batch: int = 256,
    stride: int = 24,
    *,
    verbose_skips: bool = True,
) -> dict[str, np.ndarray]:
    """Forward an NBEATSxAux model over every cold apt's train segment.

    Behaviour matches ``experiments/v02_fl_8020_ratio/04_coldstart_eval.py``
    exactly:
    - per-apt warm-start z-norm (own first ``TRAIN_RATIO`` of the series),
    - sliding windows with ``stride`` (= horizon, non-overlapping),
    - frozen forward (model is expected in eval mode by the caller).

    Returns a dict of numpy arrays of length N (= total cold windows):

    ============  =================  ==========================================
    key           dtype, shape       contents
    ============  =================  ==========================================
    h_g           float32 [N, 64]    NBEATSxAux ``h_generic`` latent
    y_hat_z       float32 [N, 24]    z-norm space base forecast
    y_true_z      float32 [N, 24]    z-norm space ground truth
    pred_amp      float32 [N]        aux head amplitude scalar (z-norm space)
    pred_hr       int64   [N]        aux head argmax over 24 hour-classes
    key           float32 [N, 5]     5-d KEY descriptor (input-only)
    mean          float32 [N]        per-window denorm mean (per-apt const)
    std           float32 [N]        per-window denorm std  (per-apt const)
    apt           object  [N]        source apt name per window
    ============  =================  ==========================================
    """
    h_chunks, yhat_chunks, ytrue_chunks = [], [], []
    amp_chunks, hr_chunks, key_chunks = [], [], []
    mean_chunks, std_chunks, apt_chunks = [], [], []
    for apt in apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            if verbose_skips:
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
    """Gaussian peak template normalised so ``g.max(axis=1) == pred_amp``.

    Mirror of the v01 W1a Gaussian template
    (``experiments/v01_peak_from_latent/09_iter4_mechanisms.py:gauss_template``).
    Used by both the v02 04/05/06 scripts and v04 G5 cross-cell.
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
) -> dict[str, float]:
    """Denormalise z-norm forecasts to kW and report PAPE / HR@1 / HR@2 / MAE.

    All four metrics are bit-exact ports from
    ``Peak_Analysis/src/peak_analysis/metrics.py`` — see ``utils.metrics``.
    """
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
    """Key-Route: cold KEY → scaled 1-NN on train KEY pool → train cluster_idx.

    The cold side does **not** re-fit the KEY scaler — it loads the
    train-side ``mean`` / ``scale`` exactly as 03_fit_codebook saved them,
    so the routing decision is deterministic w.r.t. the codebook artefacts.
    """
    co_key_scaled = (co_key - key_scaler_mean) / key_scaler_scale
    nn = NearestNeighbors(n_neighbors=1).fit(key_pool_scaled)
    _, neigh_idx = nn.kneighbors(co_key_scaled)
    return train_cluster_idx[neigh_idx[:, 0]]


def route_R1(co_h_g: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Latent-Route: ``argmin_c ||h_g_cold - codebook[c]||_2`` (raw Euclidean)."""
    d = ((co_h_g[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1).astype(np.int64)
