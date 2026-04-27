"""Metrics — ported from Peak_Analysis/src/peak_analysis/metrics.py.

Definitions kept identical so v11 numbers are directly comparable to v10's
reported values:
    - PAPE   : peak absolute percentage error (%) over horizon-max
    - HR@k   : hit rate (%) at peak position with tolerance k
    - MSE/MAE: standard window-mean error
"""

from __future__ import annotations

import numpy as np


def _ensure_2d(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_pred = y_pred.reshape(1, -1)
    elif y_true.ndim > 2:
        B, T, C = y_true.shape
        y_true = y_true.transpose(0, 2, 1).reshape(B * C, T)
        y_pred = y_pred.transpose(0, 2, 1).reshape(B * C, T)
    return y_true, y_pred


def compute_pape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _ensure_2d(y_true, y_pred)
    peak_true = np.max(y_true, axis=-1)
    peak_pred = np.max(y_pred, axis=-1)
    valid = np.abs(peak_true) > 1e-5
    if not valid.any():
        return 0.0
    return float(
        np.mean(np.abs(peak_pred[valid] - peak_true[valid]) / np.abs(peak_true[valid])) * 100.0
    )


def compute_hr(y_true: np.ndarray, y_pred: np.ndarray, tol: int = 1) -> float:
    y_true, y_pred = _ensure_2d(y_true, y_pred)
    peak_true_vals = np.max(y_true, axis=-1)
    valid = np.abs(peak_true_vals) > 1e-5
    if not valid.any():
        return 0.0
    yt = y_true[valid]
    yp = y_pred[valid]
    true_argmax = np.argmax(yt, axis=-1)
    pred_argmax = np.argmax(yp, axis=-1)
    return float(np.mean(np.abs(true_argmax - pred_argmax) <= tol) * 100.0)


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _ensure_2d(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _ensure_2d(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def seven_axis_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """7-axis summary for direct comparison with v10 reports."""
    return {
        "pape": compute_pape(y_true, y_pred),
        "hr@1": compute_hr(y_true, y_pred, tol=1),
        "hr@2": compute_hr(y_true, y_pred, tol=2),
        "mse": compute_mse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "n_windows": int(np.asarray(y_true).reshape(-1, np.asarray(y_true).shape[-1]).shape[0]),
        "horizon": int(np.asarray(y_true).shape[-1]),
    }
