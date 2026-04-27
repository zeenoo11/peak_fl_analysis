"""TimesFM zero-shot wrapper (Das et al., 2024).

Reference
---------
A. Das, W. Kong, R. Sen, Y. Zhou, "A decoder-only foundation model for
time-series forecasting", ICML 2024. https://arxiv.org/abs/2310.10688
Package: ``timesfm>=1.3.0`` (pinned in pyproject.toml).

Architecture choice
-------------------
TimesFM ships two backends:
- **JAX** (default) — fastest on TPU; not used here (project is PyTorch).
- **PyTorch** — selected automatically when JAX is unavailable; this is
  what our environment uses (verified at smoke-test: "Loaded PyTorch
  TimesFM, likely because python version is 3.11.13").

We pick the 1.0 release weights (``google/timesfm-1.0-200m-pytorch``) as
the v04 default because it has the most stable license + smallest
download. Larger 2.0 weights are available at
``google/timesfm-2.0-500m-pytorch`` and can be opted into by passing a
different ``checkpoint_repo``.

Interface
---------
    fc = TimesFMForecaster()
    y_kw = fc.forecast(x_kw)   # x_kw, y_kw both numpy [B, L=96] / [B, H=24] in kW

TimesFM's API takes a list of 1D arrays (per-series) and a ``freq``
list (0=high, 1=med, 2=low frequency); UMass hourly is "high frequency"
in TimesFM's taxonomy → ``freq=0``. Output is a point forecast (no
sampling), so unlike Chronos there is no median-collapse step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from config import HORIZON, INPUT_SIZE


def _resolve_backend(device: str) -> str:
    """'cuda'/'cpu'/'auto' -> 'gpu'/'cpu' (timesfm's vocabulary)."""
    if device == "auto":
        return "gpu" if torch.cuda.is_available() else "cpu"
    return "gpu" if device == "cuda" else device


@dataclass
class TimesFMForecaster:
    """Project-uniform TimesFM wrapper (PyTorch backend).

    Args:
        checkpoint_repo:   HF Hub repo ID. Defaults to
                           ``google/timesfm-1.0-200m-pytorch``.
        context_len:       Input window length. Defaults to ``config.INPUT_SIZE`` = 96.
                           Note: TimesFM's published context is much longer (up to
                           512); we pass 96 to keep the cold input identical to v01-v03.
        horizon_len:       Forecast horizon. Defaults to ``config.HORIZON`` = 24.
        device:            ``'cuda'``, ``'cpu'``, or ``'auto'`` (default).
        per_core_batch_size: Internal batch size for the TimesFM forward.
                           1 is fine for our small per-apt batches.
        freq:              TimesFM frequency code. ``0`` = high frequency
                           (hourly / sub-daily); ``1`` = medium; ``2`` = low.
                           UMass hourly → 0.
    """

    checkpoint_repo: str = "google/timesfm-1.0-200m-pytorch"
    context_len: int = INPUT_SIZE
    horizon_len: int = HORIZON
    device: str = "auto"
    per_core_batch_size: int = 1
    freq: int = 0

    def __post_init__(self) -> None:
        # Lazy-import keeps this module's import cost flat when the FM
        # axis is not exercised.
        import timesfm

        backend = _resolve_backend(self.device)
        self._tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=self.per_core_batch_size,
                horizon_len=self.horizon_len,
                context_len=self.context_len,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=self.checkpoint_repo),
        )

    def forecast(self, x_kw: np.ndarray) -> np.ndarray:
        """Zero-shot forecast.

        Input  : ``x_kw`` shape ``[B, L]`` (kW, no z-norm).
        Output : ``y_kw`` shape ``[B, H]`` (kW point forecast).
        """
        if x_kw.ndim != 2:
            raise ValueError(f"TimesFMForecaster expects [B, L], got shape {x_kw.shape}")
        # TimesFM expects a list of 1-D series + a list of per-series freqs.
        series_list = [x_kw[i].astype(np.float32) for i in range(x_kw.shape[0])]
        freq_list = [self.freq] * x_kw.shape[0]
        point_forecast, _quantile_forecast = self._tfm.forecast(series_list, freq=freq_list)
        return np.asarray(point_forecast, dtype=np.float32)   # [B, H]
