"""Chronos zero-shot wrapper (Ansari et al., 2024).

Reference
---------
A. F. Ansari et al., "Chronos: Learning the Language of Time Series",
TMLR 2024. https://arxiv.org/abs/2403.07815
Package: ``chronos-forecasting>=2.2.2`` (pinned in pyproject.toml).

Two pipeline families are supported (selected by ``model_id``):

- **Chronos T5** (``amazon/chronos-t5-{tiny,mini,small,base,large}``) —
  autoregressive token language model. Returns ``[B, num_samples, H]``;
  we collapse to a point forecast via the **median across samples**
  (the standard Chronos-T5 point-forecast convention).
- **Chronos-Bolt** (``amazon/chronos-bolt-{tiny,small,base}``) —
  patch-based regressor returning **9 quantiles** ``[B, 9, H]``. We
  collapse to a point forecast via the **median quantile (index 4 ↔
  q=0.5)**, matching the Chronos-Bolt convention.

Both families share the same ``predict(inputs=, prediction_length=)``
API in ``chronos>=2.2.2``.

Interface
---------
    fc = ChronosForecaster(model_id='amazon/chronos-bolt-small')
    y_kw = fc.forecast(x_kw)   # x_kw, y_kw both numpy [B, L=96] / [B, H=24] in kW

The wrapper does **no per-apt z-norm** — Chronos has its own internal
normalisation (mean-scaling) and applies it identically to every input.
Forecasts come out in the same input scale (kW), so they can be fed
straight to ``utils.metrics`` without further rescaling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from config import HORIZON


def _resolve_device(device: str) -> str:
    """'cuda'/'cpu'/'auto' -> the actual device string."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@dataclass
class ChronosForecaster:
    """Project-uniform Chronos wrapper (T5 or Bolt family).

    Args:
        model_id:           HF Hub repo ID. Defaults to ``amazon/chronos-bolt-small``
                            (small + Bolt = fast, no sample noise).
        prediction_length:  Forecast horizon. Defaults to ``config.HORIZON`` = 24.
        device:             ``'cuda'``, ``'cpu'``, or ``'auto'`` (default).
        torch_dtype:        torch dtype for the model. ``None`` → default (fp32).
        num_samples:        Only used for T5 family. Defaults to 20 (Chronos default).
    """

    model_id: str = "amazon/chronos-bolt-small"
    prediction_length: int = HORIZON
    device: str = "auto"
    torch_dtype: torch.dtype | None = None
    num_samples: int = 20

    def __post_init__(self) -> None:
        # Lazy-import keeps this module's import cost flat when the FM
        # axis is not exercised (e.g. v02 reruns).
        from chronos import BaseChronosPipeline

        device = _resolve_device(self.device)
        kwargs: dict = {"device_map": device}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        # BaseChronosPipeline auto-dispatches to ChronosPipeline (T5) or
        # ChronosBoltPipeline (Bolt) based on the repo's config.
        self._pipeline = BaseChronosPipeline.from_pretrained(self.model_id, **kwargs)
        self._is_bolt = "bolt" in self.model_id.lower()

    def forecast(self, x_kw: np.ndarray) -> np.ndarray:
        """Zero-shot forecast.

        Input  : ``x_kw`` shape ``[B, L]`` (kW, no z-norm).
        Output : ``y_kw`` shape ``[B, H]`` (kW point forecast).
        """
        if x_kw.ndim != 2:
            raise ValueError(f"ChronosForecaster expects [B, L], got shape {x_kw.shape}")
        ctx = torch.from_numpy(x_kw.astype(np.float32))

        # Chronos>=2.2.2 keyword: ``inputs`` (was ``context`` in v1).
        if self._is_bolt:
            # Bolt returns [B, 9 quantiles, H]. Take the median quantile (idx 4).
            fc = self._pipeline.predict(inputs=ctx, prediction_length=self.prediction_length)
            arr = fc.detach().cpu().numpy()        # [B, 9, H]
            return arr[:, 4, :]                    # [B, H]
        else:
            # T5 returns [B, num_samples, H]. Take the median over samples.
            fc = self._pipeline.predict(
                inputs=ctx,
                prediction_length=self.prediction_length,
                num_samples=self.num_samples,
            )
            arr = fc.detach().cpu().numpy()        # [B, num_samples, H]
            return np.median(arr, axis=1)          # [B, H]
