"""Foundation-model zero-shot wrappers for v04 G3.

These wrappers expose a project-uniform interface so the v04
``03_fm_zero_shot.py`` script can call any FM with the same line:

    forecaster = ChronosForecaster(...)              # or TimesFMForecaster
    y_kw = forecaster.forecast(x_kw)                 # x_kw, y_kw both kW

- input ``x_kw``  : ``np.ndarray`` of shape ``[B, L=96]`` in kW (raw,
  unnormalised; the wrapper handles its own normalisation).
- output ``y_kw`` : ``np.ndarray`` of shape ``[B, H=24]`` in kW (point
  forecast; for distributional models we take the median sample/quantile).

No UMass training. The cold gucha sees identical inputs to v01-v03.
"""

from fm.chronos import ChronosForecaster
from fm.timesfm import TimesFMForecaster

__all__ = ["ChronosForecaster", "TimesFMForecaster"]
