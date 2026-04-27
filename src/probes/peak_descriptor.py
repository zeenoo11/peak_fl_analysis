"""KEY descriptor extractor for KV-VQ.

KEY is computed from the input 96h window only — no future leakage, no
encoder dependency. Cold households can compute their own KEY identically.

Default 5-d KEY:
    [input_max, input_argmax_norm, daily_mean, daily_std, last24_max]
"""

from __future__ import annotations

import numpy as np


def extract_key(x: np.ndarray) -> np.ndarray:
    """x: [B, 96] z-normalized input. Returns [B, 5]."""
    if x.ndim == 1:
        x = x[None, :]
    return np.stack(
        [
            x.max(axis=1),
            x.argmax(axis=1) / 96.0,
            x.mean(axis=1),
            x.std(axis=1),
            x[:, -24:].max(axis=1),
        ],
        axis=1,
    )
