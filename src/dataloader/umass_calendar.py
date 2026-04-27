"""Calendar-augmented dataset.

Each window returns (x [96], y [24], cal_features [F]).
F=2: sin/cos of forecast-start hour-of-day.
F=4: + sin/cos of forecast-start day-of-week.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import HORIZON, INPUT_SIZE


class HouseholdDatasetCal(Dataset):
    def __init__(self, series_pd: pd.Series, mean: float, std: float,
                 seq_len: int = INPUT_SIZE, pred_len: int = HORIZON,
                 stride: int = 1, use_dow: bool = True) -> None:
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mean = float(mean)
        self.std = float(std) if std > 1e-8 else 1.0
        self.use_dow = use_dow

        values = series_pd.values.astype(np.float32)
        timestamps = series_pd.index
        self.values = (values - self.mean) / self.std

        # forecast-start hour & dow per index
        forecast_start_idx = np.arange(len(values))
        # for window starting at i, forecast starts at i+seq_len
        ts_array = timestamps.to_numpy()
        self.hour_of_day = pd.DatetimeIndex(ts_array).hour.to_numpy().astype(np.int32)
        self.dow = pd.DatetimeIndex(ts_array).dayofweek.to_numpy().astype(np.int32)

        total = seq_len + pred_len
        self.indices = list(range(0, len(values) - total + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def n_cal(self) -> int:
        return 4 if self.use_dow else 2

    def __getitem__(self, idx: int):
        start = self.indices[idx]
        fc_start_idx = start + self.seq_len    # forecast start within full series
        x = self.values[start: start + self.seq_len]
        y = self.values[fc_start_idx: fc_start_idx + self.pred_len]
        h = float(self.hour_of_day[fc_start_idx])
        d = float(self.dow[fc_start_idx])
        cal = [np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)]
        if self.use_dow:
            cal += [np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(cal, dtype=torch.float32),
        )
