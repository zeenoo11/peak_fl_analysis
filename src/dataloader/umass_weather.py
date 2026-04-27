"""Calendar + Weather augmented dataset.

Returns (x, y, cal+weather features) per window. Weather features are
hourly time-aligned; we lookup by Unix timestamp.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import HORIZON, INPUT_SIZE, RAW_DIR

WEATHER_DIR = RAW_DIR / "weather"

WEATHER_COLS = ["temperature", "humidity", "apparentTemperature", "cloudCover"]
N_WEATHER = len(WEATHER_COLS)


def _load_weather(year: str = "2016") -> pd.DataFrame:
    path = WEATHER_DIR / f"apartment{year}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    w = pd.read_csv(path)
    w["dt"] = pd.to_datetime(w["time"], unit="s")
    w = w.set_index("dt").sort_index()
    # Fill missing with column median
    for c in WEATHER_COLS:
        if c in w.columns:
            w[c] = w[c].fillna(w[c].median())
        else:
            w[c] = 0.0
    return w[WEATHER_COLS]


class HouseholdDatasetCalWeather(Dataset):
    """Sliding windows + 4-d calendar + 4-d weather (z-normalized) features.

    Calendar features: sin/cos of forecast-start hour-of-day + sin/cos of
    forecast-start day-of-week (4 dims).
    Weather features: temperature, humidity, apparentTemperature, cloudCover
    at forecast-start, z-normalized over the year (4 dims).
    Total cal_feat dim = 8.
    """

    def __init__(self, series_pd: pd.Series, mean: float, std: float,
                 weather_df: pd.DataFrame, weather_mean: np.ndarray, weather_std: np.ndarray,
                 seq_len: int = INPUT_SIZE, pred_len: int = HORIZON,
                 stride: int = 1) -> None:
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mean = float(mean)
        self.std = float(std) if std > 1e-8 else 1.0
        self.weather_mean = weather_mean
        self.weather_std = np.where(weather_std > 1e-8, weather_std, 1.0)

        values = series_pd.values.astype(np.float32)
        self.values = (values - self.mean) / self.std
        ts = pd.DatetimeIndex(series_pd.index.to_numpy())
        self.hour_of_day = ts.hour.to_numpy().astype(np.int32)
        self.dow = ts.dayofweek.to_numpy().astype(np.int32)
        # Align weather to series timestamps via reindex (forward-fill nearest)
        w_aligned = weather_df.reindex(ts, method="nearest")
        self.weather = w_aligned[WEATHER_COLS].to_numpy().astype(np.float32)
        # z-normalize per column using year stats
        self.weather = (self.weather - self.weather_mean[None, :]) / self.weather_std[None, :]

        total = seq_len + pred_len
        self.indices = list(range(0, len(values) - total + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def n_cal(self) -> int:
        return 4 + N_WEATHER

    def __getitem__(self, idx: int):
        start = self.indices[idx]
        fc_start = start + self.seq_len
        x = self.values[start: start + self.seq_len]
        y = self.values[fc_start: fc_start + self.pred_len]
        h = float(self.hour_of_day[fc_start])
        d = float(self.dow[fc_start])
        cal_time = [
            np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
            np.sin(2 * np.pi * d / 7),  np.cos(2 * np.pi * d / 7),
        ]
        weather = self.weather[fc_start].tolist()
        cal = np.array(cal_time + weather, dtype=np.float32)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(cal, dtype=torch.float32),
        )


def load_weather_with_stats(year: str = "2016"):
    """Load weather DF + per-column mean/std for z-normalization."""
    w = _load_weather(year)
    mean = w.mean().to_numpy()
    std = w.std().to_numpy()
    return w, mean, std
