"""UMass Smart* hourly load loader.

Origin: ported from Peak_Analysis/src/peak_analysis/{data_loader,community}.py.
Stripped to the minimum needed for v11 NBEATSx + KV-VQ work.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    HORIZON,
    INPUT_SIZE,
    TRAIN_RATIO,
    UMASS_DIR,
    VAL_RATIO,
)

logger = logging.getLogger(__name__)


def _load_clean_minute_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, header=None, names=["timestamp", "load"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "load"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")

    series = df.set_index("timestamp")["load"].asfreq("min")
    if series.isnull().any():
        series = series.interpolate(limit_direction="both")
    return series


def list_available_apartments(year: str = "2016") -> list[str]:
    year_dir = UMASS_DIR / year
    if not year_dir.exists():
        raise FileNotFoundError(year_dir)

    def _key(stem: str) -> int:
        return int("".join(ch for ch in stem if ch.isdigit()) or "0")

    stems = sorted(
        (p.stem.replace(f"_{year}", "") for p in year_dir.glob(f"Apt*_{year}.csv")),
        key=_key,
    )
    return stems


def load_apartment_hourly(apt_name: str, year: str = "2016") -> pd.Series:
    """Load one household, resampled to hourly mean (kW)."""
    path = UMASS_DIR / year / f"{apt_name}_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return _load_clean_minute_series(path).resample("h").mean()


class HouseholdDataset(Dataset):
    """Sliding-window dataset for one household.

    Args:
        series:   raw hourly values (kW), shape (T,).
        mean:     z-score mean (computed on train segment).
        std:      z-score std.
        seq_len:  input window length.
        pred_len: forecast horizon.
        stride:   sliding stride (1 for train/val, pred_len for test).
    """

    def __init__(
        self,
        series: np.ndarray,
        mean: float,
        std: float,
        seq_len: int = INPUT_SIZE,
        pred_len: int = HORIZON,
        stride: int = 1,
    ) -> None:
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mean = float(mean)
        self.std = float(std) if std > 1e-8 else 1.0

        values = series.astype(np.float32)
        self.raw = values.copy()
        self.values = (values - self.mean) / self.std

        total = seq_len + pred_len
        self.indices = list(range(0, len(values) - total + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        x = self.values[start : start + self.seq_len]
        y = self.values[start + self.seq_len : start + self.seq_len + self.pred_len]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


def make_loaders(
    apt_name: str,
    year: str = "2016",
    batch_size: int = 64,
    train_stride: int = 1,
    test_stride: int = HORIZON,
) -> tuple[DataLoader, DataLoader, DataLoader, float, float]:
    """Build train/val/test loaders for one household using v10 70/10/20 split."""
    series_pd = load_apartment_hourly(apt_name, year)
    values = series_pd.values.astype(np.float32)
    n = len(values)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    mean = float(values[:train_end].mean())
    std = float(values[:train_end].std())

    train_ds = HouseholdDataset(values[:train_end], mean, std, stride=train_stride)
    val_ds = HouseholdDataset(values[train_end:val_end], mean, std, stride=train_stride)
    test_ds = HouseholdDataset(
        values[max(0, val_end - INPUT_SIZE) :], mean, std, stride=test_stride
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, mean, std
