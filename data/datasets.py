"""Small, leakage-safe dataset loaders used by the first experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class NormalizationStats:
    mean: float
    std: float


@dataclass(frozen=True)
class SeriesSplits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    stats: NormalizationStats


def _standardize_splits(
    train: np.ndarray, val: np.ndarray, test: np.ndarray
) -> SeriesSplits:
    mean = float(np.mean(train))
    std = float(np.std(train))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    stats = NormalizationStats(mean=mean, std=std)
    normalize = lambda values: ((values - mean) / std).astype(np.float32)
    return SeriesSplits(normalize(train), normalize(val), normalize(test), stats)


def generate_synthetic(
    kind: str,
    total_length: int = 12000,
    seed: int = 7,
    noise_std: float = 0.05,
    period: float = 32.0,
    ar_coefficients: tuple[float, ...] = (0.95, -0.2),
) -> np.ndarray:
    """Generate a deterministic contiguous univariate series."""

    if total_length < 64:
        raise ValueError("total_length must be at least 64")
    rng = np.random.default_rng(seed)
    kind = kind.lower()
    if kind in {"sine", "synthetic_sine"}:
        time = np.arange(total_length, dtype=np.float32)
        signal = np.sin(2.0 * np.pi * time / period)
        signal += 0.25 * np.sin(2.0 * np.pi * time / (period * 3.7) + 0.4)
        return (signal + rng.normal(0.0, noise_std, total_length)).astype(np.float32)
    if kind not in {"ar", "synthetic_ar"}:
        raise ValueError(f"unknown synthetic kind: {kind}")

    order = len(ar_coefficients)
    values = np.zeros(total_length, dtype=np.float32)
    values[:order] = rng.normal(0.0, 0.2, order)
    for index in range(order, total_length):
        previous = values[index - order : index][::-1]
        values[index] = float(np.dot(np.asarray(ar_coefficients), previous))
        values[index] += float(rng.normal(0.0, noise_std))
    return values


def load_ettm1(root: str | Path, target: str = "OT") -> SeriesSplits:
    """Load ETTm1 using the standard 12/4/4 month contiguous split."""

    root = Path(root)
    candidates = [root / "ETTm1.csv", root / "ETT-small" / "ETTm1.csv"]
    csv_path = next((path for path in candidates if path.is_file()), None)
    if csv_path is None:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"ETTm1.csv not found; searched: {searched}")

    frame = pd.read_csv(csv_path)
    if target not in frame.columns:
        raise ValueError(f"target {target!r} not found in columns: {list(frame.columns)}")
    values = frame[target].to_numpy(dtype=np.float32)
    # ETTm1 has 15-minute samples. These boundaries match the common benchmark split.
    train_end = 12 * 30 * 24 * 4
    val_end = train_end + 4 * 30 * 24 * 4
    if len(values) <= val_end:
        raise ValueError(f"ETTm1 is too short for the standard split: {len(values)} rows")
    return _standardize_splits(values[:train_end], values[train_end:val_end], values[val_end:])


def load_ett(root: str | Path, name: str, target: str = "OT") -> SeriesSplits:
    """Load ETTh1/ETTh2/ETTm1/ETTm2 with the standard 12/4/4 month split."""

    root = Path(root)
    wanted = f"{name}.csv".lower()
    candidates = [
        path
        for directory in (root, root / "ETT-small")
        if directory.is_dir()
        for path in directory.iterdir()
        if path.is_file() and path.name.lower() == wanted
    ]
    if not candidates:
        raise FileNotFoundError(f"{name}.csv not found under {root}")
    csv_path = candidates[0]
    frame = pd.read_csv(csv_path)
    if target not in frame.columns:
        raise ValueError(f"target {target!r} not found in columns: {list(frame.columns)}")
    values = frame[target].to_numpy(dtype=np.float32)
    is_minute = name.endswith("m1") or name.endswith("m2")
    per_month = 30 * 24 * (4 if is_minute else 1)
    train_end = 12 * per_month
    val_end = train_end + 4 * per_month
    if len(values) <= val_end:
        raise ValueError(f"{name} is too short for the standard split: {len(values)} rows")
    return _standardize_splits(values[:train_end], values[train_end:val_end], values[val_end:])


class WindowDataset(Dataset):
    """Contiguous forecasting windows with optional deterministic subsampling."""

    def __init__(
        self,
        series: np.ndarray,
        context_len: int,
        horizon: int,
        max_windows: Optional[int] = None,
        seed: int = 7,
    ) -> None:
        if series.ndim != 1:
            raise ValueError(f"expected a univariate series, got shape {series.shape}")
        if context_len <= 0 or horizon <= 0:
            raise ValueError("context_len and horizon must be positive")
        available = len(series) - context_len - horizon + 1
        if available <= 0:
            raise ValueError(
                f"series length {len(series)} cannot provide context={context_len}, horizon={horizon}"
            )
        self.series = np.asarray(series, dtype=np.float32)
        self.context_len = context_len
        self.horizon = horizon
        if max_windows is None or max_windows >= available:
            self.starts = np.arange(available, dtype=np.int64)
        else:
            rng = np.random.default_rng(seed)
            self.starts = np.sort(rng.choice(available, size=max_windows, replace=False))

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        split = start + self.context_len
        context = self.series[start:split]
        future = self.series[split : split + self.horizon]
        return torch.from_numpy(context.copy()), torch.from_numpy(future.copy())


def build_dataset(
    name: str,
    context_len: int,
    horizon: int,
    seed: int = 7,
    total_length: int = 12000,
    ett_root: str | Path = "/public/chenjiahui/波数据时序基座大模型/UniTS-main/dataset/ETT-small",
    target: str = "OT",
    max_train_windows: Optional[int] = 2048,
    max_val_windows: Optional[int] = 512,
    max_test_windows: Optional[int] = 512,
) -> tuple[WindowDataset, WindowDataset, WindowDataset, NormalizationStats]:
    """Build train/validation/test windows without cross-split leakage."""

    name = name.lower()
    if name in {"synthetic_ar", "ar"}:
        raw = generate_synthetic("ar", total_length=total_length, seed=seed)
        split_points = (int(0.70 * len(raw)), int(0.85 * len(raw)))
        splits = _standardize_splits(raw[: split_points[0]], raw[split_points[0] : split_points[1]], raw[split_points[1] :])
    elif name in {"synthetic_sine", "sine"}:
        raw = generate_synthetic("sine", total_length=total_length, seed=seed)
        split_points = (int(0.70 * len(raw)), int(0.85 * len(raw)))
        splits = _standardize_splits(raw[: split_points[0]], raw[split_points[0] : split_points[1]], raw[split_points[1] :])
    elif name == "ettm1":
        splits = load_ettm1(ett_root, target=target)
    elif name in {"etth1", "etth2"}:
        splits = load_ett(ett_root, name, target=target)
    else:
        raise ValueError(f"unknown dataset: {name}")

    common = dict(context_len=context_len, horizon=horizon, seed=seed)
    train = WindowDataset(splits.train, max_windows=max_train_windows, **common)
    val = WindowDataset(splits.val, max_windows=max_val_windows, **common)
    test = WindowDataset(splits.test, max_windows=max_test_windows, **common)
    return train, val, test, splits.stats
