"""Time-series filters for AUV closed-loop logs.

Three stages of the data pipeline:
    raw     -> hampel_filter        -> cleaned
    cleaned -> moving_average / butterworth_lowpass -> smooth

Mirrors the kmc reference ETL notebook (vectorized Hampel via median_filter,
pandas rolling mean for MA, scipy SOS zero-phase for Butterworth).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import median_filter

logger = logging.getLogger(__name__)


def check_outliers_iqr(series: pd.Series) -> bool:
    """Return True if any sample falls outside the [Q1 - 1.5 IQR, Q3 + 1.5 IQR] band."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return bool(((series < lower) | (series > upper)).any())


def _hampel_columns(arr: np.ndarray, window_size: int, n_sigmas: float) -> np.ndarray:
    """Apply a Hampel filter column-wise on a 2D numeric array."""
    k = 1.4826  # scaling so MAD estimates std for a Gaussian
    kernel = (2 * window_size + 1, 1)
    rolling_median = median_filter(arr, size=kernel, mode="nearest")
    diff = np.abs(arr - rolling_median)
    mad = median_filter(diff, size=kernel, mode="nearest")
    threshold = n_sigmas * k * mad
    outliers = diff > threshold
    cleaned = arr.copy()
    cleaned[outliers] = rolling_median[outliers]
    return cleaned


def hampel_filter(
    df: pd.DataFrame,
    window_size: int = 20,
    n_sigmas: float = 20.0,
    selective: bool = True,
) -> pd.DataFrame:
    """Outlier removal via the vectorized Hampel filter.

    Args:
        df: Input DataFrame. Non-numeric columns are passed through unchanged.
        window_size: Half-window for the rolling median (full window = 2w + 1).
        n_sigmas: Threshold in MAD-derived standard deviations.
        selective: If True, only filter numeric columns flagged by `check_outliers_iqr`.
                   If False, filter every numeric column.

    Returns:
        A new DataFrame with the same shape and column order.
    """
    out = df.copy()
    numeric_cols = out.select_dtypes(include=np.number).columns
    if "timestamp" in numeric_cols:
        numeric_cols = numeric_cols.drop("timestamp")

    target_cols = []
    for col in numeric_cols:
        if not selective or check_outliers_iqr(out[col]):
            target_cols.append(col)

    if not target_cols:
        return out

    arr = out[target_cols].to_numpy(dtype=float)
    cleaned = _hampel_columns(arr, window_size=window_size, n_sigmas=n_sigmas)
    out[target_cols] = cleaned
    logger.debug(
        "Hampel applied to %d/%d numeric cols (window=%d, n_sigmas=%g)",
        len(target_cols), len(numeric_cols), window_size, n_sigmas,
    )
    return out


def moving_average(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """Centered moving-average smoother on numeric columns.

    Non-numeric columns and `timestamp` are passed through unchanged.
    """
    out = df.copy()
    numeric_cols = out.select_dtypes(include=np.number).columns
    if "timestamp" in numeric_cols:
        numeric_cols = numeric_cols.drop("timestamp")
    out[numeric_cols] = (
        out[numeric_cols]
        .rolling(window=window_size, center=True, min_periods=1)
        .mean()
    )
    return out


def butterworth_lowpass(
    df: pd.DataFrame,
    fc: float,
    fs: float,
    order: int = 4,
) -> pd.DataFrame:
    """Zero-phase Butterworth low-pass filter on numeric columns.

    Args:
        fc: Cut-off frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order.
    """
    sos = signal.butter(order, fc, btype="low", fs=fs, output="sos")
    out = df.copy()
    numeric_cols = out.select_dtypes(include=np.number).columns
    if "timestamp" in numeric_cols:
        numeric_cols = numeric_cols.drop("timestamp")
    for col in numeric_cols:
        out[col] = signal.sosfiltfilt(sos, out[col].to_numpy(dtype=float))
    return out