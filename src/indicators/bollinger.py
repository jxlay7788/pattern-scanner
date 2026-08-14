"""Bollinger Bands (20-period SMA +/- 2 standard deviations)."""

import pandas as pd


def compute_bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def bollinger_signal(price: float, upper: float, lower: float) -> int:
    """+1 if price closed below the lower band (oversold), -1 if above the upper band
    (overbought), 0 otherwise."""
    if pd.isna(upper) or pd.isna(lower):
        return 0
    if price < lower:
        return 1
    if price > upper:
        return -1
    return 0
