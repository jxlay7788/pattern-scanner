"""Relative Strength Index (Wilder's smoothing)."""

import pandas as pd

OVERSOLD = 30
OVERBOUGHT = 70


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.where(avg_loss != 0, 100.0)


def rsi_signal(rsi_value: float) -> int:
    """+1 oversold (bullish), -1 overbought (bearish), 0 neutral."""
    if pd.isna(rsi_value):
        return 0
    if rsi_value < OVERSOLD:
        return 1
    if rsi_value > OVERBOUGHT:
        return -1
    return 0
