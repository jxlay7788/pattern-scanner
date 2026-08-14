"""MACD (12/26/9 EMA) with crossover-based signal."""

import pandas as pd


def compute_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def macd_signal(macd_line: pd.Series, signal_line: pd.Series) -> int:
    """+1 on a bullish crossover (MACD crosses above signal) within the last session,
    -1 on a bearish crossover, 0 otherwise."""
    if len(macd_line) < 2:
        return 0
    prev_diff = macd_line.iloc[-2] - signal_line.iloc[-2]
    curr_diff = macd_line.iloc[-1] - signal_line.iloc[-1]
    if pd.isna(prev_diff) or pd.isna(curr_diff):
        return 0
    if prev_diff <= 0 and curr_diff > 0:
        return 1
    if prev_diff >= 0 and curr_diff < 0:
        return -1
    return 0
