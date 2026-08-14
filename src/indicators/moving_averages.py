"""Golden/death cross on 50-day vs 200-day simple moving averages."""

import pandas as pd


def compute_sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window=window).mean()


def ma_crossover_signal(sma_short: pd.Series, sma_long: pd.Series) -> int:
    """+1 on a golden cross (short crosses above long) in the last session,
    -1 on a death cross, 0 otherwise."""
    if len(sma_short) < 2:
        return 0
    prev_diff = sma_short.iloc[-2] - sma_long.iloc[-2]
    curr_diff = sma_short.iloc[-1] - sma_long.iloc[-1]
    if pd.isna(prev_diff) or pd.isna(curr_diff):
        return 0
    if prev_diff <= 0 and curr_diff > 0:
        return 1
    if prev_diff >= 0 and curr_diff < 0:
        return -1
    return 0


def ma_trend_signal(sma_short: pd.Series, sma_long: pd.Series) -> int:
    """+1 if short-term MA is above long-term MA (uptrend), -1 if below, 0 if equal/NaN."""
    if pd.isna(sma_short.iloc[-1]) or pd.isna(sma_long.iloc[-1]):
        return 0
    diff = sma_short.iloc[-1] - sma_long.iloc[-1]
    if diff > 0:
        return 1
    if diff < 0:
        return -1
    return 0
