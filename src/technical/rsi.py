"""Relative Strength Index (RSI) oversold/overbought signals."""

import pandas as pd


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Return a copy of `df` with an `rsi_{window}` column added (Wilder's RSI)."""
    out = df.copy()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    out[f"rsi_{window}"] = 100 - (100 / (1 + rs))
    return out


def detect_rsi_signals(df: pd.DataFrame, window: int = 14, oversold: float = 30, overbought: float = 70) -> pd.DataFrame:
    """Add RSI plus an `rsi_signal` column marking the day RSI *crosses into*
    oversold (<=`oversold`) or overbought (>=`overbought`) territory, so that a
    multi-day stay below/above the threshold counts as one instance rather than
    one per day.
    """
    out = add_rsi(df, window)
    rsi_col = f"rsi_{window}"
    below = out[rsi_col] <= oversold
    above = out[rsi_col] >= overbought

    out["rsi_signal"] = None
    out.loc[below & ~below.shift(1).fillna(False), "rsi_signal"] = "oversold"
    out.loc[above & ~above.shift(1).fillna(False), "rsi_signal"] = "overbought"
    return out
