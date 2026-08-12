"""Moving average crossover signals (golden cross / death cross)."""

import pandas as pd


def add_moving_averages(df: pd.DataFrame, short_window: int = 50, long_window: int = 200) -> pd.DataFrame:
    """Return a copy of `df` with short/long SMA columns added."""
    out = df.copy()
    out[f"sma_{short_window}"] = out["Close"].rolling(short_window).mean()
    out[f"sma_{long_window}"] = out["Close"].rolling(long_window).mean()
    return out


def detect_crossover_signals(df: pd.DataFrame, short_window: int = 50, long_window: int = 200) -> pd.DataFrame:
    """Add a `signal` column (1 = short SMA above long SMA, 0 = below) and a
    `crossover` column marking the day of each 'golden_cross' / 'death_cross' event.
    """
    out = add_moving_averages(df, short_window, long_window)
    short_col, long_col = f"sma_{short_window}", f"sma_{long_window}"

    valid = out[short_col].notna() & out[long_col].notna()
    out["signal"] = pd.NA
    out.loc[valid, "signal"] = (out.loc[valid, short_col] > out.loc[valid, long_col]).astype(int)

    prev_signal = out["signal"].ffill().shift(1)
    out["crossover"] = None
    out.loc[(out["signal"] == 1) & (prev_signal == 0), "crossover"] = "golden_cross"
    out.loc[(out["signal"] == 0) & (prev_signal == 1), "crossover"] = "death_cross"

    return out
