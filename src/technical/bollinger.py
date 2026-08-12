"""Bollinger Band touch signals."""

import pandas as pd


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Return a copy of `df` with rolling mean/upper/lower Bollinger Band columns."""
    out = df.copy()
    rolling_mean = out["Close"].rolling(window).mean()
    rolling_std = out["Close"].rolling(window).std()
    out[f"bb_mid_{window}"] = rolling_mean
    out[f"bb_upper_{window}"] = rolling_mean + num_std * rolling_std
    out[f"bb_lower_{window}"] = rolling_mean - num_std * rolling_std
    return out


def detect_bollinger_touches(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Bands plus a `bollinger_signal` column marking the first day
    Close touches or breaches the upper/lower band after a period inside the
    bands, so a multi-day ride along a band counts as one instance.
    """
    out = add_bollinger_bands(df, window, num_std)
    upper_col, lower_col = f"bb_upper_{window}", f"bb_lower_{window}"

    touching_upper = out["Close"] >= out[upper_col]
    touching_lower = out["Close"] <= out[lower_col]

    out["bollinger_signal"] = None
    out.loc[touching_upper & ~touching_upper.shift(1).fillna(False), "bollinger_signal"] = "upper_touch"
    out.loc[touching_lower & ~touching_lower.shift(1).fillna(False), "bollinger_signal"] = "lower_touch"
    return out
