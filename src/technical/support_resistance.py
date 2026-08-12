"""Rolling support/resistance levels derived from local price pivots."""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def add_support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Return a copy of `df` with `resistance`/`support` columns: the most recent
    confirmed pivot high/low, carried forward until superseded by a newer one.

    A pivot (local max/min over `window` bars on each side) is only confirmed
    once `window` bars have passed after it, so the level is shifted forward by
    `window` before being used -- this avoids leaking future price data into a
    level that wouldn't have actually been knowable on that date.
    """
    out = df.copy()
    close = out["Close"].to_numpy()

    local_max_idx = argrelextrema(close, np.greater_equal, order=window)[0]
    local_min_idx = argrelextrema(close, np.less_equal, order=window)[0]

    pivot_high = pd.Series(np.nan, index=out.index)
    pivot_low = pd.Series(np.nan, index=out.index)
    pivot_high.iloc[local_max_idx] = close[local_max_idx]
    pivot_low.iloc[local_min_idx] = close[local_min_idx]

    out["resistance"] = pivot_high.shift(window).ffill()
    out["support"] = pivot_low.shift(window).ffill()
    return out


def detect_support_resistance_signals(df: pd.DataFrame, window: int = 20, touch_pct: float = 0.01) -> pd.DataFrame:
    """Add support/resistance levels plus an `sr_signal` column marking a
    'support_bounce' (price closes within `touch_pct` of support, up from the
    prior close) or a 'resistance_rejection' (price closes within `touch_pct`
    of resistance, down from the prior close).
    """
    out = add_support_resistance(df, window)

    near_support = (out["Close"] - out["support"]).abs() / out["support"] <= touch_pct
    near_resistance = (out["resistance"] - out["Close"]).abs() / out["resistance"] <= touch_pct
    closes_up = out["Close"] > out["Close"].shift(1)
    closes_down = out["Close"] < out["Close"].shift(1)

    out["sr_signal"] = None
    out.loc[near_support & closes_up, "sr_signal"] = "support_bounce"
    out.loc[near_resistance & closes_down, "sr_signal"] = "resistance_rejection"
    return out
