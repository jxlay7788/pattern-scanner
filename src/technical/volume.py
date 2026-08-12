"""Volume-spike confirmation signals."""

import pandas as pd


def add_volume_avg(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Return a copy of `df` with a rolling average-volume column added."""
    out = df.copy()
    out[f"volume_avg_{window}"] = out["Volume"].rolling(window).mean()
    return out


def detect_volume_signals(df: pd.DataFrame, window: int = 20, spike_multiple: float = 1.5) -> pd.DataFrame:
    """Add rolling average volume plus a `volume_signal` column marking days
    where volume is >= `spike_multiple`x its rolling average -- used to gauge
    whether a price move has real conviction behind it, not a direction on its
    own.
    """
    out = add_volume_avg(df, window)
    avg_col = f"volume_avg_{window}"

    is_spike = out["Volume"] >= spike_multiple * out[avg_col]
    out["volume_signal"] = None
    out.loc[is_spike, "volume_signal"] = "volume_spike"
    return out
