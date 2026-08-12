"""MACD (Moving Average Convergence/Divergence) crossover signals."""

import pandas as pd


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Return a copy of `df` with `macd`, `macd_signal`, and `macd_hist` columns added."""
    out = df.copy()
    ema_fast = out["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["Close"].ewm(span=slow, adjust=False).mean()
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(span=signal, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    return out


def detect_macd_crossovers(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Add MACD columns plus a `macd_crossover` column marking the day the MACD
    line crosses above ('bullish') or below ('bearish') its own signal line.
    """
    out = add_macd(df, fast, slow, signal)
    above = out["macd"] > out["macd_signal"]

    out["macd_crossover"] = None
    out.loc[above & ~above.shift(1).fillna(False), "macd_crossover"] = "bullish"
    out.loc[~above & above.shift(1).fillna(False), "macd_crossover"] = "bearish"
    return out
