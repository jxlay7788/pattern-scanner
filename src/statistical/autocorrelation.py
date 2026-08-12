"""Autocorrelation of daily returns at fixed lags."""

import pandas as pd


def return_autocorrelation(df: pd.DataFrame, lags: tuple[int, ...] = (1, 5, 20)) -> dict[int, float]:
    """Autocorrelation of daily returns at each lag in `lags`.

    Interpretation: positive autocorrelation at a lag means returns k days apart tend
    to move in the same direction (momentum at that horizon); negative means they tend
    to move opposite (mean reversion at that horizon); values near 0 mean returns at
    that lag are effectively unrelated, consistent with a random walk.
    """
    returns = df["Close"].pct_change().dropna()
    return {lag: returns.autocorr(lag=lag) for lag in lags}
