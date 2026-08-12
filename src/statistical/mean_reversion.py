"""Mean-reversion diagnostics: Hurst exponent and price-vs-moving-average z-score."""

import numpy as np
import pandas as pd


def hurst_exponent(series: pd.Series, min_lag: int = 2, max_lag: int = 100) -> float:
    """Estimate the Hurst exponent via the variance of lagged differences.

    Interpretation: H < 0.5 indicates mean-reverting behavior (moves tend to reverse),
    H ~= 0.5 indicates a random walk (no memory, moves are independent), and H > 0.5
    indicates a trending/persistent series (moves tend to continue). This is a coarse
    estimate over the whole series, useful as a quick screen rather than a precise
    statistical test.
    """
    values = series.dropna().to_numpy()
    max_lag = min(max_lag, len(values) // 2)
    lags = range(min_lag, max_lag)
    if len(list(lags)) < 2:
        return float("nan")

    tau = [np.std(values[lag:] - values[:-lag]) for lag in lags]
    tau = np.array(tau)
    valid = tau > 0
    if valid.sum() < 2:
        return float("nan")

    log_lags = np.log(np.array(list(lags))[valid])
    log_tau = np.log(tau[valid])
    slope = np.polyfit(log_lags, log_tau, 1)[0]
    return slope * 2.0


def zscore_vs_moving_average(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """Z-score of Close price vs its rolling `window`-day mean, in rolling std units.

    Interpretation: a large positive z-score means price is stretched well above its
    recent average (potentially overbought / due to revert lower), a large negative
    z-score means price is stretched well below it (potentially oversold / due to
    revert higher). Values near 0 mean price is trading close to its recent norm.
    """
    close = df["Close"]
    rolling_mean = close.rolling(window).mean()
    rolling_std = close.rolling(window).std()
    return (close - rolling_mean) / rolling_std


def detect_zscore_extremes(df: pd.DataFrame, window: int = 60, z_threshold: float = 2.0) -> pd.Series:
    """Boolean Series marking the first day the z-score vs the `window`-day
    moving average crosses beyond +/-`z_threshold` (a "high mean-reversion"
    signal), so a multi-day stay beyond the threshold counts as one instance.
    """
    z = zscore_vs_moving_average(df, window)
    extreme = z.abs() >= z_threshold
    return extreme & ~extreme.shift(1).fillna(False)
