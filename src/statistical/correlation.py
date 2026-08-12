"""Rolling correlation across tickers, based on daily returns."""

import pandas as pd


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a DataFrame of Close prices (one column per ticker) to daily returns."""
    return prices.pct_change()


def rolling_correlation_matrix(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Pairwise correlation matrix of daily returns over the most recent `window` days.

    Interpretation: values near +1 mean two tickers have been moving together
    (diversification between them is weak); values near 0 mean their moves have been
    roughly independent; values near -1 mean they've been moving opposite each other.
    Because this uses only the trailing window, it reflects the *current* correlation
    regime, which can shift over time (e.g. correlations often rise sharply in selloffs).
    """
    returns = daily_returns(prices).tail(window)
    return returns.corr()
