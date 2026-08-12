"""Rolling realized volatility and volatility regime classification."""

import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def rolling_realized_volatility(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """Annualized realized volatility: rolling std of daily returns scaled by sqrt(252).

    Interpretation: this is the annualized standard deviation of daily price moves
    over the trailing `window` days — e.g. 0.20 means the stock has recently moved
    at a pace consistent with +/-20%/year swings. Higher values mean bigger, choppier
    daily moves; it says nothing about direction, only magnitude.
    """
    daily_returns = df["Close"].pct_change()
    return daily_returns.rolling(window).std() * (TRADING_DAYS_PER_YEAR ** 0.5)


def classify_volatility_regime(
    volatility: pd.Series, low_quantile: float = 0.33, high_quantile: float = 0.67
) -> pd.Series:
    """Bucket a volatility series into "low"/"normal"/"high" using its own historical quantiles.

    Interpretation: thresholds are relative to the series' own history, so "high" means
    "elevated compared to how this ticker usually trades," not an absolute cutoff shared
    across tickers. Useful for spotting regime shifts (e.g. calm -> stressed markets).
    """
    low_thresh = volatility.quantile(low_quantile)
    high_thresh = volatility.quantile(high_quantile)

    regime = pd.Series(pd.NA, index=volatility.index, dtype="object")
    valid = volatility.notna()
    regime.loc[valid & (volatility <= low_thresh)] = "low"
    regime.loc[valid & (volatility > low_thresh) & (volatility <= high_thresh)] = "normal"
    regime.loc[valid & (volatility > high_thresh)] = "high"
    return regime
