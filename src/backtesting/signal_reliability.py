"""Signal-reliability backtest: for each pattern type we detect, measure how
forward returns after the signal fires compare to the stock's unconditional
(baseline) return over the same holding period.
"""

from dataclasses import dataclass

import pandas as pd

from src.statistical.mean_reversion import detect_zscore_extremes
from src.technical.bollinger import detect_bollinger_touches
from src.technical.moving_averages import detect_crossover_signals
from src.technical.rsi import detect_rsi_signals

DEFAULT_HORIZONS = (5, 10, 20)


@dataclass
class SignalReliability:
    ticker: str
    signal: str
    horizon: int
    num_instances: int
    win_rate: float
    avg_forward_return: float
    baseline_win_rate: float
    baseline_avg_return: float

    @property
    def win_rate_edge(self) -> float:
        return self.win_rate - self.baseline_win_rate

    @property
    def avg_return_edge(self) -> float:
        return self.avg_forward_return - self.baseline_avg_return


def _forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    """Pct return from each date's close to `horizon` trading days later."""
    return close.shift(-horizon) / close - 1.0


def _win_rate_and_avg(returns: pd.Series) -> tuple[float, float]:
    valid = returns.dropna()
    if valid.empty:
        return float("nan"), float("nan")
    return float((valid > 0).mean()), float(valid.mean())


def evaluate_signal(
    df: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    ticker: str,
    signal_name: str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> list[SignalReliability]:
    """Compare forward returns after `signal_dates` fire to the unconditional
    baseline forward returns of `df` (every day, regardless of signal) over the
    same holding periods.
    """
    close = df["Close"]
    results = []
    for horizon in horizons:
        fwd = _forward_returns(close, horizon)
        baseline_win, baseline_avg = _win_rate_and_avg(fwd)

        signal_fwd = fwd.reindex(signal_dates).dropna()
        signal_win, signal_avg = _win_rate_and_avg(signal_fwd)

        results.append(
            SignalReliability(
                ticker=ticker,
                signal=signal_name,
                horizon=horizon,
                num_instances=len(signal_fwd),
                win_rate=signal_win,
                avg_forward_return=signal_avg,
                baseline_win_rate=baseline_win,
                baseline_avg_return=baseline_avg,
            )
        )
    return results


def golden_cross_reliability(
    df: pd.DataFrame,
    ticker: str,
    short_window: int = 50,
    long_window: int = 200,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> list[SignalReliability]:
    signals = detect_crossover_signals(df, short_window, long_window)
    dates = signals.index[signals["crossover"] == "golden_cross"]
    return evaluate_signal(df, dates, ticker, "golden_cross", horizons)


def rsi_reliability(
    df: pd.DataFrame,
    ticker: str,
    window: int = 14,
    oversold: float = 30,
    overbought: float = 70,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> list[SignalReliability]:
    signals = detect_rsi_signals(df, window, oversold, overbought)
    results = []
    for label in ("oversold", "overbought"):
        dates = signals.index[signals["rsi_signal"] == label]
        results += evaluate_signal(df, dates, ticker, f"rsi_{label}", horizons)
    return results


def bollinger_reliability(
    df: pd.DataFrame,
    ticker: str,
    window: int = 20,
    num_std: float = 2.0,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> list[SignalReliability]:
    signals = detect_bollinger_touches(df, window, num_std)
    results = []
    for label in ("upper_touch", "lower_touch"):
        dates = signals.index[signals["bollinger_signal"] == label]
        results += evaluate_signal(df, dates, ticker, f"bollinger_{label}", horizons)
    return results


def mean_reversion_reliability(
    df: pd.DataFrame,
    ticker: str,
    window: int = 60,
    z_threshold: float = 2.0,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> list[SignalReliability]:
    is_extreme = detect_zscore_extremes(df, window, z_threshold)
    dates = is_extreme.index[is_extreme]
    return evaluate_signal(df, dates, ticker, "high_mean_reversion_zscore", horizons)


def all_signal_reliability(df: pd.DataFrame, ticker: str, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> pd.DataFrame:
    """Run every pattern-type reliability check for one ticker and return a tidy
    DataFrame with one row per (signal, horizon).
    """
    results = (
        golden_cross_reliability(df, ticker, horizons=horizons)
        + rsi_reliability(df, ticker, horizons=horizons)
        + bollinger_reliability(df, ticker, horizons=horizons)
        + mean_reversion_reliability(df, ticker, horizons=horizons)
    )
    return pd.DataFrame(
        [
            {
                "ticker": r.ticker,
                "signal": r.signal,
                "horizon_days": r.horizon,
                "num_instances": r.num_instances,
                "win_rate": r.win_rate,
                "avg_forward_return": r.avg_forward_return,
                "baseline_win_rate": r.baseline_win_rate,
                "baseline_avg_return": r.baseline_avg_return,
                "win_rate_edge": r.win_rate_edge,
                "avg_return_edge": r.avg_return_edge,
            }
            for r in results
        ]
    )
