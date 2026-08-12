"""Backtest engine for the golden-cross / death-cross moving-average strategy."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    trades: pd.DataFrame
    total_return: float
    benchmark_return: float
    cagr: float
    benchmark_cagr: float
    max_drawdown: float
    benchmark_max_drawdown: float
    num_trades: int


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return drawdown.min()


def _cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def backtest_crossover_strategy(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    cost_pct: float = 0.001,
    initial_capital: float = 10_000.0,
) -> BacktestResult:
    """Simulate: go long on golden_cross, go flat (cash) on death_cross.

    `df` must already contain `signal` and `crossover` columns (see
    `src.technical.moving_averages.detect_crossover_signals`) and should include
    enough history *before* `start_date` for the long moving average to be valid,
    so the strategy's position on day 1 of the window reflects the true prevailing
    trend rather than an artificial cold start.
    """
    data = df.copy()
    data["daily_return"] = data["Close"].pct_change()

    # Position is entered/exited the day *after* the crossover signal fires,
    # using only information known at the previous close (no lookahead).
    data["position"] = data["signal"].ffill().shift(1).fillna(0).astype(int)

    window = data.loc[data.index >= pd.Timestamp(start_date)].copy()
    if window.empty:
        raise ValueError(f"No data on or after {start_date}")

    trade_days = window.index[window["crossover"].notna()]
    trades = window.loc[trade_days, ["Close", "crossover"]].rename(columns={"Close": "price"})

    position_changed = window["position"].diff().fillna(window["position"].iloc[0]) != 0
    cost = np.where(position_changed, cost_pct, 0.0)

    strategy_return = window["position"] * window["daily_return"] - cost
    strategy_return.iloc[0] = 0.0

    benchmark_return = window["daily_return"].copy()
    benchmark_return.iloc[0] = -cost_pct  # one-time entry cost for buy-and-hold

    equity_curve = initial_capital * (1 + strategy_return).cumprod()
    benchmark_curve = initial_capital * (1 + benchmark_return).cumprod()

    return BacktestResult(
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        trades=trades,
        total_return=equity_curve.iloc[-1] / initial_capital - 1,
        benchmark_return=benchmark_curve.iloc[-1] / initial_capital - 1,
        cagr=_cagr(equity_curve),
        benchmark_cagr=_cagr(benchmark_curve),
        max_drawdown=_max_drawdown(equity_curve),
        benchmark_max_drawdown=_max_drawdown(benchmark_curve),
        num_trades=len(trades),
    )


def backtest_action_strategy(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    cost_pct: float = 0.001,
    initial_capital: float = 10_000.0,
) -> BacktestResult:
    """Simulate: go long on 'BUY', go flat (cash) on 'SELL', hold otherwise.

    `df` must already contain an `action` column ('BUY' / 'SELL' / None), e.g.
    from `src.technical.composite_signal.detect_buy_sell_signals`, and should
    include enough history *before* `start_date` for its underlying indicators
    to be valid, same as `backtest_crossover_strategy`.
    """
    data = df.copy()
    data["daily_return"] = data["Close"].pct_change()

    # Forward-fill the last BUY/SELL action into a 0/1 position, entered the
    # day *after* it fires (no lookahead), same convention as the crossover
    # backtest above.
    action_position = data["action"].map({"BUY": 1, "SELL": 0})
    data["position"] = action_position.ffill().shift(1).fillna(0).astype(int)

    window = data.loc[data.index >= pd.Timestamp(start_date)].copy()
    if window.empty:
        raise ValueError(f"No data on or after {start_date}")

    trade_days = window.index[window["action"].notna()]
    trades = window.loc[trade_days, ["Close", "action"]].rename(columns={"Close": "price"})

    position_changed = window["position"].diff().fillna(window["position"].iloc[0]) != 0
    cost = np.where(position_changed, cost_pct, 0.0)

    strategy_return = window["position"] * window["daily_return"] - cost
    strategy_return.iloc[0] = 0.0

    benchmark_return = window["daily_return"].copy()
    benchmark_return.iloc[0] = -cost_pct  # one-time entry cost for buy-and-hold

    equity_curve = initial_capital * (1 + strategy_return).cumprod()
    benchmark_curve = initial_capital * (1 + benchmark_return).cumprod()

    return BacktestResult(
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        trades=trades,
        total_return=equity_curve.iloc[-1] / initial_capital - 1,
        benchmark_return=benchmark_curve.iloc[-1] / initial_capital - 1,
        cagr=_cagr(equity_curve),
        benchmark_cagr=_cagr(benchmark_curve),
        max_drawdown=_max_drawdown(equity_curve),
        benchmark_max_drawdown=_max_drawdown(benchmark_curve),
        num_trades=len(trades),
    )
