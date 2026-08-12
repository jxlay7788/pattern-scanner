"""Systematically test every 3-of-4 combination of MA crossover / RSI /
MACD / support-resistance against buy-and-hold, across a fixed basket of
tickers, and rank them by average outperformance.

This is an explicit data-mining exercise: trying many rule combinations
against the *same* fixed historical window and picking the best-looking one
is a classic overfitting trap (see caveat printed at the end). Treat any
"winner" here as a description of what happened to work on this specific
5-year window, not a discovered, durable edge.

Usage:
    python scripts/sweep_3_indicator_strategies.py
"""

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtesting.backtest import backtest_action_strategy
from src.ingestion.yfinance_loader import fetch_price_history
from src.technical.macd import detect_macd_crossovers
from src.technical.moving_averages import detect_crossover_signals
from src.technical.rsi import detect_rsi_signals
from src.technical.support_resistance import detect_support_resistance_signals

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "SPY", "TSLA"]
BACKTEST_YEARS = 5
COST_PCT = 0.001
BUY_THRESHOLD = 2  # 2 of the 3 selected indicators must agree the same day
SELL_THRESHOLD = -2

INDICATOR_NAMES = ["ma", "rsi", "macd", "sr"]


def _vote(row: pd.Series, indicator: str) -> int:
    if indicator == "ma":
        if row.get("crossover") == "golden_cross":
            return 1
        if row.get("crossover") == "death_cross":
            return -1
    elif indicator == "rsi":
        if row.get("rsi_signal") == "oversold":
            return 1
        if row.get("rsi_signal") == "overbought":
            return -1
    elif indicator == "macd":
        if row.get("macd_crossover") == "bullish":
            return 1
        if row.get("macd_crossover") == "bearish":
            return -1
    elif indicator == "sr":
        if row.get("sr_signal") == "support_bounce":
            return 1
        if row.get("sr_signal") == "resistance_rejection":
            return -1
    return 0


def build_signals(df: pd.DataFrame, indicators: tuple[str, ...]) -> pd.DataFrame:
    out = detect_crossover_signals(df)
    out = detect_rsi_signals(out)
    out = detect_macd_crossovers(out)
    out = detect_support_resistance_signals(out)

    out["signal_score"] = out.apply(lambda row: sum(_vote(row, ind) for ind in indicators), axis=1)
    out["action"] = None
    out.loc[out["signal_score"] >= BUY_THRESHOLD, "action"] = "BUY"
    out.loc[out["signal_score"] <= SELL_THRESHOLD, "action"] = "SELL"
    return out


def main() -> None:
    combos = list(combinations(INDICATOR_NAMES, 3))
    rows = []

    raw_by_ticker = {t: fetch_price_history(t, period=f"{BACKTEST_YEARS + 1}y") for t in TICKERS}

    for combo in combos:
        combo_label = "+".join(combo)
        for ticker, raw in raw_by_ticker.items():
            signals = build_signals(raw, combo)
            start_date = signals.index[-1] - pd.DateOffset(years=BACKTEST_YEARS)
            result = backtest_action_strategy(signals, start_date=start_date, cost_pct=COST_PCT)
            rows.append(
                {
                    "combo": combo_label,
                    "ticker": ticker,
                    "num_trades": result.num_trades,
                    "strategy_return": result.total_return,
                    "benchmark_return": result.benchmark_return,
                    "outperformance": result.total_return - result.benchmark_return,
                    "doubled_benchmark": result.total_return >= 2 * result.benchmark_return,
                }
            )

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("combo")
        .agg(
            avg_outperformance=("outperformance", "mean"),
            win_count=("outperformance", lambda s: (s > 0).sum()),
            avg_trades=("num_trades", "mean"),
            any_doubled=("doubled_benchmark", "any"),
        )
        .sort_values("avg_outperformance", ascending=False)
    )

    print(f"\n3-of-4 indicator sweep across {', '.join(TICKERS)} ({BACKTEST_YEARS}y, {COST_PCT:.2%} cost/trade)\n")
    print(summary.to_string(float_format=lambda x: f"{x:.2%}" if abs(x) < 10 else f"{x:.1f}"))

    print("\nPer-ticker detail for the best-ranked combo:")
    best_combo = summary.index[0]
    print(detail.loc[detail["combo"] == best_combo].to_string(index=False))

    print(
        "\nCAVEAT: this ranking comes from testing every combination against the "
        "same fixed 5-year historical window and picking the best average result "
        "-- that is a textbook overfitting/data-mining setup. A combo 'winning' "
        "here describes what happened to work on this specific past data, not a "
        "verified, durable trading edge. Out-of-sample testing (a different time "
        "period the combo was never checked against) is the only way to tell the "
        "difference."
    )


if __name__ == "__main__":
    main()
