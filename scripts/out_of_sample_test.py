"""Out-of-sample check: take the indicator combos ranked by
sweep_3_indicator_strategies.py on 2021-2026 data and re-test them on a
*different* historical window (2015-2020) they were never fit or compared
against, to see whether their apparent edge holds up or was just a fit to
one specific period.

Usage:
    python scripts/out_of_sample_test.py
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
FETCH_PERIOD = "15y"
OOS_START = pd.Timestamp("2015-01-01")
OOS_END = pd.Timestamp("2020-01-01")
COST_PCT = 0.001
BUY_THRESHOLD = 2
SELL_THRESHOLD = -2

INDICATOR_NAMES = ["ma", "rsi", "macd", "sr"]

# Averages from the in-sample (2021-2026) sweep, for direct comparison.
IN_SAMPLE_AVG_OUTPERFORMANCE = {
    "ma+macd+sr": 0.0268,
    "ma+rsi+macd": -0.1044,
    "rsi+macd+sr": -0.2589,
    "ma+rsi+sr": -0.9037,
}


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

    for ticker in TICKERS:
        raw = fetch_price_history(ticker, period=FETCH_PERIOD)
        # Keep pre-window history for indicator warmup, but cut off anything
        # after the out-of-sample window so later data can't leak in.
        raw = raw.loc[raw.index <= OOS_END]
        if raw.loc[raw.index <= OOS_START].empty:
            print(f"  Skipping {ticker}: no history before {OOS_START.date()}")
            continue

        for combo in combos:
            combo_label = "+".join(combo)
            signals = build_signals(raw, combo)
            result = backtest_action_strategy(signals, start_date=OOS_START, cost_pct=COST_PCT)
            rows.append(
                {
                    "combo": combo_label,
                    "ticker": ticker,
                    "num_trades": result.num_trades,
                    "strategy_return": result.total_return,
                    "benchmark_return": result.benchmark_return,
                    "outperformance": result.total_return - result.benchmark_return,
                }
            )

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("combo")
        .agg(oos_avg_outperformance=("outperformance", "mean"), win_count=("outperformance", lambda s: (s > 0).sum()))
        .sort_values("oos_avg_outperformance", ascending=False)
    )
    summary["in_sample_avg_outperformance"] = summary.index.map(IN_SAMPLE_AVG_OUTPERFORMANCE)
    summary["held_up"] = (summary["oos_avg_outperformance"] > 0) & (summary["in_sample_avg_outperformance"] > 0)

    print(f"\nOut-of-sample test ({OOS_START.date()} to {OOS_END.date()}), vs. in-sample (2021-2026) sweep\n")
    print(summary.to_string(float_format=lambda x: f"{x:.2%}" if isinstance(x, float) else str(x)))

    print("\nPer-ticker detail:")
    print(detail.sort_values(["combo", "ticker"]).to_string(index=False))


if __name__ == "__main__":
    main()
