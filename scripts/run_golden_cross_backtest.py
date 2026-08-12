"""Backtest the 50/200-day golden-cross strategy on SPY over the last 5 years,
vs. buy-and-hold, including transaction costs.

Usage:
    python scripts/run_golden_cross_backtest.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtesting.backtest import backtest_crossover_strategy
from src.ingestion.yfinance_loader import fetch_price_history
from src.technical.moving_averages import detect_crossover_signals

TICKER = "SPY"
SHORT_WINDOW = 50
LONG_WINDOW = 200
COST_PCT = 0.001
BACKTEST_YEARS = 5

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    # Fetch extra history before the window so the 200-day SMA (and the
    # position carried into the window) is valid from day 1 of the backtest.
    raw = fetch_price_history(TICKER, period=f"{BACKTEST_YEARS + 1}y")
    signals = detect_crossover_signals(raw, SHORT_WINDOW, LONG_WINDOW)

    start_date = signals.index[-1] - pd.DateOffset(years=BACKTEST_YEARS)
    result = backtest_crossover_strategy(signals, start_date=start_date, cost_pct=COST_PCT)

    print(f"\n{TICKER} golden-cross backtest ({start_date.date()} to {signals.index[-1].date()})")
    print(f"  {SHORT_WINDOW}/{LONG_WINDOW}-day SMA crossover, {COST_PCT:.2%} cost per trade\n")
    print(f"  Trades executed:        {result.num_trades}")
    print(f"  Strategy total return:  {result.total_return:+.2%}  (CAGR {result.cagr:+.2%})")
    print(f"  Buy-and-hold return:    {result.benchmark_return:+.2%}  (CAGR {result.benchmark_cagr:+.2%})")
    print(f"  Strategy max drawdown:  {result.max_drawdown:.2%}")
    print(f"  Buy-hold max drawdown:  {result.benchmark_max_drawdown:.2%}")
    print(f"  Outperformance:         {result.total_return - result.benchmark_return:+.2%}\n")

    if not result.trades.empty:
        print(result.trades.to_string())

    OUTPUTS_DIR.joinpath("reports").mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.joinpath("charts").mkdir(parents=True, exist_ok=True)

    result.trades.to_csv(OUTPUTS_DIR / "reports" / f"{TICKER}_golden_cross_trades.csv")

    summary = pd.DataFrame(
        [
            {
                "ticker": TICKER,
                "start_date": start_date.date(),
                "end_date": signals.index[-1].date(),
                "num_trades": result.num_trades,
                "strategy_total_return": result.total_return,
                "strategy_cagr": result.cagr,
                "strategy_max_drawdown": result.max_drawdown,
                "benchmark_total_return": result.benchmark_return,
                "benchmark_cagr": result.benchmark_cagr,
                "benchmark_max_drawdown": result.benchmark_max_drawdown,
                "cost_pct": COST_PCT,
            }
        ]
    )
    summary.to_csv(OUTPUTS_DIR / "reports" / f"{TICKER}_golden_cross_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(result.equity_curve, label="Golden-cross strategy")
    ax.plot(result.benchmark_curve, label="Buy and hold")
    for date, row in result.trades.iterrows():
        color = "green" if row["crossover"] == "golden_cross" else "red"
        ax.axvline(date, color=color, alpha=0.2, linestyle="--")
    ax.set_title(f"{TICKER}: Golden-Cross Strategy vs. Buy-and-Hold ({BACKTEST_YEARS}y, {COST_PCT:.2%} cost/trade)")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "charts" / f"{TICKER}_golden_cross_equity_curve.png", dpi=150)

    print(f"\nSaved chart to outputs/charts/{TICKER}_golden_cross_equity_curve.png")
    print(f"Saved reports to outputs/reports/{TICKER}_golden_cross_*.csv")


if __name__ == "__main__":
    main()
