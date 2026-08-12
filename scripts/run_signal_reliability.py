"""Backtest how reliably each detected pattern (golden cross, RSI
oversold/overbought, Bollinger Band touches, high mean-reversion z-score) has
been followed by a positive return, vs. each ticker's unconditional baseline
return over the same holding period.

Usage:
    python scripts/run_signal_reliability.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtesting.signal_reliability import all_signal_reliability
from src.ingestion.yfinance_loader import fetch_price_history
from src.utils.reporting import render_reliability_section

TICKERS = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"]
HORIZONS = (5, 10, 20)

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    reliability_by_ticker = {}
    for ticker in TICKERS:
        df = fetch_price_history(ticker)
        reliability = all_signal_reliability(df, ticker, horizons=HORIZONS)
        reliability_by_ticker[ticker] = reliability

        print(f"\n{ticker}")
        print(reliability.to_string(index=False))

    combined = pd.concat(reliability_by_ticker.values(), ignore_index=True)
    combined.to_csv(REPORTS_DIR / "signal_reliability_summary.csv", index=False)

    section = render_reliability_section(reliability_by_ticker)
    report_path = REPORTS_DIR / "report.md"
    report_path.write_text(section + "\n", encoding="utf-8")

    print(f"\nSaved summary to outputs/reports/signal_reliability_summary.csv")
    print(f"Saved reliability section to outputs/reports/report.md")


if __name__ == "__main__":
    main()
