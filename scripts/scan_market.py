"""CLI: scan the US stock market (S&P 500 or a custom watchlist) for technical
buy/sell signals and write a ranked CSV report.

Usage:
    python scripts/scan_market.py
    python scripts/scan_market.py --tickers-file my_watchlist.txt
    python scripts/scan_market.py --period 2y --top 15
"""

import argparse
import os
import sys
from datetime import datetime

from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.data_loader import download_price_history
from src.ingestion.universe import get_sp500_tickers, load_tickers_from_file
from src.signals.composite import analyze_universe

DISPLAY_COLUMNS = ["ticker", "close", "signal", "score", "rsi", "sma50", "sma200"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan US stocks for technical buy/sell signals.")
    parser.add_argument(
        "--tickers-file",
        help="Path to a text file of tickers (one per line). Defaults to the full S&P 500.",
    )
    parser.add_argument(
        "--period",
        default="1y",
        help="yfinance history window, e.g. 6mo, 1y, 2y (default: 1y).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top buy/sell signals to print to the console (default: 10).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV output path (default: outputs/reports/scan_<timestamp>.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.tickers_file:
        print(f"Loading tickers from {args.tickers_file}...")
        tickers = load_tickers_from_file(args.tickers_file)
    else:
        print("Fetching S&P 500 ticker list from Wikipedia...")
        tickers = get_sp500_tickers()
    print(f"Scanning {len(tickers)} tickers.")

    print(f"Downloading {args.period} of price history...")
    price_data = download_price_history(tickers, period=args.period)
    print(f"Retrieved usable history for {len(price_data)}/{len(tickers)} tickers.")

    print("Computing indicators and composite signals...")
    results = analyze_universe(price_data)

    if results.empty:
        print("No signals could be computed. Check your network connection or ticker list.")
        return

    output_path = args.output or os.path.join(
        "outputs", "reports", f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nFull report saved to {output_path}\n")

    buys = results[results["signal"].isin(["BUY", "STRONG BUY"])].head(args.top)
    sells = (
        results[results["signal"].isin(["SELL", "STRONG SELL"])]
        .sort_values("score")
        .head(args.top)
    )

    print(f"=== Top {len(buys)} BUY signals ===")
    print(tabulate(buys[DISPLAY_COLUMNS], headers="keys", tablefmt="simple", showindex=False))

    print(f"\n=== Top {len(sells)} SELL signals ===")
    print(tabulate(sells[DISPLAY_COLUMNS], headers="keys", tablefmt="simple", showindex=False))


if __name__ == "__main__":
    main()
