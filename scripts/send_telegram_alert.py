"""CLI: format today's scan results as a table + market brief and send them to
Telegram.

Usage:
    python scripts/send_telegram_alert.py --csv outputs/reports/scan_....csv \
        --brief-file outputs/reports/brief_....txt \
        --token <bot_token> --chat-id <chat_id>

The bot token and chat ID are secrets — pass them as arguments (or via
TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars) rather than committing them
anywhere in this repo.
"""

import argparse
import html
import os
import sys

import pandas as pd
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.notifications.telegram import send_message

TABLE_COLUMNS = ["ticker", "close", "signal", "score", "rsi", "sma50", "sma200"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a scan report + brief to Telegram.")
    parser.add_argument("--csv", required=True, help="Path to the scan CSV produced by scan_market.py.")
    parser.add_argument(
        "--brief-file",
        help="Path to a text file containing the market brief. Omit to skip the brief section.",
    )
    parser.add_argument("--top", type=int, default=15, help="Max rows per BUY/SELL table (default: 15).")
    parser.add_argument(
        "--token",
        default=os.environ.get("TELEGRAM_BOT_TOKEN"),
        help="Telegram bot token (default: TELEGRAM_BOT_TOKEN env var).",
    )
    parser.add_argument(
        "--chat-id",
        default=os.environ.get("TELEGRAM_CHAT_ID"),
        help="Telegram chat ID (default: TELEGRAM_CHAT_ID env var).",
    )
    return parser.parse_args()


def build_table_block(df: pd.DataFrame, signals: list[str], ascending: bool, top: int) -> str:
    subset = df[df["signal"].isin(signals)].copy()
    subset = subset.sort_values("score", ascending=ascending).head(top)
    if subset.empty:
        return "(none)"
    return tabulate(subset[TABLE_COLUMNS], headers="keys", tablefmt="simple", showindex=False)


def main() -> None:
    args = parse_args()
    if not args.token or not args.chat_id:
        raise SystemExit("Missing Telegram credentials: pass --token/--chat-id or set env vars.")

    results = pd.read_csv(args.csv)

    buy_table = build_table_block(results, ["STRONG BUY", "BUY"], ascending=False, top=args.top)
    sell_table = build_table_block(results, ["STRONG SELL", "SELL"], ascending=True, top=args.top)

    sections = [f"<b>Stock Signal Alert</b>"]

    if args.brief_file and os.path.exists(args.brief_file):
        with open(args.brief_file, "r", encoding="utf-8") as f:
            brief = f.read().strip()
        if brief:
            sections.append(html.escape(brief))

    sections.append(f"<b>BUY signals</b>\n<pre>{html.escape(buy_table)}</pre>")
    sections.append(f"<b>SELL signals</b>\n<pre>{html.escape(sell_table)}</pre>")
    sections.append(
        "<i>Automated technical-indicator output (RSI, MACD crossover, "
        "50/200-day SMA cross, Bollinger Bands) — not investment advice.</i>"
    )

    message = "\n\n".join(sections)
    send_message(args.token, args.chat_id, message)
    print("Sent to Telegram.")


if __name__ == "__main__":
    main()
