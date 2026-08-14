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

SIGNAL_EMOJI = {
    "STRONG BUY": "\U0001F7E2\U0001F7E2",
    "BUY": "\U0001F7E2",
    "SELL": "\U0001F534",
    "STRONG SELL": "\U0001F534\U0001F534",
}


def build_reason(row: pd.Series) -> str:
    """Plain-English explanation of the strongest 1-2 indicators behind a signal."""
    reasons = []
    if row.get("ma_cross_signal") == 1:
        reasons.append("just golden-crossed (50d MA above 200d)")
    elif row.get("ma_cross_signal") == -1:
        reasons.append("just death-crossed (50d MA below 200d)")
    if row.get("rsi_signal") == 1:
        reasons.append(f"oversold, RSI {row.get('rsi')}")
    elif row.get("rsi_signal") == -1:
        reasons.append(f"overbought, RSI {row.get('rsi')}")
    if row.get("bollinger_signal") == 1:
        reasons.append("price broke below its normal range")
    elif row.get("bollinger_signal") == -1:
        reasons.append("price broke above its normal range")
    if row.get("macd_signal") == 1:
        reasons.append("momentum just turned up")
    elif row.get("macd_signal") == -1:
        reasons.append("momentum just turned down")
    if not reasons:
        if row.get("ma_trend_signal") == 1:
            reasons.append("trending up")
        elif row.get("ma_trend_signal") == -1:
            reasons.append("trending down")
    return "; ".join(reasons[:2]) if reasons else "mixed signals"


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

    rows = []
    for _, row in subset.iterrows():
        rows.append(
            [
                f"{SIGNAL_EMOJI.get(row['signal'], '')} {row['ticker']}",
                f"${row['close']:.2f}",
                build_reason(row),
            ]
        )
    return tabulate(rows, headers=["Stock", "Price", "Why"], tablefmt="simple")


def main() -> None:
    args = parse_args()
    if not args.token or not args.chat_id:
        raise SystemExit("Missing Telegram credentials: pass --token/--chat-id or set env vars.")

    results = pd.read_csv(args.csv)

    buy_table = build_table_block(results, ["STRONG BUY", "BUY"], ascending=False, top=args.top)
    sell_table = build_table_block(results, ["STRONG SELL", "SELL"], ascending=True, top=args.top)

    sections = [
        "<b>Stock Signal Alert</b>\n"
        "\U0001F7E2\U0001F7E2 = strong buy   \U0001F7E2 = buy   "
        "\U0001F534 = sell   \U0001F534\U0001F534 = strong sell"
    ]

    if args.brief_file and os.path.exists(args.brief_file):
        with open(args.brief_file, "r", encoding="utf-8") as f:
            brief = f.read().strip()
        if brief:
            sections.append(html.escape(brief))

    sections.append(f"<b>Consider buying</b>\n<pre>{html.escape(buy_table)}</pre>")
    sections.append(f"<b>Consider selling</b>\n<pre>{html.escape(sell_table)}</pre>")
    sections.append(
        "<i>Automated technical-indicator output (RSI, MACD, moving averages, "
        "Bollinger Bands) — not investment advice.</i>"
    )

    message = "\n\n".join(sections)
    send_message(args.token, args.chat_id, message)
    print("Sent to Telegram.")


if __name__ == "__main__":
    main()
