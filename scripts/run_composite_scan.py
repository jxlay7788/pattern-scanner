"""Scan a ticker for composite BUY/SELL signals combining moving averages,
RSI, MACD, support/resistance, and volume confirmation.

Usage:
    python scripts/run_composite_scan.py [TICKER]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.yfinance_loader import fetch_price_history
from src.technical.composite_signal import detect_buy_sell_signals

TICKER = sys.argv[1] if len(sys.argv) > 1 else "SPY"
PERIOD = "5y"

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    raw = fetch_price_history(TICKER, period=PERIOD)
    signals = detect_buy_sell_signals(raw)

    actions = signals.loc[signals["action"].notna(), ["Close", "signal_score", "action"]]

    print(f"\n{TICKER}: composite BUY/SELL signals ({PERIOD})\n")
    if actions.empty:
        print("  No BUY/SELL signals fired over this period.")
    else:
        print(actions.tail(20).to_string())

    latest = signals.iloc[-1]
    print(f"\n  Latest close ({signals.index[-1].date()}): ${latest['Close']:.2f}")
    print(f"  Latest signal score: {latest['signal_score']:+d}  -> {latest['action'] or 'HOLD'}")

    OUTPUTS_DIR.joinpath("reports").mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.joinpath("charts").mkdir(parents=True, exist_ok=True)
    actions.to_csv(OUTPUTS_DIR / "reports" / f"{TICKER}_composite_signals.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(signals.index, signals["Close"], color="black", linewidth=1, label="Close")
    ax1.plot(signals.index, signals["sma_50"], color="tab:blue", linewidth=1, alpha=0.7, label="SMA 50")
    ax1.plot(signals.index, signals["sma_200"], color="tab:orange", linewidth=1, alpha=0.7, label="SMA 200")

    buys = signals[signals["action"] == "BUY"]
    sells = signals[signals["action"] == "SELL"]
    ax1.scatter(buys.index, buys["Close"], marker="^", color="green", s=100, zorder=5, label="BUY")
    ax1.scatter(sells.index, sells["Close"], marker="v", color="red", s=100, zorder=5, label="SELL")

    ax1.set_title(f"{TICKER}: Composite Buy/Sell Signals")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left")

    ax2.plot(signals.index, signals["rsi_14"], color="purple", linewidth=1, label="RSI (14)")
    ax2.axhline(70, color="red", linestyle="--", alpha=0.5)
    ax2.axhline(30, color="green", linestyle="--", alpha=0.5)
    ax2.set_ylabel("RSI")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "charts" / f"{TICKER}_composite_signals.png", dpi=150)

    print(f"\nSaved chart to outputs/charts/{TICKER}_composite_signals.png")
    print(f"Saved signal log to outputs/reports/{TICKER}_composite_signals.csv")


if __name__ == "__main__":
    main()
