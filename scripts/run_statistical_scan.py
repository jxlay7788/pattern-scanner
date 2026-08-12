"""Run the statistical pattern scan (volatility, correlation, mean reversion,
autocorrelation) over a basket of tickers.

Usage:
    python scripts/run_statistical_scan.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.yfinance_loader import fetch_price_history
from src.statistical.autocorrelation import return_autocorrelation
from src.statistical.correlation import rolling_correlation_matrix
from src.statistical.mean_reversion import hurst_exponent, zscore_vs_moving_average
from src.statistical.volatility import classify_volatility_regime, rolling_realized_volatility
from src.utils.plotting import plot_correlation_heatmap, plot_volatility_regime

TICKERS = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"]
PRIMARY_TICKER = "SPY"
VOL_WINDOW = 30
CORR_WINDOW = 60
MA_WINDOW = 60
AUTOCORR_LAGS = (1, 5, 20)

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    price_history = {ticker: fetch_price_history(ticker) for ticker in TICKERS}
    closes = pd.DataFrame({ticker: df["Close"] for ticker, df in price_history.items()}).dropna(how="all")

    OUTPUTS_DIR.joinpath("reports").mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.joinpath("charts").mkdir(parents=True, exist_ok=True)

    rows = []
    for ticker, df in price_history.items():
        volatility = rolling_realized_volatility(df, window=VOL_WINDOW)
        regime = classify_volatility_regime(volatility)
        zscore = zscore_vs_moving_average(df, window=MA_WINDOW)
        hurst = hurst_exponent(df["Close"])
        autocorr = return_autocorrelation(df, lags=AUTOCORR_LAGS)

        print(f"\n{ticker}")
        print(f"  Realized vol ({VOL_WINDOW}d, annualized): {volatility.iloc[-1]:.2%}  [{regime.iloc[-1]} regime]")
        print(f"  Z-score vs {MA_WINDOW}d MA:               {zscore.iloc[-1]:+.2f}")
        print(f"  Hurst exponent:                       {hurst:.3f}")
        print(f"  Return autocorrelation:               " + ", ".join(f"lag{k}={v:+.3f}" for k, v in autocorr.items()))

        rows.append(
            {
                "ticker": ticker,
                "realized_vol_30d": volatility.iloc[-1],
                "vol_regime": regime.iloc[-1],
                "zscore_vs_60d_ma": zscore.iloc[-1],
                "hurst_exponent": hurst,
                **{f"autocorr_lag{k}": v for k, v in autocorr.items()},
            }
        )

        if ticker == PRIMARY_TICKER:
            plot_volatility_regime(
                df, regime, ticker, OUTPUTS_DIR / "charts" / f"{ticker}_volatility_regime.png"
            )

    pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "reports" / "statistical_scan_summary.csv", index=False)

    corr_matrix = rolling_correlation_matrix(closes, window=CORR_WINDOW)
    corr_matrix.to_csv(OUTPUTS_DIR / "reports" / "correlation_matrix.csv")
    plot_correlation_heatmap(corr_matrix, OUTPUTS_DIR / "charts" / "correlation_heatmap.png")

    print(f"\nSaved reports to outputs/reports/statistical_scan_summary.csv and correlation_matrix.csv")
    print(f"Saved charts to outputs/charts/{PRIMARY_TICKER}_volatility_regime.png and correlation_heatmap.png")


if __name__ == "__main__":
    main()
