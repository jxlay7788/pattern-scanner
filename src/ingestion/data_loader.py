"""Batched historical price downloads via yfinance."""

import time

import pandas as pd
import yfinance as yf

CHUNK_SIZE = 50
MIN_ROWS_REQUIRED = 210  # need ~200 sessions for the 200-day SMA plus warmup


def _chunks(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def download_price_history(
    tickers: list[str], period: str = "1y", retries: int = 2
) -> dict[str, pd.DataFrame]:
    """Downloads OHLCV history for each ticker, chunked to keep requests reliable.

    Returns a dict of ticker -> DataFrame, skipping tickers with insufficient history.
    """
    result: dict[str, pd.DataFrame] = {}
    chunk_list = list(_chunks(tickers, CHUNK_SIZE))

    for idx, chunk in enumerate(chunk_list, start=1):
        print(f"  Downloading batch {idx}/{len(chunk_list)} ({len(chunk)} tickers)...")
        data = None
        for attempt in range(retries + 1):
            try:
                data = yf.download(
                    tickers=chunk,
                    period=period,
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                    progress=False,
                )
                break
            except Exception as exc:  # noqa: BLE001 - network calls can fail in many ways
                if attempt == retries:
                    print(f"    Batch {idx} failed after {retries + 1} attempts: {exc}")
                else:
                    time.sleep(2)

        if data is None or data.empty:
            continue

        for ticker in chunk:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    df = data[ticker].dropna(how="all")
                else:
                    df = data.dropna(how="all")
            except KeyError:
                continue
            df = df.dropna(subset=["Close"])
            if len(df) >= MIN_ROWS_REQUIRED:
                result[ticker] = df

    return result
