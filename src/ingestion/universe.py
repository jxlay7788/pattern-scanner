"""Fetches the current S&P 500 ticker list from Wikipedia."""

import io

import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Wikipedia rejects requests with no User-Agent header (HTTP 403).
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; pattern-scanner/1.0)"}


def get_sp500_tickers() -> list[str]:
    response = requests.get(WIKI_URL, headers=HEADERS, timeout=15)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0]
    tickers = df["Symbol"].astype(str).str.strip().tolist()
    # yfinance uses '-' where Wikipedia uses '.' (e.g. BRK.B -> BRK-B)
    return [t.replace(".", "-") for t in tickers]


def load_tickers_from_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
