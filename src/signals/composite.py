"""Combines RSI, MACD, moving-average crossover, and Bollinger Bands into a
single weighted buy/sell score per ticker."""

import pandas as pd

from src.indicators.bollinger import bollinger_signal, compute_bollinger_bands
from src.indicators.macd import compute_macd, macd_signal
from src.indicators.moving_averages import compute_sma, ma_crossover_signal, ma_trend_signal
from src.indicators.rsi import compute_rsi, rsi_signal

# Golden/death crosses are the strongest confirmed trend-change signal, so they
# carry double weight; a raw trend read (short MA vs long MA, no fresh cross)
# only confirms direction, so it carries half weight.
WEIGHTS = {
    "rsi": 1.0,
    "macd": 1.0,
    "ma_cross": 2.0,
    "ma_trend": 0.5,
    "bollinger": 1.0,
}

LABEL_THRESHOLDS = [
    (2.5, "STRONG BUY"),
    (1.0, "BUY"),
    (-0.99, "HOLD"),
    (-2.49, "SELL"),
]
STRONG_SELL_LABEL = "STRONG SELL"


def score_to_label(score: float) -> str:
    for threshold, label in LABEL_THRESHOLDS:
        if score >= threshold:
            return label
    return STRONG_SELL_LABEL


def analyze_ticker(ticker: str, df: pd.DataFrame) -> dict | None:
    """Computes indicators and a composite signal for the latest session.

    Returns None if there isn't enough history to compute all indicators.
    """
    close = df["Close"]
    if len(close) < 200:
        return None

    rsi = compute_rsi(close)
    macd_line, macd_sig, _ = compute_macd(close)
    sma50 = compute_sma(close, 50)
    sma200 = compute_sma(close, 200)
    upper, mid, lower = compute_bollinger_bands(close)

    if pd.isna(sma200.iloc[-1]):
        return None

    s_rsi = rsi_signal(rsi.iloc[-1])
    s_macd = macd_signal(macd_line, macd_sig)
    s_cross = ma_crossover_signal(sma50, sma200)
    s_trend = ma_trend_signal(sma50, sma200)
    s_boll = bollinger_signal(close.iloc[-1], upper.iloc[-1], lower.iloc[-1])

    score = (
        s_rsi * WEIGHTS["rsi"]
        + s_macd * WEIGHTS["macd"]
        + s_cross * WEIGHTS["ma_cross"]
        + s_trend * WEIGHTS["ma_trend"]
        + s_boll * WEIGHTS["bollinger"]
    )

    return {
        "ticker": ticker,
        "close": round(float(close.iloc[-1]), 2),
        "score": round(score, 2),
        "signal": score_to_label(score),
        "rsi": round(float(rsi.iloc[-1]), 1) if not pd.isna(rsi.iloc[-1]) else None,
        "rsi_signal": s_rsi,
        "macd_signal": s_macd,
        "ma_cross_signal": s_cross,
        "ma_trend_signal": s_trend,
        "sma50": round(float(sma50.iloc[-1]), 2),
        "sma200": round(float(sma200.iloc[-1]), 2),
        "bollinger_signal": s_boll,
        "bb_upper": round(float(upper.iloc[-1]), 2) if not pd.isna(upper.iloc[-1]) else None,
        "bb_lower": round(float(lower.iloc[-1]), 2) if not pd.isna(lower.iloc[-1]) else None,
    }


def analyze_universe(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ticker, df in price_data.items():
        row = analyze_ticker(ticker, df)
        if row is not None:
            rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
    return result
