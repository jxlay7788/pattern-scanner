"""Composite buy/sell detector: combines moving averages, RSI, MACD, support/
resistance, and volume into a single scored signal.

Each indicator casts one vote for the day it fires (+1 bullish / -1 bearish);
a volume spike amplifies whatever direction the other indicators already
point to, rather than voting a direction of its own. `action` fires once the
net score crosses `buy_threshold` / `sell_threshold`.
"""

import pandas as pd

from src.technical.macd import detect_macd_crossovers
from src.technical.moving_averages import detect_crossover_signals
from src.technical.rsi import detect_rsi_signals
from src.technical.support_resistance import detect_support_resistance_signals
from src.technical.volume import detect_volume_signals

BUY_THRESHOLD = 2
SELL_THRESHOLD = -2


def build_composite_signals(
    df: pd.DataFrame,
    short_window: int = 50,
    long_window: int = 200,
    rsi_window: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sr_window: int = 20,
    volume_window: int = 20,
) -> pd.DataFrame:
    """Run every individual detector and merge their signal columns onto one DataFrame."""
    out = detect_crossover_signals(df, short_window, long_window)
    out = detect_rsi_signals(out, rsi_window)
    out = detect_macd_crossovers(out, macd_fast, macd_slow, macd_signal)
    out = detect_support_resistance_signals(out, sr_window)
    out = detect_volume_signals(out, volume_window)
    return out


def _score_row(row: pd.Series) -> int:
    score = 0
    if row.get("crossover") == "golden_cross":
        score += 1
    elif row.get("crossover") == "death_cross":
        score -= 1

    if row.get("rsi_signal") == "oversold":
        score += 1
    elif row.get("rsi_signal") == "overbought":
        score -= 1

    if row.get("macd_crossover") == "bullish":
        score += 1
    elif row.get("macd_crossover") == "bearish":
        score -= 1

    if row.get("sr_signal") == "support_bounce":
        score += 1
    elif row.get("sr_signal") == "resistance_rejection":
        score -= 1

    if row.get("volume_signal") == "volume_spike" and score != 0:
        score += 1 if score > 0 else -1

    return score


def detect_buy_sell_signals(
    df: pd.DataFrame,
    buy_threshold: int = BUY_THRESHOLD,
    sell_threshold: int = SELL_THRESHOLD,
    **kwargs,
) -> pd.DataFrame:
    """Add every individual indicator plus a `signal_score` (int) and a final
    `action` column ('BUY' / 'SELL' / None) once enough indicators agree on the
    same day.
    """
    out = build_composite_signals(df, **kwargs)
    out["signal_score"] = out.apply(_score_row, axis=1)

    out["action"] = None
    out.loc[out["signal_score"] >= buy_threshold, "action"] = "BUY"
    out.loc[out["signal_score"] <= sell_threshold, "action"] = "SELL"
    return out


def build_ma_rsi_macd_signals(
    df: pd.DataFrame,
    short_window: int = 50,
    long_window: int = 200,
    rsi_window: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
    """Lighter 3-indicator variant: MA crossover + RSI momentum + MACD crossover
    only (no support/resistance, no volume).
    """
    out = detect_crossover_signals(df, short_window, long_window)
    out = detect_rsi_signals(out, rsi_window)
    out = detect_macd_crossovers(out, macd_fast, macd_slow, macd_signal)

    # "RSI going up" = momentum crossing its own midline (50), rather than the
    # oversold/overbought extremes `rsi_signal` already flags -- a rising RSI
    # crossing above 50 signals strengthening bullish momentum, and vice versa.
    rsi_col = f"rsi_{rsi_window}"
    above_mid = out[rsi_col] > 50
    out["rsi_momentum"] = None
    out.loc[above_mid & ~above_mid.shift(1).fillna(False), "rsi_momentum"] = "turning_up"
    out.loc[~above_mid & above_mid.shift(1).fillna(False), "rsi_momentum"] = "turning_down"
    return out


def _score_ma_rsi_macd_row(row: pd.Series) -> int:
    score = 0
    if row.get("crossover") == "golden_cross":
        score += 1
    elif row.get("crossover") == "death_cross":
        score -= 1

    if row.get("rsi_momentum") == "turning_up":
        score += 1
    elif row.get("rsi_momentum") == "turning_down":
        score -= 1

    if row.get("macd_crossover") == "bullish":
        score += 1
    elif row.get("macd_crossover") == "bearish":
        score -= 1

    return score


def detect_ma_rsi_macd_signals(
    df: pd.DataFrame,
    buy_threshold: int = 2,
    sell_threshold: int = -2,
    **kwargs,
) -> pd.DataFrame:
    """Add MA crossover + RSI momentum + MACD crossover plus a `signal_score`
    and `action` ('BUY' / 'SELL' / None) once at least 2 of the 3 agree.
    """
    out = build_ma_rsi_macd_signals(df, **kwargs)
    out["signal_score"] = out.apply(_score_ma_rsi_macd_row, axis=1)

    out["action"] = None
    out.loc[out["signal_score"] >= buy_threshold, "action"] = "BUY"
    out.loc[out["signal_score"] <= sell_threshold, "action"] = "SELL"
    return out


def build_ma_macd_signals(
    df: pd.DataFrame,
    short_window: int = 50,
    long_window: int = 200,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
    """2-indicator variant: MA crossover + MACD crossover only (RSI dropped)."""
    out = detect_crossover_signals(df, short_window, long_window)
    out = detect_macd_crossovers(out, macd_fast, macd_slow, macd_signal)
    return out


def _score_ma_macd_row(row: pd.Series) -> int:
    score = 0
    if row.get("crossover") == "golden_cross":
        score += 1
    elif row.get("crossover") == "death_cross":
        score -= 1

    if row.get("macd_crossover") == "bullish":
        score += 1
    elif row.get("macd_crossover") == "bearish":
        score -= 1

    return score


def detect_ma_macd_signals(
    df: pd.DataFrame,
    buy_threshold: int = 2,
    sell_threshold: int = -2,
    **kwargs,
) -> pd.DataFrame:
    """Add MA crossover + MACD crossover plus a `signal_score` and `action`
    ('BUY' / 'SELL' / None) -- with only 2 votes and a threshold of 2, both
    have to agree on the same day to fire.
    """
    out = build_ma_macd_signals(df, **kwargs)
    out["signal_score"] = out.apply(_score_ma_macd_row, axis=1)

    out["action"] = None
    out.loc[out["signal_score"] >= buy_threshold, "action"] = "BUY"
    out.loc[out["signal_score"] <= sell_threshold, "action"] = "SELL"
    return out


def build_ma_rsi_sr_macd_signals(
    df: pd.DataFrame,
    short_window: int = 50,
    long_window: int = 200,
    rsi_window: int = 14,
    rsi_bull_level: float = 55,
    rsi_bear_level: float = 45,
    sr_window: int = 20,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
    """4-indicator variant: MA crossover + RSI threshold + support/resistance +
    MACD crossover (no volume).

    RSI uses two separate levels (55 bull / 45 bear, not a single midline) so a
    day where RSI is just drifting around 50 doesn't flip the vote back and
    forth on every tiny wiggle -- it has to actually clear a stronger threshold
    in either direction, and holds that vote until it clears the *other* level.
    """
    out = detect_crossover_signals(df, short_window, long_window)
    out = detect_rsi_signals(out, rsi_window)
    out = detect_support_resistance_signals(out, sr_window)
    out = detect_macd_crossovers(out, macd_fast, macd_slow, macd_signal)

    rsi_col = f"rsi_{rsi_window}"
    bullish = out[rsi_col] >= rsi_bull_level
    bearish = out[rsi_col] <= rsi_bear_level
    rsi_state = pd.Series(None, index=out.index, dtype="object")
    rsi_state[bullish] = "bullish"
    rsi_state[bearish] = "bearish"
    out["rsi_level_state"] = rsi_state.ffill()
    return out


def _score_ma_rsi_sr_macd_row(row: pd.Series) -> int:
    score = 0
    if row.get("crossover") == "golden_cross":
        score += 1
    elif row.get("crossover") == "death_cross":
        score -= 1

    if row.get("rsi_level_state") == "bullish":
        score += 1
    elif row.get("rsi_level_state") == "bearish":
        score -= 1

    if row.get("sr_signal") == "support_bounce":
        score += 1
    elif row.get("sr_signal") == "resistance_rejection":
        score -= 1

    if row.get("macd_crossover") == "bullish":
        score += 1
    elif row.get("macd_crossover") == "bearish":
        score -= 1

    return score


def detect_ma_rsi_sr_macd_signals(
    df: pd.DataFrame,
    buy_threshold: int = 2,
    sell_threshold: int = -2,
    **kwargs,
) -> pd.DataFrame:
    """Add MA crossover + RSI threshold + support/resistance + MACD crossover
    plus a `signal_score` and `action` ('BUY' / 'SELL' / None) once at least
    2 of the 4 agree.
    """
    out = build_ma_rsi_sr_macd_signals(df, **kwargs)
    out["signal_score"] = out.apply(_score_ma_rsi_sr_macd_row, axis=1)

    out["action"] = None
    out.loc[out["signal_score"] >= buy_threshold, "action"] = "BUY"
    out.loc[out["signal_score"] <= sell_threshold, "action"] = "SELL"
    return out
