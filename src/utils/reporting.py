"""Markdown report-generation helpers."""

import pandas as pd

RELIABILITY_CAVEAT = (
    "**Caveat:** the figures below describe how often each signal has "
    "*historically* been followed by a positive return, and by how much, on "
    "this ticker's own price history. This is a backward-looking frequency, "
    "not a guarantee of future performance -- market conditions change, and "
    "past patterns can stop working. Signals that fire rarely (a low "
    "`Instances` count) can show a win rate or average return that swings "
    "wildly based on a handful of data points, so treat those rows with "
    "extra skepticism."
)

SIGNAL_LABELS = {
    "golden_cross": "Golden cross (50/200-day SMA)",
    "rsi_oversold": "RSI oversold (<=30)",
    "rsi_overbought": "RSI overbought (>=70)",
    "bollinger_upper_touch": "Bollinger upper-band touch",
    "bollinger_lower_touch": "Bollinger lower-band touch",
    "high_mean_reversion_zscore": "High mean-reversion z-score (|z|>=2)",
}


def _fmt_pct(value: float, signed: bool = False) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:+.2%}" if signed else f"{value:.1%}"


def render_reliability_table(df: pd.DataFrame) -> str:
    """Render one ticker's signal-reliability rows as a markdown table."""
    header = (
        "| Signal | Horizon (days) | Instances | Win Rate | Avg Fwd Return "
        "| Baseline Win Rate | Baseline Avg Return | Win Rate Edge | Avg Return Edge |"
    )
    separator = "|---|---|---|---|---|---|---|---|---|"
    lines = [header, separator]

    for _, row in df.iterrows():
        label = SIGNAL_LABELS.get(row["signal"], row["signal"])
        lines.append(
            "| {signal} | {horizon} | {n} | {win} | {avg} | {base_win} | {base_avg} | {win_edge} | {avg_edge} |".format(
                signal=label,
                horizon=int(row["horizon_days"]),
                n=int(row["num_instances"]),
                win=_fmt_pct(row["win_rate"]),
                avg=_fmt_pct(row["avg_forward_return"], signed=True),
                base_win=_fmt_pct(row["baseline_win_rate"]),
                base_avg=_fmt_pct(row["baseline_avg_return"], signed=True),
                win_edge=_fmt_pct(row["win_rate_edge"], signed=True),
                avg_edge=_fmt_pct(row["avg_return_edge"], signed=True),
            )
        )
    return "\n".join(lines)


def render_reliability_section(reliability_by_ticker: dict[str, pd.DataFrame]) -> str:
    """Render the full 'Signal Reliability' report.md section: a caveat followed
    by one table per ticker.
    """
    parts = ["## Signal Reliability", "", RELIABILITY_CAVEAT, ""]
    for ticker, df in reliability_by_ticker.items():
        parts.append(f"### {ticker}")
        parts.append("")
        parts.append(render_reliability_table(df))
        parts.append("")
    return "\n".join(parts)
