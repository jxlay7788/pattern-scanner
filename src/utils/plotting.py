"""Shared matplotlib helpers for statistical pattern charts."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REGIME_COLORS = {"low": "tab:green", "normal": "tab:gray", "high": "tab:red"}


def plot_volatility_regime(df: pd.DataFrame, regime: pd.Series, ticker: str, save_path: Path) -> None:
    """Plot Close price with background shading indicating the volatility regime."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df.index, df["Close"], color="black", linewidth=1, label="Close")

    # Shade contiguous runs of the same regime so the background reads as bands
    # rather than one axvspan per day.
    regime_filled = regime.ffill()
    run_id = (regime_filled != regime_filled.shift()).cumsum()
    for _, run in regime_filled.groupby(run_id):
        label = run.iloc[0]
        if pd.isna(label):
            continue
        ax.axvspan(run.index[0], run.index[-1], color=REGIME_COLORS.get(label, "tab:gray"), alpha=0.15)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.3) for color in REGIME_COLORS.values()]
    ax.legend(handles + ax.get_legend_handles_labels()[0], list(REGIME_COLORS.keys()) + ["Close"], loc="upper left")
    ax.set_title(f"{ticker}: Price with Volatility Regime")
    ax.set_ylabel("Price ($)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(corr: pd.DataFrame, save_path: Path) -> None:
    """Plot a correlation matrix as an annotated heatmap."""
    fig, ax = plt.subplots(figsize=(1.2 * len(corr.columns) + 2, 1.2 * len(corr.columns) + 2))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title("Rolling Correlation Matrix")
    fig.colorbar(im, ax=ax, label="correlation")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
