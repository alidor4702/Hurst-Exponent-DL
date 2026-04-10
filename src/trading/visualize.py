"""
Visualization functions for Part 5: Real data trading.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from src.utils.plotting import setup_style, save_fig, COLORS

setup_style()


def plot_rolling_hurst(result: dict, save_path: str):
    """Plot rolling H estimates over time with regime shading."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    ticker = result["ticker"]
    dates = result["h_dates"]

    for ax, (name, label) in zip(axes, [
        ("dense", "Dense"),
        ("cnn", "CNN (Stone 2020)"),
        ("ensemble", "Ensemble"),
    ]):
        h = result["h_estimates"][name]
        ax.plot(dates, h, linewidth=0.6, alpha=0.8, color=COLORS["primary"])

        # Shade regions
        ax.axhspan(0, 0.45, alpha=0.06, color=COLORS["mean_reverting"])
        ax.axhspan(0.55, 1.0, alpha=0.06, color=COLORS["trending"])
        ax.axhline(0.5, color=COLORS["neutral"], linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(0.5 + result["threshold"], color=COLORS["danger"], linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axhline(0.5 - result["threshold"], color=COLORS["mean_reverting"], linestyle=":", linewidth=0.8, alpha=0.5)

        ax.set_ylabel(f"H ({label})")
        ax.set_ylim(0.1, 0.9)
        ax.legend([f"Mean H = {np.mean(h):.3f}"], loc="upper right", fontsize=9)

    axes[0].set_title(f"{ticker} — Rolling Hurst Exponent Estimates (window = 100 days)")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)

    save_fig(fig, save_path)


def plot_strategy_vs_buyhold(result: dict, save_path: str):
    """Plot cumulative returns: strategy vs buy-and-hold."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ticker = result["ticker"]
    dates = result["strategy_dates"]

    ax.plot(dates, result["cumulative_strategy"], linewidth=1.2,
            color=COLORS["primary"], label="Hurst Strategy")
    ax.plot(dates, result["cumulative_buy_hold"], linewidth=1.2,
            color=COLORS["neutral"], alpha=0.7, label="Buy & Hold")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.5, alpha=0.3)

    ax.set_title(f"{ticker} — Cumulative Returns: Hurst Strategy vs Buy & Hold")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Add performance box
    stats_text = (
        f"Strategy: {result['total_return']:+.1%} | Sharpe: {result['sharpe']:.2f} | MaxDD: {result['max_drawdown']:.1%}\n"
        f"Buy&Hold: {result['bh_return']:+.1%} | Sharpe: {result['bh_sharpe']:.2f} | MaxDD: {result['bh_max_drawdown']:.1%}"
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="gray"))

    save_fig(fig, save_path)


def plot_signals_on_price(result: dict, save_path: str):
    """Plot price with colored background showing trading positions."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    ticker = result["ticker"]

    # Top: Price with signal coloring
    ax1 = axes[0]
    dates = result["strategy_dates"]
    # Get close prices for strategy dates
    close_aligned = result["close"].reindex(dates).ffill()

    ax1.plot(dates, close_aligned.values, linewidth=0.8, color="black")

    # Color background by position
    positions = result["positions"]
    for i in range(len(dates) - 1):
        if positions[i] == 1:
            ax1.axvspan(dates[i], dates[i + 1], alpha=0.15, color=COLORS["success"], linewidth=0)
        elif positions[i] == -1:
            ax1.axvspan(dates[i], dates[i + 1], alpha=0.15, color=COLORS["danger"], linewidth=0)

    legend_elements = [
        Patch(facecolor=COLORS["success"], alpha=0.3, label="Long (H > 0.55)"),
        Patch(facecolor=COLORS["danger"], alpha=0.3, label="Short (H < 0.45)"),
        Patch(facecolor="white", edgecolor="gray", label="Flat"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax1.set_title(f"{ticker} — Price with Trading Signals")
    ax1.set_ylabel("Price")

    # Bottom: H estimate
    ax2 = axes[1]
    h_dates = result["h_dates"]
    h_ens = result["h_estimates"]["ensemble"]
    ax2.plot(h_dates, h_ens, linewidth=0.6, color=COLORS["primary"])
    ax2.axhline(0.5, color=COLORS["neutral"], linestyle="--", linewidth=1)
    ax2.axhline(0.5 + result["threshold"], color=COLORS["danger"], linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.axhline(0.5 - result["threshold"], color=COLORS["mean_reverting"], linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.set_ylabel("H (Ensemble)")
    ax2.set_ylim(0.15, 0.85)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    save_fig(fig, save_path)


def plot_h_distribution(result: dict, save_path: str):
    """Histogram of H estimates across all windows."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ticker = result["ticker"]

    for name, label, color in [
        ("dense", "Dense", COLORS["secondary"]),
        ("cnn", "CNN", COLORS["danger"]),
        ("ensemble", "Ensemble", COLORS["primary"]),
    ]:
        h = result["h_estimates"][name]
        ax.hist(h, bins=60, alpha=0.4, label=f"{label} (mean={np.mean(h):.3f})", color=color, edgecolor="white")

    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="H = 0.5 (random walk)")
    ax.set_xlabel("Estimated Hurst Exponent")
    ax.set_ylabel("Count")
    ax.set_title(f"{ticker} — Distribution of Rolling H Estimates")
    ax.legend()

    save_fig(fig, save_path)


def plot_position_breakdown(result: dict, save_path: str):
    """Pie chart of time spent in each position."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ticker = result["ticker"]

    sizes = [result["pct_long"], result["pct_flat"], result["pct_short"]]
    labels = [
        f"Long ({result['pct_long']:.1f}%)",
        f"Flat ({result['pct_flat']:.1f}%)",
        f"Short ({result['pct_short']:.1f}%)",
    ]
    colors_list = [COLORS["success"], COLORS["neutral"], COLORS["danger"]]
    explode = (0.03, 0.03, 0.03)

    ax.pie(sizes, labels=labels, colors=colors_list, explode=explode,
           autopct="", startangle=90, textprops={"fontsize": 12})
    ax.set_title(f"{ticker} — Position Breakdown\n({result['n_trades']} trades over {result['total_days']} days)")

    save_fig(fig, save_path)


def plot_multi_asset_comparison(all_results: list, save_path: str):
    """Compare strategy performance across all assets."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Cumulative returns comparison
    ax = axes[0, 0]
    for r in all_results:
        ax.plot(r["strategy_dates"], r["cumulative_strategy"], linewidth=1.2, label=r["ticker"])
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.5, alpha=0.3)
    ax.set_title("Strategy Cumulative Returns")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 2. Mean H by asset
    ax = axes[0, 1]
    tickers = [r["ticker"] for r in all_results]
    mean_h = [np.mean(r["h_estimates"]["ensemble"]) for r in all_results]
    bar_colors = [COLORS["trending"] if h > 0.5 else COLORS["mean_reverting"] for h in mean_h]
    bars = ax.bar(tickers, mean_h, color=bar_colors, edgecolor="white", width=0.5)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_title("Mean Hurst Exponent by Asset")
    ax.set_ylabel("Mean H")
    for bar, h in zip(bars, mean_h):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    # 3. Sharpe comparison
    ax = axes[1, 0]
    x = np.arange(len(tickers))
    w = 0.3
    ax.bar(x - w / 2, [r["sharpe"] for r in all_results], w,
           label="Strategy", color=COLORS["primary"], edgecolor="white")
    ax.bar(x + w / 2, [r["bh_sharpe"] for r in all_results], w,
           label="Buy & Hold", color=COLORS["neutral"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_title("Sharpe Ratio Comparison")
    ax.set_ylabel("Annualized Sharpe")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()

    # 4. Summary table
    ax = axes[1, 1]
    ax.axis("off")
    headers = ["Asset", "Strat Return", "B&H Return", "Sharpe", "MaxDD", "Trades"]
    rows = []
    for r in all_results:
        rows.append([
            r["ticker"],
            f"{r['total_return']:+.1%}",
            f"{r['bh_return']:+.1%}",
            f"{r['sharpe']:.2f}",
            f"{r['max_drawdown']:.1%}",
            str(r["n_trades"]),
        ])

    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(COLORS["primary"])
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F8F9FA" if row % 2 == 0 else "white")
    ax.set_title("Performance Summary", pad=20)

    fig.suptitle("Multi-Asset Hurst Trading Strategy Comparison", fontsize=16, fontweight="bold", y=1.02)
    save_fig(fig, save_path)


def plot_rolling_h_all_assets(all_results: list, save_path: str):
    """Overlay rolling H for all assets on one plot."""
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = [COLORS["primary"], COLORS["danger"], COLORS["success"],
              COLORS["secondary"], COLORS["accent"]]

    for r, c in zip(all_results, colors):
        h = r["h_estimates"]["ensemble"]
        # Smooth with rolling mean for readability
        h_smooth = pd.Series(h, index=r["h_dates"]).rolling(20, min_periods=1).mean()
        ax.plot(h_smooth.index, h_smooth.values, linewidth=1, label=r["ticker"], color=c, alpha=0.85)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2)
    ax.axhspan(0, 0.45, alpha=0.04, color=COLORS["mean_reverting"])
    ax.axhspan(0.55, 1.0, alpha=0.04, color=COLORS["trending"])
    ax.set_title("Rolling Hurst Exponent — All Assets (20-day smoothed)")
    ax.set_ylabel("H (Ensemble)")
    ax.set_ylim(0.2, 0.8)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    save_fig(fig, save_path)


# Need pandas import for rolling mean
import pandas as pd
