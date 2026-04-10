"""
Part 5: Application to Real Data
================================
Downloads real financial data, estimates rolling Hurst exponents using
our pre-trained models, and backtests a simple trading strategy.

Assets: META (stock), EURCHF=X (FX), BTC-USD (crypto), SPY (index)

Usage:
    python run_part5_real_data.py
    python run_part5_real_data.py --plot-only   # skip download, use cached
"""

import argparse
import sys
import joblib
from pathlib import Path

from src.utils.config import get_device
from src.trading.real_data import load_models, run_full_pipeline
from src.trading.visualize import (
    plot_rolling_hurst,
    plot_strategy_vs_buyhold,
    plot_signals_on_price,
    plot_h_distribution,
    plot_position_breakdown,
    plot_multi_asset_comparison,
    plot_rolling_h_all_assets,
)

ASSETS = [
    "META",       # Facebook/Meta — US tech stock
    "EURCHF=X",   # EUR/CHF — FX pair (often mean-reverting)
    "BTC-USD",    # Bitcoin — crypto (strong trends)
    "SPY",        # S&P 500 ETF — broad market index
]

PLOT_DIR = Path("plots/real_data")
CACHE_DIR = Path("data/real_data")


def main():
    parser = argparse.ArgumentParser(description="Part 5: Real data trading")
    parser.add_argument("--plot-only", action="store_true", help="Only regenerate plots from cached results")
    args = parser.parse_args()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    if args.plot_only:
        print("Loading cached results...")
        all_results = joblib.load(CACHE_DIR / "all_results.joblib")
    else:
        # Load pre-trained models
        print("Loading pre-trained models...")
        dense, cnn, ensemble = load_models(device)
        print("  Models loaded successfully.\n")

        # Run pipeline for each asset
        all_results = []
        for ticker in ASSETS:
            print(f"\n{'='*60}")
            print(f"  Processing {ticker}")
            print(f"{'='*60}")
            try:
                result = run_full_pipeline(
                    ticker=ticker,
                    device=device,
                    dense=dense, cnn=cnn, ensemble=ensemble,
                    window_size=100,
                    threshold=0.05,
                    transaction_cost=0.001,
                    period="10y",
                )
                all_results.append(result)

                # Print summary
                print(f"\n  Results for {ticker}:")
                print(f"    Strategy return: {result['total_return']:+.2%}")
                print(f"    Buy&Hold return: {result['bh_return']:+.2%}")
                print(f"    Sharpe ratio:    {result['sharpe']:.3f}")
                print(f"    Max drawdown:    {result['max_drawdown']:.1%}")
                print(f"    Trades:          {result['n_trades']}")
                print(f"    Position split:  {result['pct_long']:.0f}% long / {result['pct_flat']:.0f}% flat / {result['pct_short']:.0f}% short")

            except Exception as e:
                print(f"  ERROR with {ticker}: {e}")
                continue

        # Cache results
        joblib.dump(all_results, CACHE_DIR / "all_results.joblib")
        print(f"\nResults cached to {CACHE_DIR / 'all_results.joblib'}")

    # ── Generate plots ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Generating plots")
    print(f"{'='*60}\n")

    # Per-asset plots
    for result in all_results:
        ticker = result["ticker"].replace("=", "").replace("-", "")
        prefix = f"{PLOT_DIR}/{ticker}"

        plot_rolling_hurst(result, f"{prefix}_rolling_hurst.png")
        plot_strategy_vs_buyhold(result, f"{prefix}_strategy.png")
        plot_signals_on_price(result, f"{prefix}_signals.png")
        plot_h_distribution(result, f"{prefix}_h_distribution.png")
        plot_position_breakdown(result, f"{prefix}_positions.png")

    # Cross-asset comparison plots
    if len(all_results) >= 2:
        plot_multi_asset_comparison(all_results, f"{PLOT_DIR}/multi_asset_comparison.png")
        plot_rolling_h_all_assets(all_results, f"{PLOT_DIR}/all_assets_rolling_h.png")

    print(f"\nAll plots saved to {PLOT_DIR}/")
    print("Part 5 complete!")


if __name__ == "__main__":
    main()
