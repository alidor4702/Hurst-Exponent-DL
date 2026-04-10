"""
Part 5: Real data download, rolling-window H estimation, and trading strategy.

Pipeline:
  1. Download daily prices from Yahoo Finance
  2. Compute log returns
  3. Build rolling-window matrix (each row = T consecutive log returns)
  4. Per-sample rescale each window (same as training)
  5. Predict H for each window using saved Dense, CNN, Ensemble models
  6. Generate trading signals based on H deviation from 0.5
  7. Compute cumulative strategy returns using real (simple) returns
"""

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from pathlib import Path

from models.architectures.dense import DenseMedium
from models.architectures.cnn import HurstCNN
from models.architectures.ensemble import EnsembleMetaLearner


def download_asset(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""
    print(f"Downloading {ticker}...")
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    print(f"  {ticker}: {len(data)} days, {data.index[0].date()} to {data.index[-1].date()}")
    return data


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns: r_t = ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()


def build_rolling_windows(returns: np.ndarray, window_size: int = 100) -> np.ndarray:
    """
    Build the input matrix X as specified in the TP:
        X = [[r_1, ..., r_T],
             [r_2, ..., r_{T+1}],
             ...]
    """
    n = len(returns)
    n_windows = n - window_size + 1
    X = np.zeros((n_windows, window_size))
    for i in range(n_windows):
        X[i] = returns[i : i + window_size]
    return X


def rescale_per_sample(X: np.ndarray) -> np.ndarray:
    """Per-sample standardization: zero mean, unit variance per row."""
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds[stds < 1e-10] = 1.0
    return (X - means) / stds


def load_models(device: torch.device, save_dir: str = "models/saved"):
    """Load the three pre-trained models."""
    save_dir = Path(save_dir)

    dense = DenseMedium(input_size=100)
    dense.load_state_dict(torch.load(save_dir / "dense_medium_best.pt", map_location=device, weights_only=True))
    dense.to(device).eval()

    cnn = HurstCNN(input_size=100)
    cnn.load_state_dict(torch.load(save_dir / "cnn_stone_best.pt", map_location=device, weights_only=True))
    cnn.to(device).eval()

    ensemble = EnsembleMetaLearner(n_inputs=2)
    ensemble.load_state_dict(torch.load(save_dir / "ensemble_basic_best.pt", map_location=device, weights_only=True))
    ensemble.to(device).eval()

    return dense, cnn, ensemble


def predict_hurst(X: np.ndarray, dense, cnn, ensemble, device: torch.device) -> dict:
    """Predict H for each window using Dense, CNN, and Ensemble."""
    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        h_dense = dense(X_tensor).cpu().numpy()
        h_cnn = cnn(X_tensor).cpu().numpy()
        ens_input = torch.stack([
            torch.FloatTensor(h_dense),
            torch.FloatTensor(h_cnn),
        ], dim=1).to(device)
        h_ensemble = ensemble(ens_input).cpu().numpy()

    return {"dense": h_dense, "cnn": h_cnn, "ensemble": h_ensemble}


def generate_signals(h_values: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """
    H > 0.5 + threshold -> +1 (long)
    H < 0.5 - threshold -> -1 (short)
    else                -> 0  (flat)
    """
    signals = np.zeros_like(h_values)
    signals[h_values > 0.5 + threshold] = 1.0
    signals[h_values < 0.5 - threshold] = -1.0
    return signals


def compute_strategy_returns(
    signals: np.ndarray,
    simple_returns: np.ndarray,
    transaction_cost: float = 0.001,
) -> dict:
    """Compute strategy cumulative returns. Yesterday's signal, today's return."""
    positions = signals[:-1]
    returns = simple_returns[1:]

    strategy_returns = positions * returns

    position_changes = np.abs(np.diff(np.concatenate([[0], positions])))
    costs = position_changes * transaction_cost
    strategy_returns_after_costs = strategy_returns - costs

    cumulative_strategy = np.cumprod(1 + strategy_returns_after_costs)
    cumulative_buy_hold = np.cumprod(1 + returns)

    n_trades = np.sum(position_changes > 0)
    pct_long = np.mean(positions == 1) * 100
    pct_short = np.mean(positions == -1) * 100
    pct_flat = np.mean(positions == 0) * 100

    if np.std(strategy_returns_after_costs) > 0:
        sharpe = np.mean(strategy_returns_after_costs) / np.std(strategy_returns_after_costs) * np.sqrt(252)
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(cumulative_strategy)
    drawdown = (peak - cumulative_strategy) / peak
    max_drawdown = np.max(drawdown)

    if np.std(returns) > 0:
        sharpe_bh = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe_bh = 0.0
    peak_bh = np.maximum.accumulate(cumulative_buy_hold)
    dd_bh = (peak_bh - cumulative_buy_hold) / peak_bh
    max_dd_bh = np.max(dd_bh)

    return {
        "strategy_returns": strategy_returns_after_costs,
        "cumulative_strategy": cumulative_strategy,
        "cumulative_buy_hold": cumulative_buy_hold,
        "positions": positions,
        "n_trades": int(n_trades),
        "total_days": len(returns),
        "pct_long": pct_long,
        "pct_short": pct_short,
        "pct_flat": pct_flat,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "total_return": cumulative_strategy[-1] - 1,
        "bh_return": cumulative_buy_hold[-1] - 1,
        "bh_sharpe": sharpe_bh,
        "bh_max_drawdown": max_dd_bh,
    }


def run_full_pipeline(
    ticker: str,
    device: torch.device,
    dense, cnn, ensemble,
    window_size: int = 100,
    threshold: float = 0.05,
    transaction_cost: float = 0.001,
    period: str = "10y",
) -> dict:
    """Run the full pipeline for one asset."""
    data = download_asset(ticker, period=period)
    close = data["Close"]

    log_returns = compute_log_returns(close)
    simple_returns = (close / close.shift(1) - 1).dropna()

    common_dates = log_returns.index.intersection(simple_returns.index)
    log_returns = log_returns.loc[common_dates]
    simple_returns = simple_returns.loc[common_dates]

    X_windows = build_rolling_windows(log_returns.values, window_size)
    X_rescaled = rescale_per_sample(X_windows)

    h_estimates = predict_hurst(X_rescaled, dense, cnn, ensemble, device)
    h_dates = log_returns.index[window_size - 1:]

    signals = generate_signals(h_estimates["ensemble"], threshold)
    aligned_returns = simple_returns.loc[h_dates].values

    results = compute_strategy_returns(signals, aligned_returns, transaction_cost)
    strategy_dates = h_dates[1:]

    return {
        "ticker": ticker,
        "close": close,
        "log_returns": log_returns,
        "simple_returns": simple_returns,
        "h_estimates": h_estimates,
        "h_dates": h_dates,
        "signals": signals,
        "strategy_dates": strategy_dates,
        "threshold": threshold,
        **results,
    }
