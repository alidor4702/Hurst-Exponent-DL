"""
Classical Hurst exponent estimators: R/S analysis and DFA.
Used as baselines to compare against neural network performance.
"""

import numpy as np
from tqdm import tqdm


def rs_hurst(x: np.ndarray, min_window: int = 10) -> float:
    """
    Estimate Hurst exponent using Rescaled Range (R/S) analysis.

    For each window size n, compute R/S, then fit log(R/S) vs log(n)
    to get the slope = H.
    """
    n = len(x)
    # Window sizes: powers of 2 that fit in the series
    window_sizes = []
    size = min_window
    while size <= n // 2:
        window_sizes.append(size)
        size = int(size * 1.5)

    if len(window_sizes) < 3:
        return 0.5  # not enough data

    rs_values = []
    for w in window_sizes:
        n_windows = n // w
        rs_list = []
        for i in range(n_windows):
            segment = x[i * w:(i + 1) * w]
            mean = segment.mean()
            devs = np.cumsum(segment - mean)
            R = devs.max() - devs.min()
            S = segment.std(ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append(np.mean(rs_list))
        else:
            rs_values.append(np.nan)

    # Filter out NaNs
    valid = ~np.isnan(rs_values)
    if valid.sum() < 3:
        return 0.5

    log_n = np.log(np.array(window_sizes)[valid])
    log_rs = np.log(np.array(rs_values)[valid])

    # Linear regression: log(R/S) = H * log(n) + c
    coeffs = np.polyfit(log_n, log_rs, 1)
    H = np.clip(coeffs[0], 0.01, 0.99)
    return H


def dfa_hurst(x: np.ndarray, min_window: int = 4) -> float:
    """
    Estimate Hurst exponent using Detrended Fluctuation Analysis (DFA).

    Computes fluctuation function F(n) for different window sizes,
    then fits log(F) vs log(n) to get the slope = H.
    """
    n = len(x)
    # Cumulative sum (profile)
    y = np.cumsum(x - x.mean())

    # Window sizes
    window_sizes = []
    size = min_window
    while size <= n // 4:
        window_sizes.append(size)
        size = int(size * 1.3)

    if len(window_sizes) < 3:
        return 0.5

    fluctuations = []
    for w in window_sizes:
        n_windows = n // w
        f_squared = []
        for i in range(n_windows):
            segment = y[i * w:(i + 1) * w]
            t = np.arange(w)
            # Fit linear trend
            coeffs = np.polyfit(t, segment, 1)
            trend = np.polyval(coeffs, t)
            residual = segment - trend
            f_squared.append(np.mean(residual ** 2))

        if f_squared:
            fluctuations.append(np.sqrt(np.mean(f_squared)))
        else:
            fluctuations.append(np.nan)

    valid = ~np.isnan(fluctuations) & (np.array(fluctuations) > 0)
    if valid.sum() < 3:
        return 0.5

    log_n = np.log(np.array(window_sizes)[valid])
    log_f = np.log(np.array(fluctuations)[valid])

    coeffs = np.polyfit(log_n, log_f, 1)
    H = np.clip(coeffs[0], 0.01, 0.99)
    return H


def estimate_all_classical(X: np.ndarray, method: str = "both") -> dict:
    """
    Run classical estimators on an entire dataset.

    Args:
        X: array of shape (n_samples, series_length)
        method: "rs", "dfa", or "both"

    Returns:
        dict with keys like "rs_predictions", "dfa_predictions"
    """
    results = {}

    if method in ("rs", "both"):
        print("Running R/S analysis...")
        rs_preds = np.array([rs_hurst(x) for x in tqdm(X, desc="R/S")])
        results["rs"] = rs_preds

    if method in ("dfa", "both"):
        print("Running DFA...")
        dfa_preds = np.array([dfa_hurst(x) for x in tqdm(X, desc="DFA")])
        results["dfa"] = dfa_preds

    return results
