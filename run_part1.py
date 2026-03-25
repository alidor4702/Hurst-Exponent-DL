"""
Part 1: Synthetic Data Generation, Preprocessing, and Visualization.

This script:
1. Generates 21,000 fBM time series for 100 values of H in [0.01, 0.99]
2. Rescales data per-sample (zero mean, unit variance)
3. Splits into 1/3 train / 1/3 val / 1/3 test
4. Saves everything (joblib + parquet)
5. Generates all exploratory visualizations

Usage:
    python run_part1.py                  # Full run (generate + preprocess + plot)
    python run_part1.py --plot-only      # Only generate plots (data must exist)
    python run_part1.py --generate-only  # Only generate data (no plots)
"""

import sys
import time
from pathlib import Path

from src.utils.config import load_config
from src.data.generate import generate_fbm_dataset, save_raw_data, load_raw_data
from src.data.preprocessing import rescale_per_sample, split_data, save_processed_data
from src.data.visualize import run_all_visualizations


def run_generation(cfg: dict):
    print("=" * 60)
    print("STEP 1: Generating synthetic fBM data")
    print("=" * 60)

    data_cfg = cfg["data"]
    raw_path = Path(cfg["paths"]["raw_data"]) / "fbm_dataset.joblib"

    if raw_path.exists():
        print(f"Raw data already exists at {raw_path}")
        print("Loading existing data...")
        raw = load_raw_data(cfg["paths"]["raw_data"])
        return raw["X"], raw["y"], raw["H_values"]

    start = time.time()
    X, y, H_values = generate_fbm_dataset(
        H_min=data_cfg["H_min"],
        H_max=data_cfg["H_max"],
        n_H_values=data_cfg["n_H_values"],
        samples_per_H=data_cfg["samples_per_H"],
        series_length=data_cfg["series_length"],
        seed=data_cfg["seed"],
    )
    elapsed = time.time() - start
    print(f"Generation took {elapsed:.1f}s")

    save_raw_data(X, y, H_values, cfg["paths"]["raw_data"])
    return X, y, H_values


def run_preprocessing(X, y, cfg: dict):
    print("\n" + "=" * 60)
    print("STEP 2: Rescaling and splitting data")
    print("=" * 60)

    processed_path = Path(cfg["paths"]["processed_data"]) / "splits.joblib"

    if processed_path.exists():
        print(f"Processed data already exists at {processed_path}")
        return

    # Rescale per-sample
    print("Rescaling each sample independently (zero mean, unit variance)...")
    X_scaled, means, stds = rescale_per_sample(X)
    print(f"  Before: mean of means = {X.mean(axis=1).mean():.6f}, mean of stds = {X.std(axis=1).mean():.6f}")
    print(f"  After:  mean of means = {X_scaled.mean(axis=1).mean():.6f}, mean of stds = {X_scaled.std(axis=1).mean():.6f}")

    # Split
    print("\nSplitting into 1/3 train / 1/3 val / 1/3 test...")
    splits = split_data(X_scaled, y, ratios=cfg["data"]["split_ratios"], seed=cfg["data"]["seed"])

    # Save
    save_processed_data(splits, cfg["paths"]["processed_data"])

    print("\nWhy 1/3 / 1/3 / 1/3 instead of 60/20/20?")
    print("  - With synthetic data, we can generate unlimited training samples.")
    print("  - Equal-sized splits give more reliable validation and test estimates.")
    print("  - A larger test set means tighter confidence intervals on performance.")


def main():
    cfg = load_config()

    plot_only = "--plot-only" in sys.argv
    generate_only = "--generate-only" in sys.argv

    if plot_only:
        print("Plot-only mode: skipping generation and preprocessing.\n")
        run_all_visualizations()
        return

    X, y, H_values = run_generation(cfg)
    run_preprocessing(X, y, cfg)

    if generate_only:
        print("\nGenerate-only mode: skipping visualizations.")
        return

    print("\n")
    run_all_visualizations()

    print("\n" + "=" * 60)
    print("PART 1 COMPLETE")
    print("=" * 60)
    print(f"\nData saved in: {cfg['paths']['raw_data']}/ and {cfg['paths']['processed_data']}/")
    print(f"Plots saved in: plots/data_exploration/")
    print(f"\nNext: run_part2.py (Dense network training)")


if __name__ == "__main__":
    main()
