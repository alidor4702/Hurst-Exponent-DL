"""
Part 2: Dense Network for Hurst Exponent Estimation.

This script:
1. Loads processed data from Part 1
2. Runs classical estimators (R/S, DFA) as baselines
3. Trains 3 Dense network configurations (small, medium, large)
4. Generates all evaluation plots (bias, MAD, comparison, error analysis)
5. Runs MC Dropout uncertainty analysis on the best model

Usage:
    python run_part2_dense.py              # Full run
    python run_part2_dense.py --skip-classical  # Skip classical (if already computed)
"""

import sys
import time
import numpy as np
import joblib
from pathlib import Path

from src.utils.config import load_config, get_device
from src.data.preprocessing import load_processed_data
from src.classical.estimators import estimate_all_classical
from models.architectures.dense import get_dense_model, count_parameters
from src.training.trainer import HurstTrainer
from src.training.evaluate import (
    compute_metrics_by_h,
    plot_bias,
    plot_mad,
    plot_predictions_scatter,
    plot_training_history,
    plot_comparison,
    plot_comparison_summary_table,
    plot_error_analysis,
    plot_uncertainty_analysis,
)


def run_classical_baselines(X_test, y_test, save_dir="data/processed"):
    """Run R/S and DFA on the test set."""
    cache_path = Path(save_dir) / "classical_predictions.joblib"

    if cache_path.exists():
        print("Loading cached classical predictions...")
        return joblib.load(cache_path)

    print("=" * 60)
    print("CLASSICAL BASELINES (R/S + DFA)")
    print("=" * 60)

    start = time.time()
    results = estimate_all_classical(X_test)
    elapsed = time.time() - start
    print(f"Classical estimation took {elapsed:.1f}s")

    joblib.dump(results, cache_path)
    print(f"Saved to {cache_path}")
    return results


def train_dense_model(size, X_train, y_train, X_val, y_val, device, cfg):
    """Train a single Dense model configuration."""
    model = get_dense_model(size, input_size=X_train.shape[1])
    n_params = count_parameters(model)
    print(f"\n{'=' * 60}")
    print(f"DENSE ({size.upper()}) — {n_params:,} parameters")
    print(f"{'=' * 60}")

    trainer = HurstTrainer(
        model=model,
        device=device,
        lr=cfg["training"]["learning_rate"],
        batch_size=cfg["training"]["batch_size"],
        patience=cfg["training"]["patience"],
        model_name=f"dense_{size}",
    )

    history = trainer.train(X_train, y_train, X_val, y_val, epochs=cfg["training"]["epochs"])
    return trainer, history, n_params


def main():
    cfg = load_config()
    device = get_device(cfg["training"]["device"])
    skip_classical = "--skip-classical" in sys.argv

    # Load data
    print("Loading processed data...")
    splits = load_processed_data()
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    all_metrics = {}
    plot_dir = "plots/dense"

    # ── Step 1: Classical baselines ──
    if not skip_classical:
        classical_results = run_classical_baselines(X_test, y_test)

        for method_name, preds in classical_results.items():
            label = method_name.upper()
            metrics = compute_metrics_by_h(y_test, preds)
            all_metrics[label] = metrics
            print(f"  {label}: MAE={metrics['overall_mae']:.4f}, RMSE={metrics['overall_rmse']:.4f}")

            plot_bias(metrics, f"{label} — Bias vs H_true", f"plots/classical/{method_name}_bias.png")
            plot_mad(metrics, f"{label} — MAD vs H_true", f"plots/classical/{method_name}_mad.png")
            plot_predictions_scatter(y_test, preds, f"{label} — Predicted vs True", f"plots/classical/{method_name}_scatter.png")

    # ── Step 2: Train Dense models ──
    dense_trainers = {}
    for size in ["small", "medium", "large"]:
        trainer, history, n_params = train_dense_model(
            size, X_train, y_train, X_val, y_val, device, cfg
        )
        dense_trainers[size] = trainer

        # Predict on test set
        preds = trainer.predict(X_test)
        metrics = compute_metrics_by_h(y_test, preds)
        label = f"Dense ({size})"
        all_metrics[label] = metrics
        print(f"  {label}: MAE={metrics['overall_mae']:.4f}, RMSE={metrics['overall_rmse']:.4f}, Params={n_params:,}")

        # Individual plots
        plot_training_history(history, f"Dense ({size}) — Training History", f"{plot_dir}/{size}_training.png")
        plot_bias(metrics, f"Dense ({size}) — Bias vs H_true ({n_params:,} params)", f"{plot_dir}/{size}_bias.png")
        plot_mad(metrics, f"Dense ({size}) — MAD vs H_true ({n_params:,} params)", f"{plot_dir}/{size}_mad.png")
        plot_predictions_scatter(y_test, preds, f"Dense ({size}) — Predicted vs True", f"{plot_dir}/{size}_scatter.png")
        plot_error_analysis(y_test, preds, f"Dense ({size}) — Error Analysis", f"{plot_dir}/{size}_error_analysis.png")

    # ── Step 3: Comparison plots ──
    print(f"\n{'=' * 60}")
    print("GENERATING COMPARISON PLOTS")
    print("=" * 60)

    plot_comparison(all_metrics, f"{plot_dir}/comparison_bias_mad.png")
    plot_comparison_summary_table(all_metrics, f"{plot_dir}/comparison_summary.png")

    # ── Step 4: MC Dropout uncertainty (best model) ──
    print(f"\n{'=' * 60}")
    print("MC DROPOUT UNCERTAINTY (Medium model)")
    print("=" * 60)

    best_trainer = dense_trainers["medium"]
    mean_pred, std_pred = best_trainer.predict_with_uncertainty(X_test, n_samples=50)
    plot_uncertainty_analysis(
        y_test, mean_pred, std_pred,
        "Dense (medium) — MC Dropout Uncertainty",
        f"{plot_dir}/medium_uncertainty.png",
    )

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("PART 2 RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} {'MAE':>8} {'RMSE':>8} {'Bias':>8}")
    print("-" * 48)
    for name, m in all_metrics.items():
        print(f"{name:<20} {m['overall_mae']:>8.4f} {m['overall_rmse']:>8.4f} {m['overall_bias']:>8.4f}")

    print(f"\nAll plots saved to: plots/classical/ and {plot_dir}/")
    print("Next: run_part3_cnn.py (CNN architecture)")


if __name__ == "__main__":
    main()
