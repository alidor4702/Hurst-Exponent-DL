"""
Part 3 & 4: CNN + Ensemble for Hurst Exponent Estimation.

This script:
1. Loads processed data and pre-trained Dense model from Part 2
2. Trains the CNN (Stone 2020 architecture)
3. Compares Dense vs CNN (bias, MAD, scatter)
4. Trains the Ensemble meta-learner on Dense + CNN predictions
5. Trains enhanced ensemble with uncertainty features
6. Generates all comparison plots

Usage:
    python run_part3_4_cnn_ensemble.py
"""

import numpy as np
import torch
import joblib
from pathlib import Path

from src.utils.config import load_config, get_device
from src.data.preprocessing import load_processed_data
from models.architectures.dense import get_dense_model, count_parameters as count_dense
from models.architectures.cnn import HurstCNN, count_parameters as count_cnn
from models.architectures.ensemble import EnsembleMetaLearner, EnsembleWithFeatures, count_parameters as count_ens
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
)


def load_dense_model(X_test, device):
    """Load the best Dense model from Part 2."""
    model = get_dense_model("medium", input_size=100)
    model.load_state_dict(torch.load("models/saved/dense_medium_best.pt", weights_only=True))
    model = model.to(device)
    model.eval()

    trainer = HurstTrainer(model=model, device=device, model_name="dense_medium_loaded")
    preds = trainer.predict(X_test)

    # MC Dropout uncertainty
    mean_pred, std_pred = trainer.predict_with_uncertainty(X_test, n_samples=30)

    print(f"Loaded Dense (medium): {count_dense(model):,} params")
    return preds, mean_pred, std_pred


def train_cnn(X_train, y_train, X_val, y_val, X_test, y_test, device, cfg):
    """Train the CNN model."""
    model = HurstCNN(input_size=X_train.shape[1])
    n_params = count_cnn(model)
    print(f"\n{'=' * 60}")
    print(f"CNN (Stone 2020) — {n_params:,} parameters")
    print(f"{'=' * 60}")

    # Stone (2020) uses batch_size=64 and epochs=30
    # We use more epochs with early stopping for flexibility
    trainer = HurstTrainer(
        model=model,
        device=device,
        lr=cfg["training"]["learning_rate"],
        batch_size=64,  # paper uses 64
        patience=20,
        model_name="cnn_stone",
    )

    history = trainer.train(X_train, y_train, X_val, y_val, epochs=150)
    preds = trainer.predict(X_test)
    mean_pred, std_pred = trainer.predict_with_uncertainty(X_test, n_samples=30)

    return trainer, history, preds, mean_pred, std_pred, n_params


def train_ensemble(
    dense_preds, cnn_preds, y_train, y_val, y_test,
    dense_preds_train, cnn_preds_train,
    dense_preds_val, cnn_preds_val,
    device, cfg,
):
    """Train the basic ensemble meta-learner."""
    # Stack predictions as features
    X_ens_train = np.column_stack([dense_preds_train, cnn_preds_train])
    X_ens_val = np.column_stack([dense_preds_val, cnn_preds_val])
    X_ens_test = np.column_stack([dense_preds, cnn_preds])

    model = EnsembleMetaLearner(n_inputs=2)
    n_params = count_ens(model)
    print(f"\n{'=' * 60}")
    print(f"ENSEMBLE (basic) — {n_params:,} parameters")
    print(f"{'=' * 60}")

    trainer = HurstTrainer(
        model=model,
        device=device,
        lr=cfg["training"]["learning_rate"],
        batch_size=cfg["training"]["batch_size"],
        patience=15,
        model_name="ensemble_basic",
    )

    history = trainer.train(X_ens_train, y_train, X_ens_val, y_val, epochs=cfg["training"]["epochs"])
    preds = trainer.predict(X_ens_test)

    return trainer, history, preds, n_params


def train_enhanced_ensemble(
    dense_preds, cnn_preds, dense_std, cnn_std,
    y_train, y_val, y_test,
    dense_preds_train, cnn_preds_train, dense_std_train, cnn_std_train,
    dense_preds_val, cnn_preds_val, dense_std_val, cnn_std_val,
    device, cfg,
):
    """Train enhanced ensemble with uncertainty features."""
    def make_features(d_pred, c_pred, d_std, c_std):
        return np.column_stack([
            d_pred, c_pred,
            d_std, c_std,
            np.abs(d_pred - c_pred),  # disagreement
        ])

    X_ens_train = make_features(dense_preds_train, cnn_preds_train, dense_std_train, cnn_std_train)
    X_ens_val = make_features(dense_preds_val, cnn_preds_val, dense_std_val, cnn_std_val)
    X_ens_test = make_features(dense_preds, cnn_preds, dense_std, cnn_std)

    model = EnsembleWithFeatures(n_inputs=5)
    n_params = count_ens(model)
    print(f"\n{'=' * 60}")
    print(f"ENSEMBLE (enhanced, +uncertainty) — {n_params:,} parameters")
    print(f"{'=' * 60}")

    trainer = HurstTrainer(
        model=model,
        device=device,
        lr=cfg["training"]["learning_rate"],
        batch_size=cfg["training"]["batch_size"],
        patience=15,
        model_name="ensemble_enhanced",
    )

    history = trainer.train(X_ens_train, y_train, X_ens_val, y_val, epochs=cfg["training"]["epochs"])
    preds = trainer.predict(X_ens_test)

    return trainer, history, preds, n_params


def main():
    cfg = load_config()
    device = get_device(cfg["training"]["device"])

    # Load data
    print("Loading processed data...")
    splits = load_processed_data()
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    all_metrics = {}

    # ── Step 1: Load Dense model and get predictions on all splits ──
    print("\n" + "=" * 60)
    print("LOADING DENSE MODEL FROM PART 2")
    print("=" * 60)

    dense_model = get_dense_model("medium", input_size=100)
    dense_model.load_state_dict(torch.load("models/saved/dense_medium_best.pt", weights_only=True))
    dense_model = dense_model.to(device)

    dense_trainer = HurstTrainer(model=dense_model, device=device, model_name="dense_loaded")
    dense_preds_test = dense_trainer.predict(X_test)
    dense_preds_train = dense_trainer.predict(X_train)
    dense_preds_val = dense_trainer.predict(X_val)
    dense_mean_test, dense_std_test = dense_trainer.predict_with_uncertainty(X_test, n_samples=30)
    dense_mean_train, dense_std_train = dense_trainer.predict_with_uncertainty(X_train, n_samples=30)
    dense_mean_val, dense_std_val = dense_trainer.predict_with_uncertainty(X_val, n_samples=30)

    dense_metrics = compute_metrics_by_h(y_test, dense_preds_test)
    all_metrics["Dense"] = dense_metrics
    print(f"  Dense: MAE={dense_metrics['overall_mae']:.4f}")

    # ── Step 2: Train CNN ──
    cnn_trainer, cnn_history, cnn_preds_test, cnn_mean_test, cnn_std_test, cnn_params = train_cnn(
        X_train, y_train, X_val, y_val, X_test, y_test, device, cfg
    )

    # CNN predictions on train/val for ensemble
    cnn_preds_train = cnn_trainer.predict(X_train)
    cnn_preds_val = cnn_trainer.predict(X_val)
    cnn_mean_train, cnn_std_train = cnn_trainer.predict_with_uncertainty(X_train, n_samples=30)
    cnn_mean_val, cnn_std_val = cnn_trainer.predict_with_uncertainty(X_val, n_samples=30)

    cnn_metrics = compute_metrics_by_h(y_test, cnn_preds_test)
    all_metrics["CNN"] = cnn_metrics
    print(f"  CNN: MAE={cnn_metrics['overall_mae']:.4f}, Params={cnn_params:,}")

    # CNN plots
    plot_dir_cnn = "plots/cnn"
    plot_training_history(cnn_history, "CNN (Stone 2020) — Training History", f"{plot_dir_cnn}/training.png")
    plot_bias(cnn_metrics, f"CNN — Bias vs H_true ({cnn_params:,} params)", f"{plot_dir_cnn}/bias.png")
    plot_mad(cnn_metrics, f"CNN — MAD vs H_true ({cnn_params:,} params)", f"{plot_dir_cnn}/mad.png")
    plot_predictions_scatter(y_test, cnn_preds_test, "CNN — Predicted vs True", f"{plot_dir_cnn}/scatter.png")
    plot_error_analysis(y_test, cnn_preds_test, "CNN — Error Analysis", f"{plot_dir_cnn}/error_analysis.png")

    # Dense vs CNN comparison
    plot_comparison(
        {"Dense": dense_metrics, "CNN": cnn_metrics},
        f"{plot_dir_cnn}/dense_vs_cnn.png",
    )

    # ── Step 3: Train Ensemble (basic) ──
    ens_trainer, ens_history, ens_preds, ens_params = train_ensemble(
        dense_preds_test, cnn_preds_test, y_train, y_val, y_test,
        dense_preds_train, cnn_preds_train,
        dense_preds_val, cnn_preds_val,
        device, cfg,
    )

    ens_metrics = compute_metrics_by_h(y_test, ens_preds)
    all_metrics["Ensemble"] = ens_metrics
    print(f"  Ensemble (basic): MAE={ens_metrics['overall_mae']:.4f}, Params={ens_params:,}")

    # ── Step 4: Train Enhanced Ensemble ──
    enh_trainer, enh_history, enh_preds, enh_params = train_enhanced_ensemble(
        dense_preds_test, cnn_preds_test, dense_std_test, cnn_std_test,
        y_train, y_val, y_test,
        dense_preds_train, cnn_preds_train, dense_std_train, cnn_std_train,
        dense_preds_val, cnn_preds_val, dense_std_val, cnn_std_val,
        device, cfg,
    )

    enh_metrics = compute_metrics_by_h(y_test, enh_preds)
    all_metrics["Ensemble+"] = enh_metrics
    print(f"  Ensemble (enhanced): MAE={enh_metrics['overall_mae']:.4f}, Params={enh_params:,}")

    # ── Step 5: Ensemble plots ──
    plot_dir_ens = "plots/ensemble"

    plot_training_history(ens_history, "Ensemble (basic) — Training", f"{plot_dir_ens}/basic_training.png")
    plot_bias(ens_metrics, f"Ensemble — Bias vs H_true", f"{plot_dir_ens}/basic_bias.png")
    plot_mad(ens_metrics, f"Ensemble — MAD vs H_true", f"{plot_dir_ens}/basic_mad.png")
    plot_predictions_scatter(y_test, ens_preds, "Ensemble — Predicted vs True", f"{plot_dir_ens}/basic_scatter.png")

    plot_training_history(enh_history, "Ensemble+ (enhanced) — Training", f"{plot_dir_ens}/enhanced_training.png")
    plot_bias(enh_metrics, f"Ensemble+ — Bias vs H_true", f"{plot_dir_ens}/enhanced_bias.png")
    plot_mad(enh_metrics, f"Ensemble+ — MAD vs H_true", f"{plot_dir_ens}/enhanced_mad.png")
    plot_predictions_scatter(y_test, enh_preds, "Ensemble+ — Predicted vs True", f"{plot_dir_ens}/enhanced_scatter.png")
    plot_error_analysis(y_test, enh_preds, "Ensemble+ — Error Analysis", f"{plot_dir_ens}/enhanced_error_analysis.png")

    # ── Step 6: Full comparison ──
    # Add classical baselines if available
    classical_path = Path("data/processed/classical_predictions.joblib")
    if classical_path.exists():
        classical = joblib.load(classical_path)
        for method, preds in classical.items():
            all_metrics[method.upper()] = compute_metrics_by_h(y_test, preds)

    plot_comparison(all_metrics, f"{plot_dir_ens}/full_comparison.png")
    plot_comparison_summary_table(all_metrics, f"{plot_dir_ens}/full_comparison_summary.png")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("PARTS 3 & 4 RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} {'MAE':>8} {'RMSE':>8} {'Bias':>8}")
    print("-" * 48)
    for name, m in all_metrics.items():
        print(f"{name:<20} {m['overall_mae']:>8.4f} {m['overall_rmse']:>8.4f} {m['overall_bias']:>8.4f}")

    print(f"\nPlots saved to: plots/cnn/ and plots/ensemble/")
    print("Next: run_part5_hurst_trading.py (Real data & trading)")


if __name__ == "__main__":
    main()
