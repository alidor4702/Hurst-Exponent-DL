# Deep Learning for Financial Time Series

Memory estimation, return prediction, and trading strategies using neural networks on financial data.

## Overview

This project tackles two fundamental questions in quantitative finance using deep learning:

**Part A — Does the market have memory?**
The Hurst exponent (H) characterizes whether a time series is mean-reverting (H < 0.5), a random walk (H = 0.5), or trending (H > 0.5). We train Dense, CNN, and Ensemble networks on synthetic fractional Brownian motion data to estimate H, benchmark against classical estimators (R/S, DFA), then apply the best model to real assets to build a Hurst-based trading strategy.

**Part B — Can we predict returns?**
Using GRU and LSTM recurrent networks with robust training techniques (revIN normalization, gradient clipping, dropout), we predict multi-asset price returns and evaluate with Kendall correlation. Rolling recalibration keeps the model adapted to changing market regimes.

**Part C — Combining both**
The Hurst estimate informs the prediction model: only trade when the market shows detectable memory (H significantly different from 0.5), and adapt strategy type (momentum vs mean-reversion) based on the regime.

## Project Structure

```
dl-financial-timeseries/
├── configs/                 # Hyperparameters & experiment configs
│   └── default.yaml
├── data/
│   ├── raw/                 # Generated fBM data & downloaded market data
│   └── processed/           # Rescaled, split, sequence-formatted data
├── docs/                    # TP assignment PDFs
├── models/
│   ├── architectures/       # All model definitions
│   └── saved/               # Trained model weights
├── plots/                   # All saved figures, organized by part
│   ├── data_exploration/    # Part 1: synthetic data analysis
│   ├── classical/           # Part 2: classical estimator baselines
│   ├── dense/               # Part 2: dense network results
│   ├── cnn/                 # Part 3: CNN results
│   ├── ensemble/            # Part 4: ensemble results
│   ├── hurst_trading/       # Part 5: Hurst-based trading
│   ├── rnn/                 # Part 6: RNN prediction results
│   └── rnn_trading/         # Part 7: RNN-based trading
├── src/
│   ├── data/                # Data generation, loading, preprocessing
│   ├── classical/           # Classical H estimators (R/S, DFA)
│   ├── training/            # Training loops, evaluation
│   ├── trading/             # Backtesting & strategy logic
│   └── utils/               # Plotting, config, metrics
├── Study.md                 # Comprehensive study guide & reference
├── requirements.txt
└── run_part*.py             # Runner scripts for each part
```

## Parts

| Part | Topic | Status |
|------|-------|--------|
| **1. Synthetic Data** | fBM generation, rescaling, visualization | Done |
| **2. Dense Network** | H estimation with fully connected nets + classical baselines | Next |
| **3. CNN** | H estimation with 1D ConvNet (Stone 2020) | |
| **4. Ensemble** | Stacking Dense + CNN predictions | |
| **5. Hurst Trading** | Real data H estimation & trading strategy | |
| **6. RNN Prediction** | GRU/LSTM return prediction, revIN, robust training | |
| **7. RNN Trading** | Rolling calibration & trading strategy | |
| **8. Combined** | Hurst-informed prediction & unified strategy | |

## Architectures

| Model | Purpose | Part |
|-------|---------|------|
| **Dense** | Fully connected network for H estimation | A |
| **CNN** | 1D ConvNet following Stone (2020, QF) | A |
| **Ensemble** | Meta-learner stacking Dense + CNN | A |
| **GRU** | Gated recurrent unit for return prediction | B |
| **LSTM** | Long short-term memory for return prediction | B |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python run_part1.py              # Generate synthetic data + all visualizations
python run_part2_dense.py        # Train dense network + classical baselines
# More parts coming...
```

## References

- Stone, H. (2020). "Calibrating rough volatility models: a convolutional neural network approach." *Quantitative Finance*.
- Gatheral, J., Jaquier, T. & Rosenbaum, M. (2018). "Volatility is rough." *Quantitative Finance*.
- Kim, T. et al. (2022). "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift." *ICLR*.
