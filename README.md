# Deep Learning for Hurst Exponent Estimation & Trading

Estimating the Hurst exponent of financial time series using deep learning, with applications to trading strategy design.

## Overview

The **Hurst exponent** (H) characterizes the memory of a time series:
- **H < 0.5**: Mean-reverting (anti-persistent)
- **H = 0.5**: Random walk (no memory)
- **H > 0.5**: Trending (persistent)

This project trains multiple neural network architectures on synthetic **fractional Brownian motion (fBM)** data to learn H, then applies the best estimator to real financial data to build a Hurst-based trading strategy.

## Project Structure

```
hurst-exponent-dl/
├── configs/                # Hyperparameters & experiment configs
│   └── default.yaml
├── data/
│   ├── raw/                # Raw generated/downloaded data
│   └── processed/          # Rescaled, split data ready for training
├── models/
│   ├── architectures/      # Model definitions (Dense, CNN, Ensemble, ...)
│   └── saved/              # Trained model weights
├── notebooks/              # Jupyter notebooks for exploration & reporting
├── plots/                  # All saved figures
│   ├── data_exploration/
│   ├── dense/
│   ├── cnn/
│   ├── ensemble/
│   └── trading/
├── src/
│   ├── data/               # Data generation, loading, rescaling
│   ├── training/           # Training loops, evaluation
│   ├── trading/            # Backtesting & strategy logic
│   └── utils/              # Plotting helpers, metrics, misc
├── requirements.txt
└── README.md
```

## Architectures

| Model | Description |
|-------|-------------|
| **Dense** | Fully connected deep network |
| **CNN** | 1D ConvNet following Stone (2020, Quantitative Finance) |
| **Ensemble** | Meta-learner stacking Dense + CNN predictions |

## Key Results

*Results and plots will be added as experiments are completed.*

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## References

- Stone, H. (2020). "Calibrating rough volatility models: a convolutional neural network approach." *Quantitative Finance*.
- Gatheral, J., Jaquier, T., & Rosenbaum, M. (2018). "Volatility is rough." *Quantitative Finance*.
