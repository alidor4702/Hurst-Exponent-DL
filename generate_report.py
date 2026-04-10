"""
Generate the TP5 submission report as a PDF.
Answers all questions with code references, figures, and discussion.
"""

from fpdf import FPDF
from pathlib import Path
import os


class TP5Report(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(130, 130, 130)
            self.cell(0, 8, "TP5: Hurst Exponent Estimation with Deep Learning", align="C")
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def title_page(self):
        self.add_page()
        self.ln(50)
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(37, 99, 235)
        self.cell(0, 15, "TP5: Hurst Exponent", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 20)
        self.set_text_color(60, 60, 60)
        self.cell(0, 12, "Deep Learning for Financial Time Series", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(15)
        self.set_font("Helvetica", "", 13)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "AI for Finance", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, "MSc AI - Second Semester", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(25)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(80, 80, 80)
        self.cell(0, 7, "Repository: github.com/alidor4702/dl-financial-timeseries", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(20)

        # TOC
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(37, 99, 235)
        self.cell(0, 10, "Contents", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 11)
        self.set_text_color(60, 60, 60)
        sections = [
            "1. Synthetic Data Generation",
            "2. Dense Networks",
            "3. CNNs (Stone 2020)",
            "4. Dense + CNN Ensemble",
            "5. Application: Real Data Trading",
        ]
        for s in sections:
            self.cell(0, 7, s, new_x="LMARGIN", new_y="NEXT")

    def section_title(self, num, title):
        self.add_page()
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(37, 99, 235)
        self.cell(0, 12, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)
        # Separator line
        self.set_draw_color(37, 99, 235)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def question(self, q_num, text):
        self.ln(3)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(37, 99, 235)
        self.multi_cell(0, 6, f"Q{q_num}: {text}")
        self.ln(1)

    def answer(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def code_ref(self, text):
        self.set_font("Courier", "", 9)
        self.set_text_color(100, 40, 40)
        self.set_fill_color(245, 245, 245)
        self.multi_cell(0, 5, f"  Code: {text}", fill=True)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.ln(1)

    def add_figure(self, path, caption="", width=170):
        if not os.path.exists(path):
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(200, 0, 0)
            self.cell(0, 6, f"[Figure not found: {path}]", new_x="LMARGIN", new_y="NEXT")
            return

        # Check if we need a new page
        if self.get_y() > 200:
            self.add_page()

        x = (210 - width) / 2
        self.image(path, x=x, w=width)
        if caption:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 5, caption, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def results_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)

        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(37, 99, 235)
        self.set_text_color(255, 255, 255)
        for h, w in zip(headers, col_widths):
            self.cell(w, 7, h, border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        for i, row in enumerate(rows):
            fill = i % 2 == 0
            if fill:
                self.set_fill_color(245, 248, 255)
            for val, w in zip(row, col_widths):
                self.cell(w, 6, str(val), border=1, fill=fill, align="C")
            self.ln()
        self.ln(3)


def build_report():
    pdf = TP5Report()
    pdf.alias_nb_pages()
    plot_dir = "plots"

    # ── Title Page ──
    pdf.title_page()

    # ══════════════════════════════════════════════════════════
    # PART 1: SYNTHETIC DATA
    # ══════════════════════════════════════════════════════════
    pdf.section_title("1", "Synthetic Data")

    pdf.question("1.1", "Generate M = 21000 fBM time series of length T = 100")
    pdf.answer(
        "We generate 21,000 fractional Brownian motion (fBM) increment series using the Davies-Harte "
        "algorithm (exact circulant embedding method). We use 100 equally spaced H values in [0.01, 0.99], "
        "with 210 independent samples per H value, giving 100 x 210 = 21,000 total samples. Each sample is a "
        "vector of T = 100 increments. The fbm Python library is used for generation."
    )
    pdf.code_ref("src/data/generate.py (generate_fbm_dataset function)")

    pdf.add_figure(f"{plot_dir}/data_exploration/01_sample_paths.png",
                   "Fig 1.1: fBM increment paths for different H values", width=160)

    pdf.add_figure(f"{plot_dir}/data_exploration/02_cumulative_paths.png",
                   "Fig 1.2: Cumulative paths showing trending (high H) vs mean-reverting (low H)", width=160)

    pdf.question("1.2", "Split X into 1/3 / 1/3 / 1/3 train/val/test")
    pdf.answer(
        "The dataset is randomly shuffled and split into three equal parts: "
        "Train = 6,993 samples, Val = 6,993 samples, Test = 7,014 samples. "
        "Shuffling ensures each split has a uniform distribution of H values."
    )
    pdf.code_ref("src/data/preprocessing.py (split_data function)")

    pdf.add_figure(f"{plot_dir}/data_exploration/07_split_distributions.png",
                   "Fig 1.3: H distribution is uniform across all splits", width=150)

    pdf.question("1.3", "Why not 60/20/20?")
    pdf.answer(
        "With synthetic data, there is no data scarcity -- we can generate unlimited samples. The typical "
        "60/20/20 split maximizes training data when data is scarce (real-world scenario). With synthetic data:\n\n"
        "  - A larger validation set (1/3 vs 20%) gives more reliable early-stopping decisions\n"
        "  - A larger test set (1/3 vs 20%) gives tighter confidence intervals on final metrics\n"
        "  - If we needed more training data, we would simply generate more samples\n\n"
        "Equal splits are standard practice when data generation is cheap."
    )

    pdf.question("1.4", "Rescale X wisely")
    pdf.answer(
        "We apply per-sample standardization: for each individual sample, subtract its mean and divide by its "
        "standard deviation. This is critical because the variance of fBM increments depends on H -- without "
        "rescaling, a network could trivially estimate H from the sample variance alone (the 'cheat'). "
        "After rescaling, every sample has mean=0 and std=1 regardless of H, forcing the network to learn "
        "from the autocorrelation structure -- the actual memory signature of the Hurst exponent."
    )
    pdf.code_ref("src/data/preprocessing.py (rescale_per_sample function)")

    pdf.add_figure(f"{plot_dir}/data_exploration/03_variance_vs_h.png",
                   "Fig 1.4: Left: variance leaks H. Right: after rescaling, variance is constant.", width=160)

    pdf.add_figure(f"{plot_dir}/data_exploration/06_rescaling_effect.png",
                   "Fig 1.5: Rescaling removes scale but preserves autocorrelation patterns", width=160)

    pdf.question("1.5", "Save long computations with joblib/Parquet")
    pdf.answer(
        "All generated data is persisted to avoid re-computation:\n"
        "  - data/raw/fbm_dataset.joblib: raw 21,000 samples before rescaling\n"
        "  - data/processed/splits.joblib: rescaled and split data, ready for training\n"
        "  - data/processed/{train,val,test}.parquet: same data in columnar format"
    )
    pdf.code_ref("src/data/preprocessing.py (save_splits, load_splits functions)")

    # ══════════════════════════════════════════════════════════
    # PART 2: DENSE NETWORKS
    # ══════════════════════════════════════════════════════════
    pdf.section_title("2", "Dense Networks")

    pdf.question("2.1", "Train a dense deep neural network. How many parameters?")
    pdf.answer(
        "We train three Dense (fully connected) architectures to compare:\n\n"
        "  - DenseSmall: 2 hidden layers (64, 32 neurons) -> 8,577 parameters\n"
        "  - DenseMedium: 4 hidden layers (256, 128, 64, 32) with BatchNorm + Dropout -> 70,017 parameters\n"
        "  - DenseLarge: 5 hidden layers (512, 256, 128, 64, 32) with BatchNorm + Dropout -> 228,225 parameters\n\n"
        "All use LeakyReLU activation, Adam optimizer (lr=0.001), MSE loss, batch size 256, "
        "ReduceLROnPlateau scheduler, and early stopping with patience=10."
    )
    pdf.code_ref("models/architectures/dense.py")

    pdf.results_table(
        ["Method", "MAE", "RMSE", "Bias", "Parameters"],
        [
            ["R/S analysis", "0.1407", "0.1673", "+0.096", "n/a"],
            ["DFA", "0.0979", "0.1179", "+0.068", "n/a"],
            ["Dense (small)", "0.0807", "0.1013", "+0.001", "8,577"],
            ["Dense (medium)", "0.0733", "0.0933", "-0.008", "70,017"],
            ["Dense (large)", "0.0699", "0.0887", "-0.012", "228,225"],
        ],
        col_widths=[40, 25, 25, 25, 30],
    )

    pdf.answer(
        "All Dense networks significantly outperform classical methods (R/S and DFA). The classical methods "
        "have strong positive bias (+0.096 and +0.068), while Dense networks have near-zero bias. "
        "Larger networks help but with diminishing returns: going from 8.6K to 228K parameters (26x) "
        "only reduces MAE from 0.0807 to 0.0699. The medium model offers the best accuracy/complexity trade-off."
    )

    pdf.question("2.2", "Why is the validation set not really needed?")
    pdf.answer(
        "With synthetic data, we can generate unlimited training samples, and the dataset (21,000 samples) "
        "is large relative to our model sizes (8K-228K parameters). Overfitting is therefore unlikely. "
        "The validation set is still useful for early stopping and learning rate scheduling, but it's not "
        "critical -- we could nearly merge it into the training set and rely only on the test set for evaluation. "
        "In contrast, with scarce real data, the validation set is essential to prevent overfitting."
    )

    pdf.question("2.3", "Plot the average bias of predicted values as a function of H_true")

    pdf.add_figure(f"{plot_dir}/dense/comparison_bias_mad.png",
                   "Fig 2.1: Bias and MAD comparison across all methods", width=165)

    pdf.answer(
        "The bias plots show that Dense networks exhibit slight 'regression to the mean': they overestimate "
        "low H and underestimate high H. This is because when uncertain, the safest prediction is the dataset mean "
        "(H ~ 0.5). Classical methods (R/S, DFA) have uniformly positive bias -- they systematically overestimate H."
    )

    pdf.question("2.4", "Plot the average absolute deviation as a function of H_true")

    pdf.add_figure(f"{plot_dir}/dense/comparison_summary.png",
                   "Fig 2.2: Overall comparison bar chart (MAE, RMSE, |Bias|)", width=160)

    pdf.answer(
        "MAD is highest at extreme H values (near 0 and 1) for all methods, because these are inherently "
        "harder to estimate from short series (T=100). Dense networks achieve uniformly lower MAD than "
        "classical methods across the entire H range. The Dense Medium model is used as our best Dense "
        "model for subsequent comparisons."
    )

    pdf.add_figure(f"{plot_dir}/dense/medium_uncertainty.png",
                   "Fig 2.3: MC Dropout uncertainty analysis for Dense Medium", width=155)

    # ══════════════════════════════════════════════════════════
    # PART 3: CNNs
    # ══════════════════════════════════════════════════════════
    pdf.section_title("3", "CNNs (Stone 2020)")

    pdf.question("3.1",
        "Train a CNN replicating Stone, H. (2020) QF architecture. How many parameters?")
    pdf.answer(
        "We replicate the exact architecture from Stone (2020), 'Calibrating rough volatility models: "
        "a convolutional neural network approach', Quantitative Finance, 20:3, 379-392.\n\n"
        "Architecture (from Section 3.3 of the paper):\n"
        "  Conv1d(1->32, kernel=20, zero-padding) + LeakyReLU(0.1) + MaxPool(3)\n"
        "  Dropout(0.25)\n"
        "  Conv1d(32->64, kernel=20, zero-padding) + LeakyReLU(0.1) + MaxPool(3)\n"
        "  Dropout(0.25)\n"
        "  Conv1d(64->128, kernel=20, zero-padding) + LeakyReLU(0.1) + MaxPool(3)\n"
        "  Dropout(0.4)\n"
        "  Flatten -> Dense(128) + LeakyReLU(0.1) + Dropout(0.3) -> Dense(1)\n\n"
        "Total: 287,841 parameters. Trained with batch_size=64, Adam optimizer, MSE loss."
    )
    pdf.code_ref("models/architectures/cnn.py (HurstCNN class)")

    pdf.answer(
        "Key design choices: kernel size 20 is unusually large but critical -- it allows each filter "
        "to see 20 consecutive increments, enough to detect the autocorrelation patterns that distinguish "
        "different H values. LeakyReLU(0.1) prevents dead neurons. MaxPool(3) aggressively downsamples "
        "(100 -> 34 -> 12 -> 5). Batch size 64 (not 256) is essential -- the heavy dropout requires "
        "smaller batches for stable training."
    )

    pdf.question("3.2", "Plot the average bias of predicted values as a function of H_true")

    pdf.add_figure(f"{plot_dir}/cnn/bias.png",
                   "Fig 3.1: CNN bias as a function of H_true", width=155)

    pdf.answer(
        "The CNN shows negative bias, particularly at high H values (H > 0.7). This is because the "
        "three rounds of MaxPool(3) compress the 100-point series to just 5 values, losing some of "
        "the very long-range correlation information needed to distinguish H=0.85 from H=0.95. "
        "However, the overall bias (-0.034) is much smaller than classical methods."
    )

    pdf.question("3.3", "Plot the average absolute deviation as a function of H_true")

    pdf.add_figure(f"{plot_dir}/cnn/mad.png",
                   "Fig 3.2: CNN MAD as a function of H_true", width=155)

    pdf.question("3.4", "Compare the performance of Dense and CNN with plots")

    pdf.results_table(
        ["Method", "MAE", "RMSE", "Bias", "Parameters"],
        [
            ["Dense (medium)", "0.0733", "0.0933", "-0.008", "70,017"],
            ["CNN (Stone 2020)", "0.0612", "0.0776", "-0.034", "287,841"],
        ],
        col_widths=[45, 25, 25, 25, 30],
    )

    pdf.add_figure(f"{plot_dir}/cnn/dense_vs_cnn.png",
                   "Fig 3.3: Direct comparison -- Dense vs CNN bias and MAD", width=165)

    pdf.answer(
        "The CNN (MAE=0.0612) outperforms the Dense network (MAE=0.0733) by 16.5%. This confirms that "
        "the CNN's inductive bias -- looking at local windows of consecutive values via large kernels -- "
        "is well suited for detecting autocorrelation patterns. The Dense network treats input values as "
        "unordered and must learn the locality structure from scratch.\n\n"
        "The CNN has slightly more negative bias (-0.034 vs -0.008) at high H, but its overall error "
        "is lower. The kernel size of 20 is critical: an earlier attempt with kernel=5 gave MAE=0.0978 "
        "(worse than Dense), because 5 values are insufficient to capture long-range correlations."
    )

    # ══════════════════════════════════════════════════════════
    # PART 4: DENSE + CNN ENSEMBLE
    # ══════════════════════════════════════════════════════════
    pdf.section_title("4", "Dense + CNN Ensemble")

    pdf.question("4.1",
        "Define a network that takes Dense and CNN predictions as input. Plot bias and MAD vs H_true.")
    pdf.answer(
        "We train a stacking meta-learner: a small neural network whose input is the two predictions "
        "(H_dense, H_cnn) and whose output is the final H estimate. The meta-learner has only 193 parameters "
        "(2 hidden layers: 16 and 8 neurons with ReLU). It learns when to trust each base model.\n\n"
        "We also try an enhanced ensemble (737 parameters) that additionally receives MC Dropout uncertainty "
        "estimates and the disagreement between models as features."
    )
    pdf.code_ref("models/architectures/ensemble.py (EnsembleMetaLearner, EnsembleWithFeatures)")

    pdf.results_table(
        ["Method", "MAE", "RMSE", "Bias", "Parameters"],
        [
            ["R/S analysis", "0.1407", "0.1673", "+0.096", "n/a"],
            ["DFA", "0.0979", "0.1179", "+0.068", "n/a"],
            ["Dense (medium)", "0.0733", "0.0933", "-0.008", "70,017"],
            ["Ensemble+", "0.0658", "0.0823", "+0.002", "737"],
            ["CNN (Stone)", "0.0612", "0.0776", "-0.034", "287,841"],
            ["Ensemble", "0.0605", "0.0760", "-0.002", "193"],
        ],
        col_widths=[40, 25, 25, 25, 30],
    )

    pdf.add_figure(f"{plot_dir}/ensemble/full_comparison_summary.png",
                   "Fig 4.1: Full comparison across all methods (MAE, RMSE, |Bias|)", width=160)

    pdf.add_figure(f"{plot_dir}/ensemble/full_comparison.png",
                   "Fig 4.2: Bias and MAD comparison across all methods", width=165)

    pdf.answer(
        "The basic Ensemble achieves the best performance (MAE=0.0605) with only 193 parameters -- far "
        "fewer than either base model. Its bias is nearly zero (-0.002), meaning it is not systematically "
        "wrong in any direction. The meta-learner has learned to exploit the complementarity between Dense "
        "(better at extreme H, lower bias) and CNN (better overall accuracy, higher bias at extremes).\n\n"
        "The progression tells a clear story:\n"
        "  R/S (1951): 0.1407 -> DFA (1994): 0.0979 -> Dense: 0.0733 -> CNN: 0.0612 -> Ensemble: 0.0605\n"
        "Each step represents a real improvement. The full pipeline cuts error by 57% vs R/S."
    )

    pdf.question("4.2", "How to improve further the resulting network to be even less biased?")
    pdf.answer(
        "Several approaches can reduce bias further:\n\n"
        "1. More diverse base models: Add LSTM, Transformer, or models trained with different hyperparameters. "
        "The key is diversity -- models that make different errors provide more complementary information.\n\n"
        "2. Residual learning: Instead of predicting H directly, have the meta-learner predict the "
        "correction to the best single model. This focuses capacity on fixing errors rather than "
        "re-learning the prediction.\n\n"
        "3. Post-hoc calibration: Fit a simple polynomial or isotonic regression to map predicted H "
        "to calibrated H using validation data, explicitly removing any remaining systematic bias.\n\n"
        "4. Train base models with different random seeds and ensemble across seeds (snapshot ensemble).\n\n"
        "Note: the enhanced ensemble with uncertainty features (Ensemble+, MAE=0.0658) did NOT improve "
        "over the basic version (0.0605). This is an informative negative result -- the MC Dropout "
        "uncertainty adds noise without useful signal, because the meta-learner already infers confidence "
        "from the agreement/disagreement of the two base predictions."
    )

    # ══════════════════════════════════════════════════════════
    # PART 5: REAL DATA APPLICATION
    # ══════════════════════════════════════════════════════════
    pdf.section_title("5", "Application: Real Data Trading")

    pdf.question("5.1", "Download at least 5 years of daily historical data")
    pdf.answer(
        "We download 10 years of daily data for four assets covering different asset classes:\n\n"
        "  - META (Facebook/Meta): US technology stock\n"
        "  - EURCHF=X: EUR/CHF foreign exchange pair\n"
        "  - BTC-USD: Bitcoin cryptocurrency\n"
        "  - SPY: S&P 500 ETF (broad US equity index)\n\n"
        "Data is downloaded via the yfinance library. All assets have 2,500+ daily observations "
        "(2016-04 to 2026-04)."
    )
    pdf.code_ref("src/trading/real_data.py (download_asset function)")

    pdf.question("5.2", "Compute the log price returns r_t vector")
    pdf.answer(
        "Log returns are computed as r_t = ln(P_t / P_{t-1}) where P_t is the adjusted close price. "
        "Log returns are preferred over simple returns because they are additive across time, symmetric "
        "(+10% and -10% cancel), and more stationary."
    )
    pdf.code_ref("src/trading/real_data.py (compute_log_returns function)")

    pdf.question("5.3", "Build the input matrix X")
    pdf.answer(
        "We construct the rolling-window matrix exactly as specified in the TP:\n\n"
        "  X = [[r_1, r_2, ..., r_100],\n"
        "       [r_2, r_3, ..., r_101],\n"
        "       [r_3, r_4, ..., r_102],\n"
        "       ...]\n\n"
        "Each row is a window of T=100 consecutive log returns. For an asset with N log returns, "
        "this produces N-99 windows. Each window is then per-sample rescaled (zero mean, unit variance) -- "
        "the same preprocessing applied during training on synthetic data."
    )
    pdf.code_ref("src/trading/real_data.py (build_rolling_windows, rescale_per_sample)")

    pdf.question("5.4", "Estimate H and trade if H sufficiently different from 1/2")
    pdf.answer(
        "Each rescaled window is fed through our pre-trained Dense, CNN, and Ensemble models "
        "(no retraining -- models are loaded from models/saved/). The Ensemble prediction is used for trading.\n\n"
        "Trading rule with threshold delta = 0.05:\n"
        "  - H > 0.55 -> long position (+1): series is trending, follow the trend\n"
        "  - H < 0.45 -> short position (-1): series is mean-reverting, bet on reversal\n"
        "  - 0.45 <= H <= 0.55 -> flat (0): random walk, no edge\n\n"
        "Position from day t is applied to the return on day t+1 (no lookahead bias). "
        "Transaction cost of 0.1% (10 bps) is charged on each position change."
    )
    pdf.code_ref("src/trading/real_data.py (predict_hurst, generate_signals, compute_strategy_returns)")
    pdf.code_ref("run_part5_real_data.py (main orchestrator)")

    # Per-asset results
    pdf.add_figure(f"{plot_dir}/real_data/META_rolling_hurst.png",
                   "Fig 5.1: META -- Rolling Hurst estimates (Dense, CNN, Ensemble)", width=165)
    pdf.add_figure(f"{plot_dir}/real_data/META_strategy.png",
                   "Fig 5.2: META -- Strategy vs Buy & Hold cumulative returns", width=165)
    pdf.add_figure(f"{plot_dir}/real_data/META_signals.png",
                   "Fig 5.3: META -- Price with trading signals overlay", width=165)

    pdf.add_figure(f"{plot_dir}/real_data/BTCUSD_rolling_hurst.png",
                   "Fig 5.4: BTC-USD -- Rolling Hurst estimates", width=165)
    pdf.add_figure(f"{plot_dir}/real_data/BTCUSD_strategy.png",
                   "Fig 5.5: BTC-USD -- Strategy vs Buy & Hold", width=165)

    pdf.add_figure(f"{plot_dir}/real_data/SPY_strategy.png",
                   "Fig 5.6: SPY -- Strategy vs Buy & Hold", width=165)

    pdf.add_figure(f"{plot_dir}/real_data/EURCHFX_strategy.png",
                   "Fig 5.7: EURCHF -- Strategy vs Buy & Hold", width=165)

    # Multi-asset comparison
    pdf.add_figure(f"{plot_dir}/real_data/multi_asset_comparison.png",
                   "Fig 5.8: Multi-asset performance comparison", width=170)

    pdf.add_figure(f"{plot_dir}/real_data/all_assets_rolling_h.png",
                   "Fig 5.9: Rolling H estimates across all assets (smoothed)", width=165)

    pdf.question("5.5", "Compute cumulated profit and compare with actual price. Comment.")
    pdf.answer(
        "RESULTS AND DISCUSSION\n\n"
        "The Hurst-based trading strategy underperforms buy-and-hold on all four assets. This is an "
        "important and honest finding that reveals several fundamental challenges:\n\n"
        "1. MISMATCH BETWEEN DAILY AND LONG-TERM BEHAVIOR: The H estimates from 100-day windows of daily "
        "returns mostly fall below 0.5 (mean-reverting regime). This is consistent with the rough volatility "
        "literature (Gatheral et al., 2018), which finds H ~ 0.1 for realized volatility. At the daily "
        "frequency, returns do appear mean-reverting. However, at longer horizons, equities and crypto have "
        "been in strong multi-year uptrends. The strategy goes short (betting on reversal) during periods "
        "where assets are trending upward, causing losses.\n\n"
        "2. THE STRATEGY IS TOO NAIVE: A simple threshold-based rule does not capture the complexity of "
        "real markets. The same H estimate can correspond to very different market conditions. A more "
        "sophisticated approach would combine H with other signals (momentum, volatility regimes, "
        "macro indicators).\n\n"
        "3. TRANSACTION COSTS COMPOUND: With 600-1100 trades over the period, transaction costs of 0.1% "
        "per trade add up to significant drag. A more selective strategy (higher threshold, fewer trades) "
        "would reduce costs but also reduce exposure.\n\n"
        "4. SYNTHETIC-TO-REAL TRANSFER GAP: The models were trained on pure fBM, which has a single fixed "
        "H. Real financial returns are more complex -- they are non-stationary, have fat tails, volatility "
        "clustering, and time-varying H. The fBM assumption is a simplification.\n\n"
        "5. INTERESTING OBSERVATIONS:\n"
        "  - All assets show H estimates concentrated below 0.5, confirming mean-reverting behavior at "
        "the daily scale\n"
        "  - H estimates vary across asset classes: EURCHF shows the lowest H (most mean-reverting, "
        "consistent with central bank interventions)\n"
        "  - H is not constant -- it fluctuates over time, suggesting regime changes in market microstructure\n\n"
        "CONCLUSION: The Hurst exponent provides genuine information about market microstructure, but "
        "translating it into a profitable trading strategy requires significantly more sophistication "
        "than a simple threshold rule. The exercise demonstrates both the promise (H captures real "
        "properties of financial data) and the limitations (profitable trading requires much more than "
        "a single signal) of this approach."
    )

    # ── Save PDF ──
    output_path = "report/TP5_Report.pdf"
    Path("report").mkdir(exist_ok=True)
    pdf.output(output_path)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    build_report()
