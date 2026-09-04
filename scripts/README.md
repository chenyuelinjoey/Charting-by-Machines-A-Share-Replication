# Scripts

This folder contains the core Python scripts for the complete replication workflow.

---

## 📋 Overview

| Script | Purpose |
| :--- | :--- |
| `01_preprocess.py` | Data cleaning, feature engineering, target computation |
| `02_train_annual.py` | CNN-LSTM training with expanding window + ensemble |
| `03_merge_predictions.py` | Merge annual predictions into one file |
| `04_analysis.py` | Generate tables, figures, and summary statistics |

---

## 📄 Script Descriptions

### `01_preprocess.py` — Data Preprocessing

**Purpose**: Convert raw CSMAR CSV files into a clean, feature-engineered dataset.

**Input**: `data/raw/*.csv` (CSMAR TRD_Mnth files)

**Output**: `data/interim/step4_full_data.parquet`

**Key Steps**:

1. Load and merge all CSV files
2. Clean data (remove missing, extreme values, invalid market cap)
3. Compute CR₁–CR₁₂ (12-month cumulative returns)
4. Compute rNorm (cross-sectional normal score)

**Dependencies**: pandas, numpy, scipy

**Note**: Requires raw CSMAR data. If you have the pre-processed dataset (`step4_full_data.parquet`), you can skip this step.

---

### `02_train_annual.py` — Model Training

**Purpose**: Train CNN-LSTM models year-by-year with expanding window and ensemble averaging.

**Input**: `data/interim/step4_full_data.parquet`

**Output**: `data/predictions/ensemble_preds_YYYY.npz` (one per year)

**Key Features**:

- **Expanding Window**: For year Y, trains on data from 1997 to Y-1
- **Ensemble**: 20 random initializations per year
- **Auto-resume**: Skips years that already have prediction files
- **GPU Support**: Uses TensorFlow with GPU acceleration if available

**Configuration**: Edit `config.py` to change:

- `N_ENSEMBLE`: Number of ensemble members (default: 20)
- `TEST_START_YEAR` / `TEST_END_YEAR`: Year range
- `BATCH_SIZE`: Training batch size (default: 2048)

**Runtime**: ~6-9 hours for 10 years on GPU T4

---

### `03_merge_predictions.py` — Merge Predictions

**Purpose**: Combine all annual prediction files into a single merged dataset.

**Input**: `data/predictions/ensemble_preds_*.npz`

**Output**:

- `data/predictions/ensemble_predictions_merged.npz` (all years combined)
- `data/tables/merge_summary.csv` (sample count summary)

**Key Features**:

- Computes Spearman correlation for full period
- Prints annual breakdown of correlations
- Saves merged file for analysis

---

### `04_analysis.py` — Results Analysis

**Purpose**: Generate all tables, figures, and summary statistics.

**Input**:

- `data/interim/step4_full_data.parquet`
- `data/predictions/ensemble_predictions_merged.npz`

**Output**:

- `outputs/results/portfolio_stats.txt` — Summary statistics
- `outputs/tables/long_short_returns.csv` — Monthly returns
- `outputs/figures/cumulative_return_2015_2024.png` — Cumulative return chart
- `outputs/figures/annual_correlation_2015_2024.png` — Annual correlation chart

**Key Metrics**:

- Spearman correlation (full period + annual)
- Long-short portfolio returns (monthly mean, t-stat, Sharpe ratio)
- CAPM Alpha (monthly, with t-stat)
- Total cumulative return

---

## 🔗 Dependencies

All scripts require the following packages:

```bash
pip install -r requirements.txt
```

Key dependencies:

- `tensorflow>=2.12.0` — Deep learning framework
- `pandas>=2.0.0` — Data manipulation
- `numpy>=1.24.0` — Numerical operations
- `scipy>=1.10.0` — Statistical functions
- `scikit-learn>=1.3.0` — Preprocessing
- `statsmodels>=0.14.0` — Regression analysis
- `matplotlib>=3.7.0` — Visualization
