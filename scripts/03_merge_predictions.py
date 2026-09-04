"""
03_merge_predictions.py
Merge all annual predictions into a single dataset and compute basic stats.

Input: data/predictions/ensemble_preds_YYYY.npz
Output: data/predictions/ensemble_predictions_merged.npz
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

from config import Config

print("=" * 60)
print("STEP 3: MERGE PREDICTIONS")
print("=" * 60)


def main():
    # Find all prediction files
    print("\n[3.1] Finding prediction files...")
    pattern = os.path.join(Config.DATA_PREDICTIONS_PATH, "ensemble_preds_*.npz")
    all_files = sorted(glob.glob(pattern))
    
    if len(all_files) == 0:
        raise FileNotFoundError(
            f"No prediction files found in {Config.DATA_PREDICTIONS_PATH}. "
            "Please run 02_train_annual.py first."
        )
    
    print(f"   Found {len(all_files)} files:")
    for f in all_files:
        print(f"     - {os.path.basename(f)}")
    
    # Load and concatenate
    print("\n[3.2] Loading and concatenating...")
    all_preds, all_trues, all_months, all_stocks = [], [], [], []
    year_list = []
    sample_counts = {}
    
    for f in all_files:
        data = np.load(f, allow_pickle=True)
        year = int(os.path.basename(f).split('_')[-1].split('.')[0])
        
        if Config.TEST_START_YEAR <= year <= Config.TEST_END_YEAR:
            all_preds.append(data['y_pred'])
            all_trues.append(data['y_true'])
            all_months.append(data['months'])
            all_stocks.append(data['stocks'])
            year_list.append(year)
            sample_counts[year] = len(data['y_pred'])
            print(f"   Loaded {year}: {len(data['y_pred']):,} samples")
    
    if len(all_preds) == 0:
        raise ValueError("No files found within the specified year range.")
    
    # Concatenate
    y_pred_all = np.concatenate(all_preds)
    y_true_all = np.concatenate(all_trues)
    months_all = np.concatenate(all_months)
    stocks_all = np.concatenate(all_stocks)
    
    print(f"\n   Total samples: {len(y_pred_all):,}")
    print(f"   Years included: {min(year_list)} - {max(year_list)}")
    
    # Spearman correlation
    corr, pval = spearmanr(y_pred_all, y_true_all)
    print(f"\n   Spearman ρ (full period): {corr:.4f} (p-value: {pval:.4f})")
    
    # Annual breakdown
    print("\n   Annual breakdown (Spearman ρ):")
    months_series = pd.Series(months_all)
    years_from_months = pd.to_datetime(months_series).dt.year
    
    for y in range(Config.TEST_START_YEAR, Config.TEST_END_YEAR + 1):
        mask = years_from_months == y
        if mask.sum() > 10:
            corr_y = spearmanr(y_pred_all[mask], y_true_all[mask])[0]
            print(f"     {y}: {mask.sum():,} samples, ρ = {corr_y:.4f}")
        else:
            print(f"     {y}: {mask.sum():,} samples (insufficient)")
    
    # Save merged file
    print("\n[3.3] Saving merged file...")
    merged_path = os.path.join(
        Config.DATA_PREDICTIONS_PATH,
        "ensemble_predictions_merged.npz"
    )
    np.savez_compressed(
        merged_path,
        y_pred=y_pred_all,
        y_true=y_true_all,
        months=months_all,
        stocks=stocks_all
    )
    print(f"   ✅ Saved: {merged_path}")
    print(f"   File size: {os.path.getsize(merged_path) / 1024 / 1024:.1f} MB")
    
    # Save summary as CSV
    summary_path = os.path.join(Config.TABLES_PATH, "merge_summary.csv")
    os.makedirs(Config.TABLES_PATH, exist_ok=True)
    summary_df = pd.DataFrame({
        'year': list(sample_counts.keys()),
        'samples': list(sample_counts.values())
    })
    summary_df.sort_values('year', inplace=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"   ✅ Saved summary: {summary_path}")
    
    print("\n" + "=" * 60)
    print("✅ MERGE COMPLETE!")
    print("=" * 60)
    print("   Next: Run 04_analysis.py to generate tables and figures.")


if __name__ == "__main__":
    main()
