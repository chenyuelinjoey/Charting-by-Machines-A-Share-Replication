"""
01_preprocess.py
Data cleaning, feature engineering, and target variable computation.

Input: CSMAR raw CSV files in data/raw/
Output: data/interim/step4_full_data.parquet
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import pandas as pd
import numpy as np
from scipy.stats import rankdata, norm
import glob
import warnings
warnings.filterwarnings('ignore')

from config import Config

print("=" * 60)
print("STEP 1: DATA PREPROCESSING")
print("=" * 60)


def load_raw_data():
    """Load all CSMAR CSV files from raw data directory."""
    all_files = glob.glob(os.path.join(Config.DATA_RAW_PATH, "*.csv"))
    if not all_files:
        raise FileNotFoundError(
            f"No CSV files found in {Config.DATA_RAW_PATH}. "
            "Please place CSMAR TRD_Mnth CSV files in this directory."
        )
    
    df_list = []
    for f in all_files:
        # Try different encodings
        try:
            df = pd.read_csv(f, dtype={'Stkcd': str}, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(f, dtype={'Stkcd': str}, encoding='utf-8')
        df_list.append(df)
        print(f"  Loaded {os.path.basename(f)}: {len(df):,} rows")
    
    df_raw = pd.concat(df_list, ignore_index=True)
    print(f"Total loaded: {len(df_raw):,} rows from {len(all_files)} files")
    return df_raw


def clean_data(df):
    """Clean and filter data."""
    # Convert date
    df['Trdmnt'] = pd.to_datetime(df['Trdmnt'])
    
    # Filter date range
    start_date = f"{Config.TRAIN_START_YEAR}-01-01"
    end_date = f"{Config.TEST_END_YEAR}-12-31"
    df = df[df['Trdmnt'] >= start_date]
    df = df[df['Trdmnt'] <= end_date]
    print(f"  After date filter ({start_date} to {end_date}): {len(df):,} rows")
    
    # Remove missing returns
    df = df[df['Mretwd'].notna()]
    print(f"  After removing missing returns: {len(df):,} rows")
    
    # Remove extreme returns (paper: >300% or <-90%)
    df = df[(df['Mretwd'] > -0.9) & (df['Mretwd'] < 3.0)]
    print(f"  After removing extreme returns: {len(df):,} rows")
    
    # Remove missing or non-positive market cap
    df = df[df['Msmvttl'].notna() & (df['Msmvttl'] > 0)]
    print(f"  After removing invalid market cap: {len(df):,} rows")
    
    return df


def compute_cumulative_returns(df):
    """Compute CR_1 to CR_12 features (12-month cumulative returns)."""
    # Sort by stock and date
    df = df.sort_values(['Stkcd', 'Trdmnt'])
    df['ret_p1'] = 1 + df['Mretwd']
    
    for k in range(1, 13):
        print(f"  Computing CR_{k}...", end='', flush=True)
        cr = (
            df.groupby('Stkcd')['ret_p1']
            .rolling(k, min_periods=k)
            .apply(lambda x: np.prod(x) - 1, raw=True)
            .reset_index(level=0, drop=True)
            .shift(1)  # Shift to avoid look-ahead bias
        )
        df[f'CR_{k}'] = cr
        del cr
        gc.collect()
        print(" done")
    
    df.drop('ret_p1', axis=1, inplace=True)
    df = df.dropna(subset=[f'CR_{k}' for k in range(1, 13)])
    print(f"  After removing rows with missing CR features: {len(df):,} rows")
    return df


def compute_rnorm(df):
    """
    Compute cross-sectional normal score (rNorm).
    rNorm = inverse CDF of normal distribution applied to percentiles.
    """
    def calc_rnorm(group):
        if len(group) < 5:
            return pd.Series(index=group.index, dtype=float)
        ranks = rankdata(group['Mretwd']) / (len(group) + 1)
        return pd.Series(norm.ppf(ranks), index=group.index)
    
    print("  Computing rNorm by month...")
    df['rNorm'] = df.groupby('Trdmnt', group_keys=False).apply(calc_rnorm)
    df = df.dropna(subset=['rNorm'])
    print(f"  After rNorm computation: {len(df):,} rows")
    return df


def main():
    print("\n[1.1] Loading raw data...")
    df = load_raw_data()
    
    print("\n[1.2] Cleaning data...")
    df = clean_data(df)
    
    print("\n[1.3] Computing cumulative returns (CR_1 to CR_12)...")
    df = compute_cumulative_returns(df)
    
    print("\n[1.4] Computing rNorm...")
    df = compute_rnorm(df)
    
    # Add year column for convenience
    df['Year'] = df['Trdmnt'].dt.year
    
    # Save output
    output_path = os.path.join(Config.DATA_INTERIM_PATH, "step4_full_data.parquet")
    df.to_parquet(output_path, index=False)
    print(f"\n✅ Saved: {output_path}")
    print(f"   Total rows: {len(df):,}")
    print(f"   Date range: {df['Trdmnt'].min()} to {df['Trdmnt'].max()}")
    print(f"   Unique stocks: {df['Stkcd'].nunique():,}")
    print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
