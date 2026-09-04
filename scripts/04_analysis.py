"""
04_analysis.py
Generate all tables and figures from merged predictions.

Inputs: data/interim/step4_full_data.parquet,
        data/predictions/ensemble_predictions_merged.npz
Outputs: figures, tables, results in outputs/
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from config import Config

print("=" * 60)
print("STEP 4: ANALYSIS")
print("=" * 60)


def calc_long_short(group):
    """Calculate value-weighted long-short portfolio return."""
    if len(group) < 10:
        return np.nan
    group = group.sort_values('pred')
    deciles = np.array_split(group, 10)
    low = np.average(deciles[0]['Mretwd'], weights=deciles[0]['Msmvttl'])
    high = np.average(deciles[-1]['Mretwd'], weights=deciles[-1]['Msmvttl'])
    return high - low


def main():
    # ---- Load Data ----
    print("\n[4.1] Loading data...")
    
    # Load full dataset
    df_full = pd.read_parquet(
        os.path.join(Config.DATA_INTERIM_PATH, "step4_full_data.parquet")
    )
    df_full['Year'] = df_full['Trdmnt'].dt.year
    
    # Load merged predictions
    merged_path = os.path.join(
        Config.DATA_PREDICTIONS_PATH,
        "ensemble_predictions_merged.npz"
    )
    if not os.path.exists(merged_path):
        raise FileNotFoundError(
            "Merged predictions not found. Run 03_merge_predictions.py first."
        )
    
    data = np.load(merged_path, allow_pickle=True)
    y_pred = data['y_pred']
    y_true = data['y_true']
    months = data['months']
    stocks = data['stocks']
    
    # Merge predictions with full data
    df_pred = pd.DataFrame({
        'Stkcd': stocks,
        'Trdmnt': months,
        'pred': y_pred
    })
    df = df_full.merge(df_pred, on=['Stkcd', 'Trdmnt'], how='inner')
    df['Month'] = df['Trdmnt'].dt.to_period('M')
    
    print(f"   Total samples: {len(df):,}")
    print(f"   Date range: {df['Trdmnt'].min()} to {df['Trdmnt'].max()}")
    
    # ---- Spearman Correlation ----
    print("\n[4.2] Computing Spearman correlations...")
    corr_full, pval_full = spearmanr(y_pred, y_true)
    print(f"   Full period: ρ = {corr_full:.4f} (p = {pval_full:.4f})")
    
    # ---- Long-Short Portfolio ----
    print("\n[4.3] Computing long-short portfolio returns...")
    monthly_ls = df.groupby('Trdmnt').apply(calc_long_short).dropna()
    
    mean_ret = monthly_ls.mean()
    std_ret = monthly_ls.std()
    t_stat = mean_ret / (std_ret / np.sqrt(len(monthly_ls)))
    sharpe = mean_ret / std_ret * np.sqrt(12)
    cumulative = (1 + monthly_ls).cumprod()
    total_return = cumulative.iloc[-1] - 1
    
    print(f"   Monthly excess return: {mean_ret:.4%}")
    print(f"   Standard deviation: {std_ret:.4%}")
    print(f"   t-stat: {t_stat:.4f}")
    print(f"   Annualized Sharpe: {sharpe:.4f}")
    print(f"   Total cumulative return: {total_return:.2%}")
    
    # ---- CAPM Alpha ----
    print("\n[4.4] Computing CAPM Alpha...")
    market_ret = df.groupby('Month').apply(
        lambda g: np.average(g['Mretwd'], weights=g['Msmvttl'])
    )
    monthly_ls_period = monthly_ls.copy()
    monthly_ls_period.index = monthly_ls_period.index.to_period('M')
    market_ret.index = market_ret.index.to_period('M')
    
    common_idx = market_ret.index.intersection(monthly_ls_period.index)
    if len(common_idx) > 2:
        m = market_ret.loc[common_idx]
        p = monthly_ls_period.loc[common_idx]
        slope, intercept, r_val, _, std_err = linregress(m, p)
        alpha_tstat = intercept / std_err
        print(f"   CAPM Alpha (monthly): {intercept:.4%}")
        print(f"   Alpha t-stat: {alpha_tstat:.4f}")
        print(f"   Market Beta: {slope:.4f}")
        print(f"   R²: {r_val**2:.4f}")
    else:
        print("   Insufficient data for CAPM regression.")
    
    # ---- Save Results ----
    print("\n[4.5] Saving results...")
    
    # Portfolio stats
    stats_path = os.path.join(Config.RESULTS_PATH, "portfolio_stats.txt")
    with open(stats_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("PORTFOLIO PERFORMANCE SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Spearman ρ: {corr_full:.4f} (p={pval_full:.4f})\n")
        f.write(f"Monthly return: {mean_ret:.4%}\n")
        f.write(f"t-stat: {t_stat:.4f}\n")
        f.write(f"Annualized Sharpe: {sharpe:.4f}\n")
        f.write(f"Total cumulative: {total_return:.2%}\n")
        if len(common_idx) > 2:
            f.write(f"CAPM Alpha (monthly): {intercept:.4%} (t={alpha_tstat:.4f})\n")
    print(f"   ✅ Saved: {stats_path}")
    
    # Monthly returns CSV
    rets_path = os.path.join(Config.TABLES_PATH, "long_short_returns.csv")
    monthly_ls.to_csv(rets_path, header=True)
    print(f"   ✅ Saved: {rets_path}")
    
    # ---- Figures ----
    print("\n[4.6] Generating figures...")
    
    # Figure: Cumulative Return
    plt.figure(figsize=(12, 5))
    plt.plot(cumulative.index, cumulative, linewidth=2, color='#1f77b4')
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    plt.title('MLER Long-Short Cumulative Return (2015-2024)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.grid(alpha=0.3)
    final = cumulative.iloc[-1]
    plt.text(cumulative.index[-1], final,
             f'+{((final-1)*100):.1f}%',
             ha='right', va='bottom', fontsize=12, fontweight='bold', color='darkgreen')
    fig_path = os.path.join(Config.FIGURES_PATH, "cumulative_return_2015_2024.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"   ✅ Saved: {fig_path}")
    
    # Figure: Annual Correlations
    annual_corr = {}
    for y in range(Config.TEST_START_YEAR, Config.TEST_END_YEAR + 1):
        sub = df[df['Year'] == y]
        corr_list = []
        for m in sub['Trdmnt'].unique():
            tmp = sub[sub['Trdmnt'] == m]
            if len(tmp) > 5:
                corr_list.append(spearmanr(tmp['pred'], tmp['Mretwd'])[0])
        annual_corr[y] = np.nanmean(corr_list) if corr_list else np.nan
    
    plt.figure(figsize=(10, 6))
    years = list(annual_corr.keys())
    corrs = list(annual_corr.values())
    bars = plt.bar([str(y) for y in years], corrs, color='steelblue', alpha=0.8)
    plt.axhline(y=corr_full, color='red', linestyle='--',
                label=f'Full period: {corr_full:.4f}')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Spearman Correlation', fontsize=12)
    plt.title('Annual Prediction Performance', fontsize=14)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, corrs):
        if not np.isnan(val):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    fig_path = os.path.join(Config.FIGURES_PATH, "annual_correlation_2015_2024.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"   ✅ Saved: {fig_path}")
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"   Results saved in: {Config.OUTPUT_PATH}")
    print("   Figures saved in: {Config.FIGURES_PATH}")


if __name__ == "__main__":
    main()
