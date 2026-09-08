import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# Diwali dates from 2000 to 2025
DIWALI_DATES = {
    2000: '2000-10-26', 2001: '2001-11-14', 2002: '2002-11-04',
    2003: '2003-10-25', 2004: '2004-11-12', 2005: '2005-11-01',
    2006: '2006-10-21', 2007: '2007-11-09', 2008: '2008-10-28',
    2009: '2009-10-17', 2010: '2010-11-05', 2011: '2011-10-26',
    2012: '2012-11-13', 2013: '2013-11-03', 2014: '2014-10-23',
    2015: '2015-11-11', 2016: '2016-10-30', 2017: '2017-10-19',
    2018: '2018-11-07', 2019: '2019-10-27', 2020: '2020-11-14',
    2021: '2021-11-04', 2022: '2022-10-24', 2023: '2023-11-12',
    2024: '2024-10-31', 2025: '2025-10-21'
}

def get_closest_trading_price(df, target_date):
    """
    Returns the closing price of the closest trading day on or before the target date.
    """
    target = pd.to_datetime(target_date)
    # Get all dates prior to or equal to target
    past_dates = df.index[df.index <= target]
    if len(past_dates) == 0:
        return np.nan
    return df.loc[past_dates[-1], 'Close']

def run_backtest():
    print("Fetching historical gold data (GC=F)...")
    # Download Gold Futures data from 2000 to present
    gold = yf.download('GC=F', start='2000-01-01', end='2026-12-31', progress=False)
    
    # Clean multi-index columns if present (common in newer yfinance versions)
    if isinstance(gold.columns, pd.MultiIndex):
        gold.columns = gold.columns.droplevel(1)
        
    gold.index = pd.to_datetime(gold.index).tz_localize(None)

    results = []

    print("Running backtest logic...")
    for year, diwali_str in DIWALI_DATES.items():
        diwali_date = pd.to_datetime(diwali_str)
        
        # Define the periods
        pre_diwali_start = diwali_date - timedelta(days=15)
        post_diwali_end = diwali_date + timedelta(days=15)
        
        # Get prices
        price_pre_start = get_closest_trading_price(gold, pre_diwali_start)
        price_diwali = get_closest_trading_price(gold, diwali_date)
        price_post_end = get_closest_trading_price(gold, post_diwali_end)
        
        # Skip years if data isn't available yet (like future dates)
        if pd.isna(price_pre_start) or pd.isna(price_diwali) or pd.isna(price_post_end):
            continue
            
        # Calculate Percentage Returns
        pre_return = ((price_diwali - price_pre_start) / price_pre_start) * 100
        post_return = ((price_post_end - price_diwali) / price_diwali) * 100
        
        results.append({
            'Year': year,
            'Diwali_Date': diwali_date.date(),
            'Pre_Diwali_Return_%': round(pre_return, 2),
            'Post_Diwali_Return_%': round(post_return, 2)
        })

    df_results = pd.DataFrame(results)
    
    # Basic Metrics
    win_rate_pre = (df_results['Pre_Diwali_Return_%'] > 0).mean() * 100
    win_rate_post = (df_results['Post_Diwali_Return_%'] > 0).mean() * 100
    
    print("\n--- BACKTEST RESULTS ---")
    print(f"Total Years Analyzed: {len(df_results)}")
    print(f"Pre-Diwali (15 days prior) Win Rate:  {win_rate_pre:.1f}% | Avg Return: {df_results['Pre_Diwali_Return_%'].mean():.2f}%")
    print(f"Post-Diwali (15 days after) Win Rate: {win_rate_post:.1f}% | Avg Return: {df_results['Post_Diwali_Return_%'].mean():.2f}%")
    
    # Save to CSV
    df_results.to_csv("diwali_gold_performance.csv", index=False)
    print("\nDetailed results saved to 'diwali_gold_performance.csv'")
    
    plot_results(df_results)

def plot_results(df):
    plt.figure(figsize=(12, 6))
    width = 0.35
    x = np.arange(len(df['Year']))
    
    plt.bar(x - width/2, df['Pre_Diwali_Return_%'], width, label='15 Days Pre-Diwali', color='#FF9900')
    plt.bar(x + width/2, df['Post_Diwali_Return_%'], width, label='15 Days Post-Diwali', color='#4CAF50')
    
    plt.xlabel('Year')
    plt.ylabel('Return (%)')
    plt.title('Gold Price Performance Around Diwali (2000 - Present)')
    plt.xticks(x, df['Year'], rotation=45)
    plt.axhline(0, color='black', linewidth=1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('diwali_gold_plot.png')
    plt.show()

if __name__ == "__main__":
    run_backtest()
