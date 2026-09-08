import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless mode: avoids display errors on GitHub runners
import matplotlib.pyplot as plt
from datetime import timedelta

# Ensure the deployment directory exists
OUTPUT_DIR = "public"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    target = pd.to_datetime(target_date)
    past_dates = df.index[df.index <= target]
    if len(past_dates) == 0:
        return np.nan
    val = df.loc[past_dates[-1], 'Close']
    return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)

def run_backtest():
    print("1. Fetching Gold Futures (GC=F)...")
    gold = yf.download('GC=F', start='2000-01-01', progress=False)

    if isinstance(gold.columns, pd.MultiIndex):
        gold.columns = gold.columns.droplevel(1)

    gold.index = pd.to_datetime(gold.index).tz_localize(None)

    results = []
    print("2. Calculating returns for 15 days pre and post Diwali...")
    for year, diwali_str in DIWALI_DATES.items():
        diwali_date = pd.to_datetime(diwali_str)
        pre_start = diwali_date - timedelta(days=15)
        post_end = diwali_date + timedelta(days=15)

        p_pre = get_closest_trading_price(gold, pre_start)
        p_diwali = get_closest_trading_price(gold, diwali_date)
        p_post = get_closest_trading_price(gold, post_end)

        if pd.isna(p_pre) or pd.isna(p_diwali) or pd.isna(p_post):
            continue

        pre_ret = ((p_diwali - p_pre) / p_pre) * 100
        post_ret = ((p_post - p_diwali) / p_diwali) * 100

        results.append({
            'Year': int(year),
            'Diwali_Date': diwali_date.strftime('%Y-%m-%d'),
            'Pre_15d_Return_%': round(pre_ret, 2),
            'Post_15d_Return_%': round(post_ret, 2)
        })

    df = pd.DataFrame(results)

    # Save CSV into public folder
    csv_path = os.path.join(OUTPUT_DIR, "diwali_gold_performance.csv")
    df.to_csv(csv_path, index=False)
    print(f" Saved data to {csv_path}")

    # Metrics
    win_rate_pre = (df['Pre_15d_Return_%'] > 0).mean() * 100
    win_rate_post = (df['Post_15d_Return_%'] > 0).mean() * 100
    avg_pre = df['Pre_15d_Return_%'].mean()
    avg_post = df['Post_15d_Return_%'].mean()

    # Create Chart directly in public/
    plot_path = os.path.join(OUTPUT_DIR, "diwali_gold_plot.png")
    plt.figure(figsize=(12, 5))
    x = np.arange(len(df['Year']))
    width = 0.38

    plt.bar(x - width/2, df['Pre_15d_Return_%'], width, label='15 Days Pre-Diwali', color='#e67e22')
    plt.bar(x + width/2, df['Post_15d_Return_%'], width, label='15 Days Post-Diwali', color='#27ae60')

    plt.xlabel('Year', fontweight='bold')
    plt.ylabel('Return (%)', fontweight='bold')
    plt.title('Gold Price Performance: 15 Days Before vs. After Diwali (2000 - Present)')
    plt.xticks(x, df['Year'], rotation=45)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f" Saved chart to {plot_path}")

    # Generate public/index.html
    html_path = os.path.join(OUTPUT_DIR, "index.html")
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gold Diwali Seasonal Backtest</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 30px auto; max-width: 960px; padding: 0 20px; line-height: 1.5; color: #1a1a1a; }}
        h1 {{ color: #b8860b; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 20px 0; }}
        .card {{ background: #fdfdfd; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; text-align: center; }}
        .card-val {{ font-size: 1.6rem; font-weight: bold; margin-top: 8px; }}
        .green {{ color: #27ae60; }}
        .orange {{ color: #e67e22; }}
        img {{ width: 100%; height: auto; border: 1px solid #e0e0e0; border-radius: 8px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.95rem; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #e5e5e5; }}
        th {{ background: #fafafa; font-weight: 600; }}
    </style>
</head>
<body>
    <h1>Gold Price Performance Around Diwali (2000–Present)</h1>
    <p>Automated backtest analyzing Gold Futures (<code>GC=F</code>) 15 calendar days before and 15 days after Diwali.</p>
    
    <div class="grid">
        <div class="card">
            <div>Pre-Diwali Win Rate</div>
            <div class="card-val orange">{win_rate_pre:.1f}%</div>
        </div>
        <div class="card">
            <div>Pre-Diwali Avg Return</div>
            <div class="card-val orange">{avg_pre:+.2f}%</div>
        </div>
        <div class="card">
            <div>Post-Diwali Win Rate</div>
            <div class="card-val green">{win_rate_post:.1f}%</div>
        </div>
        <div class="card">
            <div>Post-Diwali Avg Return</div>
            <div class="card-val green">{avg_post:+.2f}%</div>
        </div>
    </div>

    <h2>Performance Chart</h2>
    <img src="diwali_gold_plot.png" alt="Diwali Gold Chart">

    <h2>Historical Breakdown</h2>
    {df.to_html(index=False, classes="table")}
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f" Generated website dashboard at {html_path}")

if __name__ == "__main__":
    run_backtest()
