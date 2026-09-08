import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.cluster import KMeans

OUTPUT_DIR = "public"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Festival Schedule & Historical Indian Annual Gold Import Volumes (Tons)
DIWALI_CALENDAR = {
    2005: ('2005-11-01', 720), 2006: ('2006-10-21', 715), 2007: ('2007-11-09', 769),
    2008: ('2008-10-28', 712), 2009: ('2009-10-17', 580), 2010: ('2010-11-05', 958),
    2011: ('2011-10-26', 969), 2012: ('2012-11-13', 860), 2013: ('2013-11-03', 825),
    2014: ('2014-10-23', 890), 2015: ('2015-11-11', 915), 2016: ('2016-10-30', 557),
    2017: ('2017-10-19', 778), 2018: ('2018-11-07', 759), 2019: ('2019-10-27', 690),
    2020: ('2020-11-14', 430), 2021: ('2021-11-04', 1050), 2022: ('2022-10-24', 706),
    2023: ('2023-11-12', 781), 2024: ('2024-10-31', 757), 2025: ('2025-10-21', 721)
}

def get_price_at(df, target_date):
    target = pd.to_datetime(target_date)
    sub = df.loc[df.index <= target, 'Close']
    if len(sub) == 0:
        return np.nan
    val = sub.iloc[-1]
    return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)

def run_pipeline():
    print("1. Fetching historical Gold Futures data...")
    gold = yf.download('GC=F', start='2004-01-01', progress=False)
    if isinstance(gold.columns, pd.MultiIndex):
        gold.columns = gold.columns.droplevel(1)
    gold.index = pd.to_datetime(gold.index).tz_localize(None)

    records = []
    
    print("2. Extracting seasonal patterns & physical trade features...")
    for year, (diwali_str, annual_imports) in DIWALI_CALENDAR.items():
        diwali_date = pd.to_datetime(diwali_str)
        t_minus_45 = diwali_date - timedelta(days=45) # Start of wholesale accumulation
        t_minus_15 = diwali_date - timedelta(days=15) # Retail rush start
        t_post_15 = diwali_date + timedelta(days=15)  # Post-festival settlement

        p_45 = get_price_at(gold, t_minus_45)
        p_15 = get_price_at(gold, t_minus_15)
        p_diwali = get_price_at(gold, diwali_date)
        p_post = get_price_at(gold, t_post_15)

        if any(pd.isna(x) for x in [p_45, p_15, p_diwali, p_post]):
            continue

        # Feature Engineering:
        # Pre-accumulation trend (Day -45 to Day -15)
        build_up_return = ((p_15 - p_45) / p_45) * 100
        # Final retail stretch (Day -15 to Diwali Day)
        pre_event_return = ((p_diwali - p_15) / p_15) * 100
        # Target to Predict: Post-Diwali Return (Diwali to Day +15)
        post_event_return = ((p_post - p_diwali) / p_diwali) * 100

        records.append({
            'Year': year,
            'Diwali_Date': diwali_str,
            'Imports_Tons': annual_imports,
            'Buildup_Return_%': round(build_up_return, 2),
            'Pre_Diwali_15d_%': round(pre_event_return, 2),
            'Post_Diwali_15d_%': round(post_event_return, 2)
        })

    df = pd.DataFrame(records)

    # 3. Regime Clustering
    # Group years by their setup: Pre-run momentum + Import demand
    X = df[['Buildup_Return_%', 'Pre_Diwali_15d_%', 'Imports_Tons']].values
    
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Regime_Cluster'] = kmeans.fit_predict(X)

    # Regime Profiles
    cluster_summary = df.groupby('Regime_Cluster').agg(
        Count=('Year', 'count'),
        Avg_Buildup=('Buildup_Return_%', 'mean'),
        Avg_Pre15=('Pre_Diwali_15d_%', 'mean'),
        Avg_Post15=('Post_Diwali_15d_%', 'mean'),
        Post_WinRate=('Post_Diwali_15d_%', lambda x: (x > 0).mean() * 100)
    ).reset_index()

    # Assign Regime Labels based on quantitative behavior
    def name_regime(row):
        if row['Avg_Pre15'] > 1.5 and row['Avg_Post15'] < 0:
            return "Exhaustion Spike (Sell-off post festival)"
        elif row['Avg_Post15'] > 1.0:
            return "Sustained Rally (Follow-through strength)"
        else:
            return "Macro Chop (Muted/Consolidation)"

    cluster_summary['Regime_Name'] = cluster_summary.apply(name_regime, axis=1)
    df = df.merge(cluster_summary[['Regime_Cluster', 'Regime_Name']], on='Regime_Cluster', how='left')

    # Latest record acts as the current active scenario
    latest = df.iloc[-1]
    latest_regime = cluster_summary[cluster_summary['Regime_Cluster'] == latest['Regime_Cluster']].iloc[0]

    # 4. Generate Visualizations
    plot_path = os.path.join(OUTPUT_DIR, "predictive_clusters.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for c in range(n_clusters):
        sub = df[df['Regime_Cluster'] == c]
        ax.scatter(sub['Pre_Diwali_15d_%'], sub['Post_Diwali_15d_%'], 
                   s=sub['Imports_Tons']/4, c=colors[c], alpha=0.7, 
                   label=f"Regime {c}: {cluster_summary.loc[c, 'Regime_Name'][:22]}...")

    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel("Pre-Diwali 15-Day Return (%)", fontweight='bold')
    ax.set_ylabel("Post-Diwali 15-Day Return (%) [Target]", fontweight='bold')
    ax.set_title("Gold Regimes: Physical Import Volume (Bubble Size) vs Pre/Post Returns")
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # 5. Build Unified Dashboard
    generate_html(df, cluster_summary, latest, latest_regime)

def generate_html(df, summary, latest, latest_regime):
    html_path = os.path.join(OUTPUT_DIR, "index.html")
    
    summary_table = summary[['Regime_Cluster', 'Regime_Name', 'Count', 'Avg_Pre15', 'Avg_Post15', 'Post_WinRate']].to_html(
        index=False, classes="table", float_format=lambda x: f"{x:.2f}"
    )

    history_table = df[['Year', 'Diwali_Date', 'Imports_Tons', 'Pre_Diwali_15d_%', 'Post_Diwali_15d_%', 'Regime_Name']].to_html(
        index=False, classes="table"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gold Seasonal Intelligence & Regime Prediction</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 30px auto; max-width: 1000px; padding: 0 20px; line-height: 1.5; color: #1e293b; background: #f8fafc; }}
        .header {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 16px; margin-bottom: 24px; }}
        .header h1 {{ margin: 0; color: #0f172a; }}
        .prediction-card {{ background: #ffffff; border: 2px solid #3b82f6; border-radius: 12px; padding: 20px; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }}
        .badge {{ background: #dbeafe; color: #1d4ed8; font-weight: bold; padding: 4px 10px; border-radius: 9999px; font-size: 0.85rem; text-transform: uppercase; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 16px 0; }}
        .stat-box {{ background: #f1f5f9; padding: 14px; border-radius: 8px; text-align: center; }}
        .stat-val {{ font-size: 1.5rem; font-weight: bold; margin-top: 6px; }}
        .green {{ color: #16a34a; }}
        .red {{ color: #dc2626; }}
        img {{ width: 100%; border-radius: 10px; border: 1px solid #cbd5e1; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; margin-top: 15px; font-size: 0.9rem; border: 1px solid #e2e8f0; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f8fafc; font-weight: 600; color: #475569; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Gold Seasonal Forecasting: Physical Flows & Clustering</h1>
        <p>Combining trade volumes, pre-festival momentum, and machine learning pattern recognition to forecast forward post-Diwali returns.</p>
    </div>

    <div class="prediction-card">
        <span class="badge">Active Regime Classification</span>
        <h2>Latest Matched Setup ({latest['Year']}): {latest_regime['Regime_Name']}</h2>
        <p>Based on the pre-Diwali momentum ({latest['Pre_Diwali_15d_%']:+.2f}%) and physical import volume ({latest['Imports_Tons']} Tons), the market matches <strong>Regime {int(latest['Regime_Cluster'])}</strong>.</p>
        
        <div class="stats-grid">
            <div class="stat-box">
                <div>Historical Cluster Win Rate</div>
                <div class="stat-val green">{latest_regime['Post_WinRate']:.1f}%</div>
            </div>
            <div class="stat-box">
                <div>Expected Post-15d Return</div>
                <div class="stat-val {'green' if latest_regime['Avg_Post15'] > 0 else 'red'}">{latest_regime['Avg_Post15']:+.2f}%</div>
            </div>
            <div class="stat-box">
                <div>Historical Samples</div>
                <div class="stat-val">{int(latest_regime['Count'])} Years</div>
            </div>
        </div>
    </div>

    <h2>Regime Clustering Analysis</h2>
    <p>Scatter plot showing pre-event momentum against forward returns. Bubble diameter corresponds to annual physical import volume.</p>
    <img src="predictive_clusters.png" alt="Cluster Scatter Plot">

    <h2>Regime Characteristics</h2>
    {summary_table}

    <h2>Historical Event Ledger</h2>
    {history_table}
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f" Saved full predictive dashboard to {html_path}")

if __name__ == "__main__":
    run_pipeline()
