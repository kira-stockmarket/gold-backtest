import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = "public"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_price(df, date):
    target = pd.to_datetime(date)
    sub = df.loc[df.index <= target, 'Close']
    if len(sub) == 0: return np.nan
    val = sub.iloc[-1]
    return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)

def run():
    print("Fetching Gold Data (2000-Present)...")
    gold = yf.download('GC=F', start='2000-01-01', progress=False)
    if isinstance(gold.columns, pd.MultiIndex):
        gold.columns = gold.columns.droplevel(1)
    gold.index = pd.to_datetime(gold.index).tz_localize(None)
    gold = gold.dropna(subset=['Close'])

    records = []
    print("Slicing every month into 15-day pre/post windows...")
    for year in range(2000, 2027):
        for month in range(1, 13):
            if year == 2026 and month > 9: 
                continue
                
            mid_date = pd.Timestamp(year=year, month=month, day=15)
            start_date = mid_date - timedelta(days=15)
            end_date = mid_date + timedelta(days=15)
            
            if start_date < gold.index[0] or end_date > gold.index[-1]:
                continue
                
            p_start = get_price(gold, start_date)
            p_mid = get_price(gold, mid_date)
            p_end = get_price(gold, end_date)
            
            if any(pd.isna(x) for x in [p_start, p_mid, p_end]):
                continue
                
            pre_ret = ((p_mid - p_start) / p_start) * 100
            post_ret = ((p_end - p_mid) / p_mid) * 100
            
            records.append({
                'Year': year,
                'Month': mid_date.strftime('%B'),
                'Mid_Date': mid_date.strftime('%Y-%m-%d'),
                'Pre_15d_Ret_%': round(pre_ret, 2),
                'Post_15d_Ret_%': round(post_ret, 2)
            })

    df = pd.DataFrame(records)

    print("Clustering the data...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(df[['Pre_15d_Ret_%', 'Post_15d_Ret_%']])

    cluster_means = df.groupby('Cluster_ID')[['Pre_15d_Ret_%', 'Post_15d_Ret_%']].mean()
    
    def name_cluster(row):
        pre = "UP" if row['Pre_15d_Ret_%'] > 0 else "DOWN"
        post = "UP" if row['Post_15d_Ret_%'] > 0 else "DOWN"
        return f"Cluster {row.name} ({pre} -> {post})"
        
    cluster_names = {i: name_cluster(row) for i, row in cluster_means.iterrows()}
    df['Cluster_Name'] = df['Cluster_ID'].map(cluster_names)

    # 1. Excel Export
    excel_path = os.path.join(OUTPUT_DIR, "gold_monthly_clusters.xlsx")
    df.to_excel(excel_path, index=False)

    # 2. Crosstab (Which month matches which cluster)
    month_dist = pd.crosstab(df['Month'], df['Cluster_Name'])
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December']
    month_dist = month_dist.reindex(months_order).fillna(0).astype(int)

    # 3. HTML Dashboard Generation
    html_path = os.path.join(OUTPUT_DIR, "index.html")
    
    # FIX: Generate the table HTML outside of the f-string!
    rename_dict = {
        'Pre_15d_Ret_%': 'Avg Pre-15 Return (%)', 
        'Post_15d_Ret_%': 'Avg Post-15 Return (%)'
    }
    
    cluster_table_html = cluster_means.rename(columns=rename_dict).to_html(classes="table", float_format=lambda x: f"{x:.2f}")
    month_dist_html = month_dist.to_html(classes="table")
    
    html_content = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Gold Monthly Clustering Analysis</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; margin: 40px auto; max-width: 900px; padding: 0 20px; color: #333; background: #f9f9f9; }}
            .card {{ background: #fff; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 25px; }}
            h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px; }}
            th, td {{ padding: 12px; text-align: center; border: 1px solid #ddd; }}
            th {{ background-color: #2c3e50; color: white; }}
            td:first-child {{ text-align: left; font-weight: bold; background: #f1f5f9; }}
            .btn {{ display: inline-block; background: #27ae60; color: white; padding: 12px 20px; text-decoration: none; border-radius: 5px; font-weight: bold; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Gold: 15-Day Pre/Post Monthly Clustering (2000-Present)</h1>
        <a href="gold_monthly_clusters.xlsx" download class="btn">📥 Download Full Raw Excel Data</a>
        
        <div class="card">
            <h2>Cluster Definitions (Average Returns)</h2>
            {cluster_table_html}
        </div>

        <div class="card">
            <h2>Which Month Matches Which Cluster Most?</h2>
            <p>This grid shows how many times each month fell into specific patterns over the last 26 years.</p>
            {month_dist_html}
        </div>
    </body>
    </html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print("✅ Successfully generated website and Excel files in public/")

if __name__ == "__main__":
    run()
