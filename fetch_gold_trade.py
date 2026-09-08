import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Prevents headless server crashes on GitHub
import matplotlib.pyplot as plt

# Ensure output directory exists for GitHub Pages
os.makedirs("public", exist_ok=True)

# Read API key securely from GitHub Actions environment variable
API_KEY = os.getenv("COMTRADE_API_KEY")

def fetch_trade_data():
    if not API_KEY:
        print("⚠️ COMTRADE_API_KEY not found in environment secrets. Skipping live fetch.")
        return None

    try:
        import comtradeapicall
        print("Fetching Gold trade data (HS 7108) from UN Comtrade API...")
        
        # Pull India (356) gold imports from the World (0)
        df = comtradeapicall.getFinalData(
            subscription_key=API_KEY,
            typeCode='C',
            freqCode='A',
            clCode='HS',
            period='2018,2019,2020,2021,2022,2023,2024,2025',
            reporterCode='356',     # India
            cmdCode='7108',         # Gold
            flowCode='M',           # Imports
            partnerCode='0',        # World total
            format_output='JSON'
        )

        if df is None or df.empty:
            print("No records returned from Comtrade API.")
            return None

        # Process and clean data
        data = df[['period', 'netWgt', 'primaryValue']].copy()
        data.columns = ['Year', 'Weight_KG', 'Value_USD']
        data['Tonnes'] = data['Weight_KG'] / 1000.0
        data['Value_Billion_USD'] = data['Value_USD'] / 1e9
        data = data.sort_values('Year')

        # Save CSV to public/
        csv_path = "public/india_gold_imports.csv"
        data.to_csv(csv_path, index=False)
        print(f" Saved trade data to {csv_path}")

        # Plot trade chart
        fig, ax1 = plt.subplots(figsize=(10, 5))

        color = '#d4af37'
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Import Value ($ Billion)', color=color)
        ax1.bar(data['Year'].astype(str), data['Value_Billion_USD'], color=color, alpha=0.7, label='Value ($B)')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = '#2c3e50'
        ax2.set_ylabel('Volume (Tonnes)', color=color)
        ax2.plot(data['Year'].astype(str), data['Tonnes'], color=color, marker='o', linewidth=2, label='Volume (Tonnes)')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('India Annual Gold Imports (Value vs Volume)')
        plt.tight_layout()
        plot_path = "public/india_gold_trade.png"
        plt.savefig(plot_path)
        plt.close()
        print(f" Saved trade plot to {plot_path}")

        return data

    except Exception as e:
        print(f"❌ Error fetching Comtrade data: {e}")
        return None

if __name__ == "__main__":
    fetch_trade_data()
