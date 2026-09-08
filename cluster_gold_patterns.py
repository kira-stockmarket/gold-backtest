import yfinance as yf
import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt

def fetch_data():
    print("Fetching Gold Futures data (GC=F)...")
    gold = yf.download('GC=F', start='2000-01-01', progress=False)
    
    # Handle yfinance multi-index columns if present
    if isinstance(gold.columns, pd.MultiIndex):
        gold.columns = gold.columns.droplevel(1)
        
    gold = gold.dropna(subset=['Close'])
    return gold

def run_clustering():
    gold = fetch_data()
    
    window_size = 60 # 60 trading days (~3 months) per pattern
    sequences = []
    dates = []

    print(f"Slicing historical data into rolling {window_size}-day windows...")
    prices = gold['Close'].values
    index_dates = gold.index

    for i in range(len(prices) - window_size):
        window = prices[i : i + window_size]
        
        # Normalize window: Convert to percentage change from Day 1 of the window
        # This allows us to compare the *shape* of 2005 gold with 2023 gold
        normalized_window = (window / window[0]) - 1 
        
        sequences.append(normalized_window)
        # Record the end date of the pattern so we can map it to historical events later
        dates.append(index_dates[i + window_size]) 

    # Convert to the 3D array format required by tslearn (Samples, Timesteps, Dimensions)
    X = np.array(sequences).reshape(-1, window_size, 1)

    num_clusters = 4 # We will look for the 4 most common macro patterns
    
    print(f"Clustering {len(X)} historical patterns using DTW (Dynamic Time Warping)...")
    print("Please wait, this may take a couple of minutes...")
    
    km = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", random_state=42)
    labels = km.fit_predict(X)

    # Map the results to a DataFrame
    results = pd.DataFrame({'Pattern_End_Date': dates, 'Cluster_ID': labels})
    
    # Save the dates and their assigned clusters to a CSV
    results.to_csv("gold_pattern_clusters.csv", index=False)
    print("\n✅ Cluster mapping saved to 'gold_pattern_clusters.csv'")
    print("You can open this CSV to cross-reference Cluster IDs with historical events.")

    # Plot the distinct cluster shapes
    plot_clusters(km, num_clusters)

def plot_clusters(km, num_clusters):
    plt.figure(figsize=(12, 8))
    
    for i in range(num_clusters):
        plt.subplot(2, 2, i + 1)
        # km.cluster_centers_ holds the 'average' shape of all patterns in that cluster
        center = km.cluster_centers_[i].ravel() * 100 # Convert back to percentage for display
        
        plt.plot(center, color='#1f77b4', linewidth=2)
        plt.title(f'Cluster {i} Average Shape')
        plt.xlabel('Trading Days (0 to 60)')
        plt.ylabel('Return (%)')
        plt.axhline(0, color='black', linestyle='--', alpha=0.6)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gold_cluster_shapes.png')
    print("✅ Cluster visualization saved to 'gold_cluster_shapes.png'")
    plt.show()

if __name__ == "__main__":
    run_clustering()
