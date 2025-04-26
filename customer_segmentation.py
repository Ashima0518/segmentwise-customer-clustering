import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import os

# Load dataset from UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
try:
    df = pd.read_excel(url, sheet_name='Year 2009-2010')
    print("Dataset loaded successfully from UCI repository")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Using backup dataset...")
    url = "https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/data/retail.csv"
    df = pd.read_csv(url)

# Data preprocessing and feature engineering
def prepare_data(df):
    # Clean data
    df = df.dropna()
    df = df[df['Quantity'] > 0]
    
    # Create RFM features
    df['TotalPrice'] = df['Quantity'] * df['Price']
    current_date = df['InvoiceDate'].max() + pd.DateOffset(days=1)
    
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (current_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return rfm

# Prepare data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import os

# Load dataset from UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
try:
    df = pd.read_excel(url, sheet_name='Year 2009-2010')
    print("Dataset loaded successfully from UCI repository")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Using backup dataset...")
    url = "https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/data/retail.csv"
    df = pd.read_csv(url)

# Data preprocessing and feature engineering
def prepare_data(df):
    # Clean data
    df = df.dropna()
    df = df[df['Quantity'] > 0]
    
    # Create RFM features
    df['TotalPrice'] = df['Quantity'] * df['Price']
    current_date = df['InvoiceDate'].max() + pd.DateOffset(days=1)
    
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (current_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return rfm

# Prepare data
rfm_df = prepare_data(df)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

# Clustering and visualization functions
def perform_clustering(data, n_clusters=5):
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(data)
    
    # Visualization
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    sns.scatterplot(x=data[:,0], y=data[:,1], hue=kmeans_labels, palette='viridis')
    plt.title('K-Means Clustering')
    
    plt.subplot(122)
    sns.scatterplot(x=data[:,1], y=data[:,2], hue=kmeans_labels, palette='viridis')
    plt.title('Frequency vs Monetary')
    
    plt.tight_layout()
    plt.savefig('plots/cluster_visualization.png')
    plt.close()
    
    return kmeans_labels

# Main analysis
def main():
    os.makedirs('plots', exist_ok=True)
    
    # Perform clustering
    clusters = perform_clustering(scaled_data)
    
    # Add clusters to dataframe
    rfm_df['Cluster'] = clusters
    
    # Cluster analysis
    cluster_profile = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'sum'],
        'CustomerID': 'count'
    }).reset_index()
    
    print("\nCluster Profile Analysis:")
    print(cluster_profile)
    
    # Save results
    rfm_df.to_csv('customer_segmentation_results.csv', index=False)
    print("\nResults saved to customer_segmentation_results.csv")

if __name__ == "__main__":
    main()
def perform_clustering(data, n_clusters=5):
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(data)
    
    # Visualization
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    sns.scatterplot(x=data[:,0], y=data[:,1], hue=kmeans_labels, palette='viridis')
    plt.title('K-Means Clustering')
    
    plt.subplot(122)
    sns.scatterplot(x=data[:,1], y=data[:,2], hue=kmeans_labels, palette='viridis')
    plt.title('Frequency vs Monetary')
    
    plt.tight_layout()
    plt.savefig('plots/cluster_visualization.png')
    plt.close()
    
    return kmeans_labels

# Main analysis
def main():
    os.makedirs('plots', exist_ok=True)
    
    # Perform clustering
    clusters = perform_clustering(scaled_data)
    
    # Add clusters to dataframe
    rfm_df['Cluster'] = clusters
    
    # Cluster analysis
    cluster_profile = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'sum'],
        'CustomerID': 'count'
    }).reset_index()
    
    print("\nCluster Profile Analysis:")
    print(cluster_profile)
    
    # Save results
    rfm_df.to_csv('customer_segmentation_results.csv', index=False)
    print("\nResults saved to customer_segmentation_results.csv")

if __name__ == "__main__":
    main()
