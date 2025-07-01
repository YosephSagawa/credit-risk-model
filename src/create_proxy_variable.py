import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime

def calculate_rfm(df, snapshot_date):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,  # Recency
        'TransactionId': 'count',  # Frequency
        'Amount': 'sum'  # Monetary
    }).reset_index()
    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
    return rfm

def create_proxy_variable(df, snapshot_date):
    # Calculate RFM
    rfm = calculate_rfm(df, snapshot_date)
    
    # Scale RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify high-risk cluster (low Frequency and Monetary)
    cluster_summary = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    })
    high_risk_cluster = cluster_summary[
        (cluster_summary['Frequency'] == cluster_summary['Frequency'].min()) &
        (cluster_summary['Monetary'] == cluster_summary['Monetary'].min())
    ].index[0]
    
    # Assign high-risk label
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    
    # Merge back to main dataset
    df = df.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
    return df

if __name__ == "__main__":
    df = pd.read_csv('../data/raw/transactions.csv')
    snapshot_date = datetime(2025, 7, 1)
    df = create_proxy_variable(df, snapshot_date)
    df.to_csv('../data/processed/transactions_with_proxy.csv', index=False)