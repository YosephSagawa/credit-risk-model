import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime

def calculate_rfm(df, snapshot_date):
    # Convert TransactionStartTime to timezone-naive datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True).dt.tz_localize(None)
    
    # Aggregate RFM metrics
    rfm = df.groupby('CustomerId')[['TransactionStartTime', 'TransactionId', 'Amount']].agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,  # Recency
        'TransactionId': 'count',  # Frequency
        'Amount': 'sum'  # Monetary
    }).reset_index()
    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
    return rfm

def create_proxy_variable(df, snapshot_date):
    # Drop any existing is_high_risk columns to prevent duplicates
    for col in ['is_high_risk', 'is_high_risk_x', 'is_high_risk_y']:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Calculate RFM
    rfm = calculate_rfm(df, snapshot_date)
    
    # Scale RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify high-risk cluster
    cluster_summary = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()
    
    # Standardize Frequency and Monetary for composite score
    cluster_summary['Frequency_scaled'] = StandardScaler().fit_transform(cluster_summary[['Frequency']])
    cluster_summary['Monetary_scaled'] = StandardScaler().fit_transform(cluster_summary[['Monetary']])
    
    # Compute composite risk score (lower Frequency and Monetary = higher risk)
    cluster_summary['Risk_score'] = -(cluster_summary['Frequency_scaled'] + cluster_summary['Monetary_scaled'])
    
    # Select cluster with highest risk score (lowest Frequency and Monetary)
    high_risk_cluster = cluster_summary.loc[cluster_summary['Risk_score'].idxmax(), 'Cluster']
    
    # Assign high-risk label
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    
    # Merge back to main dataset
    df = df.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
    
    # Fill any NaN values in is_high_risk (e.g., for unmatched CustomerIds)
    df['is_high_risk'] = df['is_high_risk'].fillna(0).astype(int)
    
    return df

if __name__ == "__main__":
    df = pd.read_csv('../data/raw/data.csv')
    snapshot_date = datetime(2025, 7, 1)
    df = create_proxy_variable(df, snapshot_date)
    df.to_csv('../data/processed/transactions_with_proxy.csv', index=False)