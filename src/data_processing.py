import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xverse.transformer import WOE
from datetime import datetime

def extract_temporal_features(df):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year
    return df

def create_aggregate_features(df):
    agg_features = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'count', 'std'],
        'TransactionId': 'nunique'
    }).reset_index()
    agg_features.columns = [
        'CustomerId', 'TotalAmount', 'AvgAmount', 'TransactionCount', 'StdAmount', 'UniqueTransactions'
    ]
    return agg_features

def preprocess_data(df, fit=True):
    # Extract temporal features
    df = extract_temporal_features(df)
    
    # Create aggregate features
    agg_df = create_aggregate_features(df)
    df = df.merge(agg_df, on='CustomerId', how='left')
    
    # Define numerical and categorical columns
    numerical_cols = ['TotalAmount', 'AvgAmount', 'TransactionCount', 'StdAmount', 'TransactionHour']
    categorical_cols = ['ProductCategory', 'ChannelId']
    
    # Handle missing values and scaling
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Fit and transform
    if fit:
        processed_data = preprocessor.fit_transform(df)
    else:
        processed_data = preprocessor.transform(df)
    
    # Apply WoE transformation
    woe_transformer = WOE()
    if fit:
        processed_data = woe_transformer.fit_transform(processed_data, df['is_high_risk'])
    else:
        processed_data = woe_transformer.transform(processed_data)
    
    return processed_data, preprocessor, woe_transformer

if __name__ == "__main__":
    df = pd.read_csv('../data/raw/data.csv')
    processed_data, preprocessor, woe_transformer = preprocess_data(df)
    pd.DataFrame(processed_data).to_csv('../data/processed/processed_data.csv', index=False)