import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from datetime import datetime
import pickle
import os

def extract_temporal_features(df):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True).dt.tz_localize(None)
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
    # Handle NaN in StdAmount (e.g., for customers with a single transaction)
    agg_features['StdAmount'] = agg_features['StdAmount'].fillna(0)
    return agg_features

def preprocess_data(df, fit=True, preprocessor=None):
    # Handle duplicate is_high_risk columns
    if 'is_high_risk_x' in df.columns and 'is_high_risk_y' in df.columns:
        df['is_high_risk'] = df['is_high_risk_x']
        df = df.drop(['is_high_risk_x', 'is_high_risk_y'], axis=1)
    elif 'is_high_risk_x' in df.columns:
        df['is_high_risk'] = df['is_high_risk_x']
        df = df.drop('is_high_risk_x', axis=1)
    elif 'is_high_risk_y' in df.columns:
        df['is_high_risk'] = df['is_high_risk_y']
        df = df.drop('is_high_risk_y', axis=1)
    elif 'is_high_risk' not in df.columns:
        df['is_high_risk'] = 0  # Default to non-high-risk if missing
    
    # Extract temporal features
    df = extract_temporal_features(df)
    
    # Create aggregate features
    agg_df = create_aggregate_features(df)
    df = df.merge(agg_df, on='CustomerId', how='left')
    
    # Preserve is_high_risk for output
    is_high_risk = df['is_high_risk'].copy()
    
    # Define numerical and categorical columns
    numerical_cols = ['TotalAmount', 'AvgAmount', 'TransactionCount', 'StdAmount', 'TransactionHour']
    categorical_cols = ['ProductCategory', 'ChannelId']
    
    # Initialize preprocessor if not provided
    if preprocessor is None:
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop'
        )
    
    # Fit or transform data
    if fit:
        processed_data = preprocessor.fit_transform(df)
    else:
        processed_data = preprocessor.transform(df)
    
    # Convert to DataFrame with appropriate column names
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        elif name == 'cat':
            feature_names.extend(transformer.named_steps['onehot'].get_feature_names_out(cols))
    
    processed_data = pd.DataFrame(processed_data, columns=feature_names, index=df.index)
    
    # Create processed_df with only processed features and is_high_risk
    processed_df = processed_data.copy()
    processed_df['is_high_risk'] = is_high_risk
    
    return processed_data, preprocessor, processed_df

if __name__ == "__main__":
    # Load data with proxy variable
    df = pd.read_csv('../data/processed/transactions_with_proxy.csv')
    
    # Preprocess data
    processed_data, preprocessor, processed_df = preprocess_data(df)
    
    # Save processed data
    pd.DataFrame(processed_data).to_csv('../data/processed/processed_data.csv', index=False)
    processed_df.to_csv('../data/processed/processed_data_with_features.csv', index=False)
    
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    # Save preprocessor
    with open('../models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)