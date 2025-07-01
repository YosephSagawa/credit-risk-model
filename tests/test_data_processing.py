import pytest
import pandas as pd
from src.data_processing import extract_temporal_features, create_aggregate_features

def test_extract_temporal_features():
    df = pd.DataFrame({
        'TransactionStartTime': ['2025-01-01 12:00:00', '2025-02-01 15:00:00']
    })
    result = extract_temporal_features(df)
    assert 'TransactionHour' in result.columns
    assert result['TransactionHour'].iloc[0] == 12
    assert result['TransactionMonth'].iloc[1] == 2

def test_create_aggregate_features():
    df = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'Amount': [100, 200, 300],
        'TransactionId': ['T1', 'T2', 'T3']
    })
    result = create_aggregate_features(df)
    assert result.loc[result['CustomerId'] == 'C1', 'TotalAmount'].iloc[0] == 300
    assert result.loc[result['CustomerId'] == 'C1', 'TransactionCount'].iloc[0] == 2