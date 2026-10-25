import pytest
import pandas as pd
import numpy as np
from src.preprocessing import prepare_train_test_data

def test_pipeline_leakage_and_shape():
    df = pd.DataFrame({
        'Locality': ['Andheri West / Lokhandwala', 'Powai / Hiranandani', 'Thane West (Ghodbunder Road)', 'Bandra West / Khar', 'Mira Road / Bhayandar'],
        'CarpetArea_sqft': [650.0, 1100.0, np.nan, 850.0, 420.0],
        'Gymnasium': [1, 0, 1, 1, 0],
        'Price_In_Lakhs': [180.0, 260.0, 75.0, 420.0, 52.0]
    })
    result = prepare_train_test_data(df, target_col='Price_In_Lakhs', test_size=0.4, random_state=42)
    assert len(result['X_train']) == 3
    assert len(result['X_test']) == 2
    assert not np.isnan(result['X_train_trans']).any()
    assert not np.isnan(result['X_test_trans']).any()