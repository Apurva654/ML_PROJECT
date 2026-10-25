import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def inspect_data_quality(df: pd.DataFrame):
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_counts,
        'Percentage (%)': missing_pct
    }).query('`Missing Values` > 0').sort_values(by='Missing Values', ascending=False)
    
    return {
        "num_rows": df.shape[0],
        "num_cols": df.shape[1],
        "duplicates": int(df.duplicated().sum()),
        "missing_summary": missing_df
    }

def get_preprocessor_pipeline(num_cols, cat_cols):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    return preprocessor

def prepare_train_test_data(df: pd.DataFrame, target_col: str, test_size=0.2, random_state=42):
    cleaned_df = df.drop_duplicates().copy()
    cleaned_df = cleaned_df.dropna(subset=[target_col])
    cleaned_df = cleaned_df[cleaned_df[target_col] > 0]
    
    X = cleaned_df.drop(columns=[target_col])
    y = cleaned_df[target_col].astype(float)
    
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    preprocessor = get_preprocessor_pipeline(num_cols, cat_cols)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_encoded_cols = cat_encoder.get_feature_names_out(cat_cols).tolist() if cat_cols else []
    all_feature_names = num_cols + cat_encoded_cols
    
    return {
        "preprocessor": preprocessor,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feature_names": all_feature_names,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_trans": X_train_trans,
        "X_test_trans": X_test_trans,
        "y_train": y_train,
        "y_test": y_test
    }