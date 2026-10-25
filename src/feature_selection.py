import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def correlation_analysis(df: pd.DataFrame, target_col: str, top_k=15):
    num_df = df.select_dtypes(include=[np.number])
    if target_col not in num_df.columns:
        return pd.Series(dtype=float)
    return num_df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False).head(top_k)

def run_rfe_selection(X_train_trans, y_train, feature_names, n_features_to_select=10):
    estimator = LinearRegression()
    rfe = RFE(estimator=estimator, n_features_to_select=min(n_features_to_select, len(feature_names)))
    rfe.fit(X_train_trans, y_train)
    
    ranking_df = pd.DataFrame({
        'Feature': feature_names,
        'Selected': rfe.support_,
        'Rank': rfe.ranking_
    }).sort_values(by='Rank')
    return rfe, ranking_df

def run_feature_importance(X_train_trans, y_train, feature_names, top_k=15):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_trans, y_train)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    return importance_df.head(top_k)
