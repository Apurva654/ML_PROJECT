import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Baseline (Dummy Mean)": DummyRegressor(strategy='mean'),
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    results = []
    fitted_models = {}
    predictions = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2 Score": round(r2, 4)
        })
        fitted_models[name] = model
        predictions[name] = y_pred
        
    metrics_df = pd.DataFrame(results).sort_values(by="RMSE")
    best_model_name = metrics_df[metrics_df["Model"] != "Baseline (Dummy Mean)"].iloc[0]["Model"]
    return fitted_models, metrics_df, predictions, best_model_name