import numpy as np
from src.model_training import train_and_evaluate_models

def test_model_training_and_metrics():
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_train = np.array([10, 20, 30, 40, 50])
    X_test = np.array([[2, 2], [4, 4]])
    y_test = np.array([20, 40])
    
    fitted_models, metrics_df, predictions, best_model = train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )
    assert len(fitted_models) >= 3
    assert "RMSE" in metrics_df.columns
    assert best_model in fitted_models