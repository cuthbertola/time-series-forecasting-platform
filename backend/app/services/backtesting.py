import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error

class WalkForwardBacktester:
    def __init__(self, model_class, model_params: Dict[str, Any], initial_train_size: int = 365, test_size: int = 30, step_size: int = 30):
        self.model_class = model_class
        self.model_params = model_params
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.results = []
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else None
        return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape) if mape else None}
    
    def backtest(self, X: np.ndarray, y: np.ndarray, dates: pd.Series) -> Dict[str, Any]:
        n_samples = len(y)
        if n_samples < self.initial_train_size + self.test_size:
            raise ValueError(f"Not enough data. Need {self.initial_train_size + self.test_size}, got {n_samples}")
        
        self.results = []
        fold = 0
        train_end = self.initial_train_size
        all_predictions, all_actuals, all_dates = [], [], []
        
        while train_end + self.test_size <= n_samples:
            fold += 1
            X_train, y_train = X[:train_end], y[:train_end]
            test_end = min(train_end + self.test_size, n_samples)
            X_test, y_test = X[train_end:test_end], y[train_end:test_end]
            test_dates = dates.iloc[train_end:test_end]
            
            model = self.model_class(**self.model_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = self._calculate_metrics(y_test, y_pred)
            self.results.append({
                "fold": fold,
                "train_end": dates.iloc[train_end-1].strftime('%Y-%m-%d'),
                "test_start": test_dates.iloc[0].strftime('%Y-%m-%d'),
                "test_end": test_dates.iloc[-1].strftime('%Y-%m-%d'),
                "metrics": metrics
            })
            
            all_predictions.extend(y_pred.tolist())
            all_actuals.extend(y_test.tolist())
            all_dates.extend(test_dates.tolist())
            train_end += self.step_size
        
        overall_metrics = self._calculate_metrics(np.array(all_actuals), np.array(all_predictions))
        mapes = [r["metrics"]["mape"] for r in self.results if r["metrics"]["mape"]]
        
        return {
            "num_folds": fold,
            "overall_metrics": overall_metrics,
            "metric_statistics": {
                "mape_mean": float(np.mean(mapes)) if mapes else None,
                "mape_std": float(np.std(mapes)) if mapes else None
            },
            "fold_results": self.results
        }

def run_backtest(model_type: str, X: np.ndarray, y: np.ndarray, dates: pd.Series, initial_train_size: int = 365, test_size: int = 30, step_size: int = 30) -> Dict[str, Any]:
    if model_type == "xgboost":
        from xgboost import XGBRegressor
        model_class, model_params = XGBRegressor, {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "random_state": 42}
    elif model_type == "lightgbm":
        from lightgbm import LGBMRegressor
        model_class, model_params = LGBMRegressor, {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "random_state": 42, "verbose": -1}
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    backtester = WalkForwardBacktester(model_class, model_params, initial_train_size, test_size, step_size)
    return backtester.backtest(X, y, dates)
