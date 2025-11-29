from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.feature_importance: Optional[Dict[str, float]] = None
        self.hyperparameters: Dict[str, Any] = {}
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "BaseForecaster":
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point predictions."""
        pass
    
    @abstractmethod
    def predict_interval(
        self, X: pd.DataFrame, confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions with confidence intervals.
        
        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        pass
    
    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        # Handle any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {"mape": np.nan, "rmse": np.nan, "mae": np.nan, "r2": np.nan}
        
        # Calculate MAPE (handle zero values)
        non_zero_mask = y_true_clean != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((y_true_clean[non_zero_mask] - y_pred_clean[non_zero_mask]) / y_true_clean[non_zero_mask])) * 100
        else:
            mape = np.nan
        
        metrics = {
            "mape": mape,
            "rmse": np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
            "mae": mean_absolute_error(y_true_clean, y_pred_clean),
            "r2": r2_score(y_true_clean, y_pred_clean)
        }
        
        return metrics
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        test_size: int = 30
    ) -> Dict[str, List[float]]:
        """Perform time series cross-validation with walk-forward validation."""
        n_samples = len(X)
        min_train_size = max(30, n_samples // 3)
        
        cv_scores = {"mape": [], "rmse": [], "mae": [], "r2": []}
        
        # Walk-forward validation
        for i in range(n_splits):
            # Calculate split point
            test_end = n_samples - (i * test_size)
            test_start = test_end - test_size
            train_end = test_start
            
            if train_end < min_train_size:
                break
            
            # Split data
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            try:
                # Fit and predict
                self.fit(X_train, y_train)
                y_pred = self.predict(X_test)
                
                # Evaluate
                metrics = self.evaluate(y_test, y_pred)
                for key in cv_scores:
                    if not np.isnan(metrics[key]):
                        cv_scores[key].append(metrics[key])
            except Exception as e:
                print(f"CV fold {i} failed: {e}")
                continue
        
        return cv_scores
    
    def save(self, path: str) -> str:
        """Save the model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            "name": self.name,
            "model": self.model,
            "is_fitted": self.is_fitted,
            "feature_importance": self.feature_importance,
            "hyperparameters": self.hyperparameters
        }
        joblib.dump(model_data, path)
        return path
    
    def load(self, path: str) -> "BaseForecaster":
        """Load the model from disk."""
        model_data = joblib.load(path)
        self.name = model_data["name"]
        self.model = model_data["model"]
        self.is_fitted = model_data["is_fitted"]
        self.feature_importance = model_data["feature_importance"]
        self.hyperparameters = model_data["hyperparameters"]
        return self
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Return feature importance if available."""
        return self.feature_importance
    
    def set_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Set model hyperparameters."""
        self.hyperparameters = params
