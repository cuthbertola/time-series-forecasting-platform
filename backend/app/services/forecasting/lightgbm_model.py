import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

from app.services.forecasting.base_model import BaseForecaster


class LightGBMForecaster(BaseForecaster):
    """LightGBM gradient boosting forecaster."""
    
    def __init__(self, **kwargs):
        super().__init__(name="lightgbm")
        self.hyperparameters = {
            "n_estimators": kwargs.get("n_estimators", 100),
            "max_depth": kwargs.get("max_depth", -1),
            "learning_rate": kwargs.get("learning_rate", 0.1),
            "num_leaves": kwargs.get("num_leaves", 31),
            "subsample": kwargs.get("subsample", 0.8),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
            "min_child_samples": kwargs.get("min_child_samples", 20),
            "reg_alpha": kwargs.get("reg_alpha", 0),
            "reg_lambda": kwargs.get("reg_lambda", 0),
            "random_state": kwargs.get("random_state", 42),
        }
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.residual_std: float = 0.0
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scale_features: bool = False,
        **kwargs
    ) -> "LightGBMForecaster":
        """Fit LightGBM model."""
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Prepare features
        X_train = X.copy()
        
        # Scale features if requested
        if scale_features:
            self.scaler = StandardScaler()
            X_train = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=self.feature_names,
                index=X.index
            )
        
        # Handle NaN values
        X_train = X_train.fillna(0)
        y_train = y.fillna(method='ffill').fillna(method='bfill')
        
        # Initialize and fit model
        self.model = LGBMRegressor(
            n_estimators=self.hyperparameters["n_estimators"],
            max_depth=self.hyperparameters["max_depth"],
            learning_rate=self.hyperparameters["learning_rate"],
            num_leaves=self.hyperparameters["num_leaves"],
            subsample=self.hyperparameters["subsample"],
            colsample_bytree=self.hyperparameters["colsample_bytree"],
            min_child_samples=self.hyperparameters["min_child_samples"],
            reg_alpha=self.hyperparameters["reg_alpha"],
            reg_lambda=self.hyperparameters["reg_lambda"],
            random_state=self.hyperparameters["random_state"],
            n_jobs=-1,
            verbose=-1
        )
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Calculate residual standard deviation for prediction intervals
        y_pred = self.model.predict(X_train)
        self.residual_std = np.std(y_train - y_pred)
        
        # Store feature importance
        importance = self.model.feature_importances_
        self.feature_importance = dict(zip(self.feature_names, importance))
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_pred = X.copy()
        
        # Scale features if scaler exists
        if self.scaler is not None:
            X_pred = pd.DataFrame(
                self.scaler.transform(X_pred),
                columns=self.feature_names,
                index=X.index
            )
        
        # Handle NaN values
        X_pred = X_pred.fillna(0)
        
        return self.model.predict(X_pred)
    
    def predict_interval(
        self,
        X: pd.DataFrame,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions with confidence intervals."""
        from scipy import stats
        
        predictions = self.predict(X)
        
        # Calculate interval using normal distribution
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin = z_score * self.residual_std
        
        lower = predictions - margin
        upper = predictions + margin
        
        return predictions, lower, upper
    
    @staticmethod
    def get_default_hyperparameter_space() -> Dict[str, Any]:
        """Return hyperparameter search space for AutoML."""
        return {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [-1, 3, 5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "num_leaves": [15, 31, 63, 127],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_samples": [5, 10, 20, 50],
            "reg_alpha": [0, 0.01, 0.1, 1],
            "reg_lambda": [0, 0.01, 0.1, 1],
        }
