import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from prophet import Prophet
import logging

from app.services.forecasting.base_model import BaseForecaster

# Suppress Prophet logging
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)


class ProphetForecaster(BaseForecaster):
    """Facebook Prophet forecasting model."""
    
    def __init__(self, **kwargs):
        super().__init__(name="prophet")
        self.hyperparameters = {
            "seasonality_mode": kwargs.get("seasonality_mode", "multiplicative"),
            "changepoint_prior_scale": kwargs.get("changepoint_prior_scale", 0.05),
            "seasonality_prior_scale": kwargs.get("seasonality_prior_scale", 10.0),
            "holidays_prior_scale": kwargs.get("holidays_prior_scale", 10.0),
            "yearly_seasonality": kwargs.get("yearly_seasonality", "auto"),
            "weekly_seasonality": kwargs.get("weekly_seasonality", "auto"),
            "daily_seasonality": kwargs.get("daily_seasonality", False),
        }
        self.date_column: Optional[str] = None
        self.target_column: Optional[str] = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        date_column: str = "ds",
        **kwargs
    ) -> "ProphetForecaster":
        """Fit Prophet model.
        
        Args:
            X: DataFrame containing at least the date column
            y: Target series
            date_column: Name of the date column
        """
        self.date_column = date_column
        
        # Prepare data in Prophet format
        df_prophet = pd.DataFrame({
            "ds": pd.to_datetime(X[date_column] if date_column in X.columns else X.index),
            "y": y.values
        })
        
        # Initialize Prophet with hyperparameters
        self.model = Prophet(
            seasonality_mode=self.hyperparameters["seasonality_mode"],
            changepoint_prior_scale=self.hyperparameters["changepoint_prior_scale"],
            seasonality_prior_scale=self.hyperparameters["seasonality_prior_scale"],
            holidays_prior_scale=self.hyperparameters["holidays_prior_scale"],
            yearly_seasonality=self.hyperparameters["yearly_seasonality"],
            weekly_seasonality=self.hyperparameters["weekly_seasonality"],
            daily_seasonality=self.hyperparameters["daily_seasonality"],
        )
        
        # Add additional regressors if provided
        additional_regressors = kwargs.get("additional_regressors", [])
        for regressor in additional_regressors:
            if regressor in X.columns:
                self.model.add_regressor(regressor)
                df_prophet[regressor] = X[regressor].values
        
        # Fit the model
        self.model.fit(df_prophet)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare future dataframe
        if self.date_column and self.date_column in X.columns:
            future = pd.DataFrame({"ds": pd.to_datetime(X[self.date_column])})
        else:
            future = pd.DataFrame({"ds": pd.to_datetime(X.index)})
        
        # Add regressors if they exist
        for col in X.columns:
            if col != self.date_column and col in self.model.extra_regressors:
                future[col] = X[col].values
        
        forecast = self.model.predict(future)
        return forecast["yhat"].values
    
    def predict_interval(
        self,
        X: pd.DataFrame,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions with confidence intervals."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare future dataframe
        if self.date_column and self.date_column in X.columns:
            future = pd.DataFrame({"ds": pd.to_datetime(X[self.date_column])})
        else:
            future = pd.DataFrame({"ds": pd.to_datetime(X.index)})
        
        # Add regressors if they exist
        for col in X.columns:
            if col != self.date_column and col in self.model.extra_regressors:
                future[col] = X[col].values
        
        # Prophet uses 80% interval by default, adjust if needed
        forecast = self.model.predict(future)
        
        predictions = forecast["yhat"].values
        lower = forecast["yhat_lower"].values
        upper = forecast["yhat_upper"].values
        
        # Scale intervals to requested confidence level
        if confidence != 0.80:
            from scipy import stats
            z_80 = stats.norm.ppf(0.90)  # 80% interval
            z_new = stats.norm.ppf((1 + confidence) / 2)
            scale_factor = z_new / z_80
            
            interval_width = (upper - lower) / 2
            lower = predictions - interval_width * scale_factor
            upper = predictions + interval_width * scale_factor
        
        return predictions, lower, upper
    
    def forecast_future(
        self,
        periods: int,
        confidence: float = 0.95
    ) -> pd.DataFrame:
        """Generate future forecasts."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        
        # Get only future predictions
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).copy()
        result.columns = ["date", "prediction", "lower_bound", "upper_bound"]
        
        return result
    
    def get_components(self) -> Dict[str, pd.DataFrame]:
        """Get decomposed forecast components."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        future = self.model.make_future_dataframe(periods=0)
        forecast = self.model.predict(future)
        
        components = {
            "trend": forecast[["ds", "trend"]],
        }
        
        if "yearly" in forecast.columns:
            components["yearly"] = forecast[["ds", "yearly"]]
        if "weekly" in forecast.columns:
            components["weekly"] = forecast[["ds", "weekly"]]
        
        return components
    
    @staticmethod
    def get_default_hyperparameter_space() -> Dict[str, Any]:
        """Return hyperparameter search space for AutoML."""
        return {
            "seasonality_mode": ["additive", "multiplicative"],
            "changepoint_prior_scale": [0.001, 0.01, 0.05, 0.1, 0.5],
            "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
            "holidays_prior_scale": [0.01, 0.1, 1.0, 10.0],
        }
