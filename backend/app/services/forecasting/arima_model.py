import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import warnings

from app.services.forecasting.base_model import BaseForecaster

warnings.filterwarnings('ignore')


class ARIMAForecaster(BaseForecaster):
    """ARIMA/Auto-ARIMA forecaster for univariate time series."""
    
    def __init__(self, **kwargs):
        super().__init__(name="arima")
        self.hyperparameters = {
            "order": kwargs.get("order", None),  # (p, d, q)
            "seasonal_order": kwargs.get("seasonal_order", None),  # (P, D, Q, s)
            "auto_arima": kwargs.get("auto_arima", True),
            "max_p": kwargs.get("max_p", 5),
            "max_q": kwargs.get("max_q", 5),
            "max_d": kwargs.get("max_d", 2),
            "seasonal": kwargs.get("seasonal", True),
            "m": kwargs.get("m", 1),  # Seasonal period
        }
        self.fitted_values: Optional[np.ndarray] = None
        self.residuals: Optional[np.ndarray] = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ) -> "ARIMAForecaster":
        """Fit ARIMA model.
        
        Note: ARIMA is univariate, so X is primarily used for the date index.
        """
        # Get the time series values
        y_values = y.values.astype(float)
        
        # Handle NaN values
        y_clean = pd.Series(y_values).fillna(method='ffill').fillna(method='bfill').values
        
        if self.hyperparameters["auto_arima"]:
            # Use auto_arima to find best parameters
            self.model = auto_arima(
                y_clean,
                start_p=0,
                start_q=0,
                max_p=self.hyperparameters["max_p"],
                max_q=self.hyperparameters["max_q"],
                max_d=self.hyperparameters["max_d"],
                seasonal=self.hyperparameters["seasonal"],
                m=self.hyperparameters["m"],
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                trace=False,
                n_fits=50
            )
            
            # Store the found order
            self.hyperparameters["order"] = self.model.order
            if self.hyperparameters["seasonal"]:
                self.hyperparameters["seasonal_order"] = self.model.seasonal_order
        else:
            # Use specified order
            order = self.hyperparameters["order"] or (1, 1, 1)
            seasonal_order = self.hyperparameters["seasonal_order"] or (0, 0, 0, 0)
            
            self.model = ARIMA(
                y_clean,
                order=order,
                seasonal_order=seasonal_order
            ).fit()
        
        self.is_fitted = True
        self.fitted_values = self.model.fittedvalues()
        self.residuals = y_clean - self.fitted_values
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point predictions for the length of X."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        n_periods = len(X)
        
        if hasattr(self.model, 'predict'):
            # For pmdarima auto_arima
            predictions = self.model.predict(n_periods=n_periods)
        else:
            # For statsmodels ARIMA
            predictions = self.model.forecast(steps=n_periods)
        
        return np.array(predictions)
    
    def predict_interval(
        self,
        X: pd.DataFrame,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions with confidence intervals."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        n_periods = len(X)
        alpha = 1 - confidence
        
        if hasattr(self.model, 'predict'):
            # For pmdarima auto_arima
            predictions, conf_int = self.model.predict(
                n_periods=n_periods,
                return_conf_int=True,
                alpha=alpha
            )
            lower = conf_int[:, 0]
            upper = conf_int[:, 1]
        else:
            # For statsmodels ARIMA
            forecast_result = self.model.get_forecast(steps=n_periods)
            predictions = forecast_result.predicted_mean.values
            conf_int = forecast_result.conf_int(alpha=alpha)
            lower = conf_int.iloc[:, 0].values
            upper = conf_int.iloc[:, 1].values
        
        return np.array(predictions), np.array(lower), np.array(upper)
    
    def forecast_future(
        self,
        periods: int,
        confidence: float = 0.95
    ) -> pd.DataFrame:
        """Generate future forecasts with dates."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        alpha = 1 - confidence
        
        if hasattr(self.model, 'predict'):
            predictions, conf_int = self.model.predict(
                n_periods=periods,
                return_conf_int=True,
                alpha=alpha
            )
            lower = conf_int[:, 0]
            upper = conf_int[:, 1]
        else:
            forecast_result = self.model.get_forecast(steps=periods)
            predictions = forecast_result.predicted_mean.values
            conf_int = forecast_result.conf_int(alpha=alpha)
            lower = conf_int.iloc[:, 0].values
            upper = conf_int.iloc[:, 1].values
        
        return pd.DataFrame({
            "prediction": predictions,
            "lower_bound": lower,
            "upper_bound": upper
        })
    
    def get_model_summary(self) -> str:
        """Get model summary."""
        if not self.is_fitted:
            return "Model not fitted"
        
        if hasattr(self.model, 'summary'):
            return str(self.model.summary())
        return f"ARIMA{self.hyperparameters['order']}"
    
    @staticmethod
    def get_default_hyperparameter_space() -> Dict[str, Any]:
        """Return hyperparameter search space for AutoML."""
        return {
            "max_p": [3, 5, 7],
            "max_q": [3, 5, 7],
            "max_d": [1, 2],
            "seasonal": [True, False],
            "m": [1, 7, 12, 52],
        }
