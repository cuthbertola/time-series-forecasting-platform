import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time
import optuna
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from app.services.forecasting import (
    ProphetForecaster,
    XGBoostForecaster,
    LightGBMForecaster,
    ARIMAForecaster,
    BaseForecaster
)
from app.services.feature_engineering import TimeSeriesFeatureEngineer

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class AutoMLService:
    """AutoML service for automatic model selection and hyperparameter tuning."""
    
    AVAILABLE_ALGORITHMS = {
        "prophet": ProphetForecaster,
        "xgboost": XGBoostForecaster,
        "lightgbm": LightGBMForecaster,
        "arima": ARIMAForecaster,
    }
    
    def __init__(
        self,
        max_trials: int = 50,
        timeout_seconds: int = 300,
        n_jobs: int = -1,
        metric: str = "mape"
    ):
        self.max_trials = max_trials
        self.timeout_seconds = timeout_seconds
        self.n_jobs = n_jobs
        self.metric = metric
        self.results: List[Dict[str, Any]] = []
        self.best_model: Optional[BaseForecaster] = None
        self.best_score: float = float('inf')
        self.best_algorithm: str = ""
    
    def run(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        algorithms: Optional[List[str]] = None,
        forecast_horizon: int = 30,
        cv_splits: int = 3
    ) -> Dict[str, Any]:
        """Run AutoML to find the best model."""
        start_time = time.time()
        
        # Default to all algorithms if none specified
        if algorithms is None:
            algorithms = list(self.AVAILABLE_ALGORITHMS.keys())
        
        # Validate algorithms
        algorithms = [a for a in algorithms if a in self.AVAILABLE_ALGORITHMS]
        
        # Prepare data
        df_processed = df.copy()
        df_processed[date_column] = pd.to_datetime(df_processed[date_column])
        df_processed = df_processed.sort_values(date_column).reset_index(drop=True)
        
        # Feature engineering for ML models
        feature_engineer = TimeSeriesFeatureEngineer(
            df_processed, date_column, target_column
        )
        df_features = feature_engineer.create_all_features()
        
        # Prepare features and target
        y = df_features[target_column]
        
        # For ML models (XGBoost, LightGBM), use engineered features
        ml_feature_cols = feature_engineer.get_feature_names()
        if feature_columns:
            ml_feature_cols.extend([c for c in feature_columns if c not in ml_feature_cols])
        
        X_ml = df_features[ml_feature_cols].copy()
        
        # For statistical models (Prophet, ARIMA), use original data
        X_stat = df_features[[date_column]].copy()
        
        # Split data for validation
        train_size = int(len(df_features) * 0.8)
        
        X_ml_train, X_ml_val = X_ml.iloc[:train_size], X_ml.iloc[train_size:]
        X_stat_train, X_stat_val = X_stat.iloc[:train_size], X_stat.iloc[train_size:]
        y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]
        
        # Run AutoML for each algorithm
        all_results = []
        
        for algorithm in algorithms:
            try:
                result = self._optimize_algorithm(
                    algorithm=algorithm,
                    X_ml_train=X_ml_train,
                    X_ml_val=X_ml_val,
                    X_stat_train=X_stat_train,
                    X_stat_val=X_stat_val,
                    y_train=y_train,
                    y_val=y_val,
                    date_column=date_column,
                    cv_splits=cv_splits
                )
                all_results.append(result)
                
                # Update best model
                if result["best_score"] < self.best_score:
                    self.best_score = result["best_score"]
                    self.best_model = result["best_model"]
                    self.best_algorithm = algorithm
                    
            except Exception as e:
                all_results.append({
                    "algorithm": algorithm,
                    "status": "failed",
                    "error": str(e),
                    "best_score": float('inf')
                })
        
        total_time = time.time() - start_time
        
        # Sort results by score
        all_results.sort(key=lambda x: x.get("best_score", float('inf')))
        self.results = all_results
        
        return {
            "best_algorithm": self.best_algorithm,
            "best_score": self.best_score,
            "best_model": self.best_model,
            "all_results": all_results,
            "total_time_seconds": total_time,
            "algorithms_tested": algorithms
        }
    
    def _optimize_algorithm(
        self,
        algorithm: str,
        X_ml_train: pd.DataFrame,
        X_ml_val: pd.DataFrame,
        X_stat_train: pd.DataFrame,
        X_stat_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        date_column: str,
        cv_splits: int
    ) -> Dict[str, Any]:
        """Optimize hyperparameters for a single algorithm."""
        start_time = time.time()
        
        model_class = self.AVAILABLE_ALGORITHMS[algorithm]
        
        def objective(trial):
            # Get hyperparameters based on algorithm
            params = self._suggest_params(trial, algorithm)
            
            try:
                model = model_class(**params)
                
                # Use appropriate data based on model type
                if algorithm in ["prophet", "arima"]:
                    model.fit(X_stat_train, y_train, date_column=date_column)
                    predictions = model.predict(X_stat_val)
                else:
                    model.fit(X_ml_train, y_train)
                    predictions = model.predict(X_ml_val)
                
                # Calculate metric
                metrics = model.evaluate(y_val, predictions)
                return metrics[self.metric]
                
            except Exception as e:
                return float('inf')
        
        # Run optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(
            objective,
            n_trials=self.max_trials // len(self.AVAILABLE_ALGORITHMS),
            timeout=self.timeout_seconds // len(self.AVAILABLE_ALGORITHMS),
            show_progress_bar=False
        )
        
        # Train final model with best params
        best_params = study.best_params
        best_model = model_class(**best_params)
        
        if algorithm in ["prophet", "arima"]:
            best_model.fit(X_stat_train, y_train, date_column=date_column)
            predictions = best_model.predict(X_stat_val)
        else:
            best_model.fit(X_ml_train, y_train)
            predictions = best_model.predict(X_ml_val)
        
        metrics = best_model.evaluate(y_val, predictions)
        
        return {
            "algorithm": algorithm,
            "status": "completed",
            "best_score": metrics[self.metric],
            "best_params": best_params,
            "best_model": best_model,
            "metrics": metrics,
            "training_time": time.time() - start_time,
            "n_trials": len(study.trials)
        }
    
    def _suggest_params(self, trial: optuna.Trial, algorithm: str) -> Dict[str, Any]:
        """Suggest hyperparameters for Optuna optimization."""
        if algorithm == "prophet":
            return {
                "seasonality_mode": trial.suggest_categorical(
                    "seasonality_mode", ["additive", "multiplicative"]
                ),
                "changepoint_prior_scale": trial.suggest_float(
                    "changepoint_prior_scale", 0.001, 0.5, log=True
                ),
                "seasonality_prior_scale": trial.suggest_float(
                    "seasonality_prior_scale", 0.01, 10.0, log=True
                ),
            }
        
        elif algorithm == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
        
        elif algorithm == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }
        
        elif algorithm == "arima":
            return {
                "auto_arima": True,
                "max_p": trial.suggest_int("max_p", 1, 5),
                "max_q": trial.suggest_int("max_q", 1, 5),
                "seasonal": trial.suggest_categorical("seasonal", [True, False]),
            }
        
        return {}
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all tested models."""
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        for result in self.results:
            if result.get("status") == "completed":
                comparison_data.append({
                    "Algorithm": result["algorithm"],
                    "MAPE": result["metrics"].get("mape", np.nan),
                    "RMSE": result["metrics"].get("rmse", np.nan),
                    "MAE": result["metrics"].get("mae", np.nan),
                    "R2": result["metrics"].get("r2", np.nan),
                    "Training Time (s)": result["training_time"],
                    "Trials": result["n_trials"]
                })
        
        return pd.DataFrame(comparison_data).sort_values("MAPE")
