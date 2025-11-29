from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import math


class AlgorithmEnum(str, Enum):
    prophet = "prophet"
    arima = "arima"
    xgboost = "xgboost"
    lightgbm = "lightgbm"
    ensemble = "ensemble"


class DatasetStatusEnum(str, Enum):
    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class ModelStatusEnum(str, Enum):
    training = "training"
    completed = "completed"
    failed = "failed"
    deployed = "deployed"


# Helper function to sanitize float values
def sanitize_float(value):
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


# Dataset Schemas
class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    date_column: Optional[str] = None
    target_column: Optional[str] = None


class DatasetResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    filename: str
    file_path: str
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    date_column: Optional[str] = None
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    frequency: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DatasetListResponse(BaseModel):
    datasets: List[DatasetResponse]
    total: int


# Training Schemas
class TrainingRequest(BaseModel):
    dataset_id: int
    algorithms: Optional[List[AlgorithmEnum]] = None
    target_column: str
    date_column: str
    feature_columns: Optional[List[str]] = None
    forecast_horizon: int = Field(default=30, ge=1, le=365)
    use_automl: bool = True
    automl_max_trials: int = Field(default=50, ge=1, le=200)
    timeout_seconds: int = Field(default=300, ge=60, le=3600)


class TrainedModelResponse(BaseModel):
    id: int
    dataset_id: int
    name: str
    algorithm: str
    hyperparameters: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    mape: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    training_time_seconds: Optional[float] = None
    cv_scores: Optional[List[float]] = None
    mlflow_run_id: Optional[str] = None
    model_path: Optional[str] = None
    status: str
    is_best_model: bool = False
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    @field_validator('mape', 'rmse', 'mae', 'r2_score', 'training_time_seconds', mode='before')
    @classmethod
    def sanitize_floats(cls, v):
        return sanitize_float(v)

    class Config:
        from_attributes = True


class ModelComparisonResponse(BaseModel):
    models: List[TrainedModelResponse]
    best_model_id: int
    comparison_metrics: Dict[str, Dict[str, Optional[float]]]


# AutoML Schemas
class AutoMLRequest(BaseModel):
    dataset_id: int
    target_column: str
    date_column: str
    feature_columns: Optional[List[str]] = None
    forecast_horizon: int = Field(default=30, ge=1, le=365)
    algorithms: Optional[List[str]] = None
    max_trials: int = Field(default=50, ge=1, le=200)
    timeout_seconds: int = Field(default=300, ge=60, le=3600)


class AutoMLResultItem(BaseModel):
    algorithm: str
    status: str
    mape: Optional[float] = None
    training_time: Optional[float] = None

    @field_validator('mape', 'training_time', mode='before')
    @classmethod
    def sanitize_floats(cls, v):
        return sanitize_float(v)


class AutoMLRunResponse(BaseModel):
    id: int
    dataset_id: int
    algorithms_tested: List[str]
    max_trials: Optional[int] = None
    best_algorithm: Optional[str] = None
    best_model_id: Optional[int] = None
    all_results: Optional[List[Dict[str, Any]]] = None
    status: str
    total_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    @field_validator('total_time_seconds', mode='before')
    @classmethod
    def sanitize_floats(cls, v):
        return sanitize_float(v)
    
    @field_validator('all_results', mode='before')
    @classmethod
    def sanitize_all_results(cls, v):
        if v is None:
            return None
        sanitized = []
        for item in v:
            if isinstance(item, dict):
                sanitized_item = {}
                for k, val in item.items():
                    if isinstance(val, float):
                        sanitized_item[k] = sanitize_float(val)
                    else:
                        sanitized_item[k] = val
                sanitized.append(sanitized_item)
            else:
                sanitized.append(item)
        return sanitized

    class Config:
        from_attributes = True


# Forecast Schemas
class ForecastRequest(BaseModel):
    model_id: int
    forecast_horizon: int = Field(default=30, ge=1, le=365)
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)


class PredictionPoint(BaseModel):
    date: str
    value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    @field_validator('value', 'lower_bound', 'upper_bound', mode='before')
    @classmethod
    def sanitize_floats(cls, v):
        return sanitize_float(v)


class ForecastResponse(BaseModel):
    id: int
    dataset_id: int
    model_id: int
    forecast_horizon: int
    confidence_level: float
    predictions: List[PredictionPoint]
    created_at: datetime

    class Config:
        from_attributes = True


# Feature Engineering Schemas
class FeatureEngineeringRequest(BaseModel):
    dataset_id: int
    create_lag_features: bool = True
    create_rolling_features: bool = True
    create_calendar_features: bool = True
    create_trend_features: bool = True
    lag_periods: List[int] = [1, 7, 14, 30]
    rolling_windows: List[int] = [7, 14, 30]


class FeatureEngineeringResponse(BaseModel):
    dataset_id: int
    features_created: List[str]
    total_features: int
    processing_time_seconds: float
