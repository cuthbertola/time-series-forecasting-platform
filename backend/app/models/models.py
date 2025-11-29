from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, ForeignKey, Boolean, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base


class DatasetStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class ModelStatus(str, enum.Enum):
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    DEPLOYED = "deployed"


class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    
    # Data characteristics
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    date_column = Column(String(100), nullable=True)
    target_column = Column(String(100), nullable=True)
    feature_columns = Column(JSON, nullable=True)
    
    # Time series metadata
    frequency = Column(String(50), nullable=True)  # daily, weekly, monthly
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    
    # Status and timestamps
    status = Column(Enum(DatasetStatus), default=DatasetStatus.PENDING)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    forecasts = relationship("Forecast", back_populates="dataset", cascade="all, delete-orphan")
    trained_models = relationship("TrainedModel", back_populates="dataset", cascade="all, delete-orphan")


class TrainedModel(Base):
    __tablename__ = "trained_models"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    name = Column(String(255), nullable=False)
    algorithm = Column(String(100), nullable=False)  # prophet, arima, xgboost, lstm, lightgbm
    
    # Model configuration
    hyperparameters = Column(JSON, nullable=True)
    feature_importance = Column(JSON, nullable=True)
    
    # Performance metrics
    mape = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    r2_score = Column(Float, nullable=True)
    
    # Training details
    training_time_seconds = Column(Float, nullable=True)
    cv_scores = Column(JSON, nullable=True)
    
    # MLflow tracking
    mlflow_run_id = Column(String(100), nullable=True)
    mlflow_model_uri = Column(String(500), nullable=True)
    
    # Model file path
    model_path = Column(String(500), nullable=True)
    
    # Status and timestamps
    status = Column(Enum(ModelStatus), default=ModelStatus.TRAINING)
    is_best_model = Column(Boolean, default=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="trained_models")
    forecasts = relationship("Forecast", back_populates="model", cascade="all, delete-orphan")


class Forecast(Base):
    __tablename__ = "forecasts"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    model_id = Column(Integer, ForeignKey("trained_models.id"), nullable=False)
    
    # Forecast configuration
    forecast_horizon = Column(Integer, nullable=False)
    confidence_level = Column(Float, default=0.95)
    
    # Forecast results stored as JSON
    predictions = Column(JSON, nullable=False)  # [{date, value, lower, upper}, ...]
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="forecasts")
    model = relationship("TrainedModel", back_populates="forecasts")


class AutoMLRun(Base):
    __tablename__ = "automl_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    
    # AutoML configuration
    algorithms_tested = Column(JSON, nullable=True)
    max_trials = Column(Integer, nullable=True)
    timeout_seconds = Column(Integer, nullable=True)
    
    # Results
    best_algorithm = Column(String(100), nullable=True)
    best_model_id = Column(Integer, ForeignKey("trained_models.id"), nullable=True)
    all_results = Column(JSON, nullable=True)  # [{algorithm, mape, training_time}, ...]
    
    # Status and timestamps
    status = Column(String(50), default="running")
    total_time_seconds = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
