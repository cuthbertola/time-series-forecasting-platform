from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import logging
import numpy as np
from datetime import datetime
import pandas as pd

from app.core.database import get_db, SessionLocal
from app.models.models import Dataset, TrainedModel, AutoMLRun, ModelStatus
from app.schemas.schemas import (
    TrainedModelResponse,
    AutoMLRequest,
    AutoMLRunResponse,
    ModelComparisonResponse
)
from app.services.automl import AutoMLService

router = APIRouter()
logger = logging.getLogger(__name__)


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


@router.post("/automl", response_model=AutoMLRunResponse)
async def run_automl(
    request: AutoMLRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Start an AutoML training run."""
    from sqlalchemy import select
    result = await db.execute(select(Dataset).filter(Dataset.id == request.dataset_id))
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if dataset.status != "ready":
        raise HTTPException(status_code=400, detail="Dataset is not ready for training")
    
    automl_run = AutoMLRun(
        dataset_id=request.dataset_id,
        algorithms_tested=request.algorithms or ["prophet", "arima", "xgboost", "lightgbm"],
        max_trials=request.max_trials or 50,
        status="running",
        created_at=datetime.utcnow()
    )
    db.add(automl_run)
    await db.commit()
    await db.refresh(automl_run)
    
    background_tasks.add_task(
        _run_automl_task,
        automl_run.id,
        request.dataset_id,
        request.target_column,
        request.date_column,
        request.feature_columns,
        request.forecast_horizon or 30,
        request.algorithms or ["prophet", "arima", "xgboost", "lightgbm"],
        request.max_trials or 50,
        request.timeout_seconds or 300
    )
    
    return automl_run


def _run_automl_task(
    automl_run_id: int,
    dataset_id: int,
    target_column: str,
    date_column: str,
    feature_columns: Optional[List[str]],
    forecast_horizon: int,
    algorithms: List[str],
    max_trials: int,
    timeout_seconds: int
):
    """Background task to run AutoML training."""
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise ValueError("Dataset not found")
        
        df = pd.read_csv(dataset.file_path)
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)
        
        logger.info(f"Loaded dataset with {len(df)} rows")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Target column: {target_column}, Date column: {date_column}")
        
        # Create AutoML service with max_trials and timeout in constructor
        automl_service = AutoMLService(
            max_trials=max_trials,
            timeout_seconds=timeout_seconds
        )
        
        # Run AutoML - only pass parameters that run() accepts
        results = automl_service.run(
            df=df,
            target_column=target_column,
            date_column=date_column,
            feature_columns=feature_columns,
            algorithms=algorithms,
            forecast_horizon=forecast_horizon
        )
        
        logger.info(f"AutoML completed. Best algorithm: {results.get('best_algorithm')}")
        
        serializable_results = convert_to_serializable(results)
        
        if serializable_results.get("best_model"):
            import joblib
            import os
            
            model_dir = "data/models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = f"{model_dir}/model_{dataset_id}_{serializable_results['best_algorithm']}.joblib"
            joblib.dump(serializable_results["best_model"], model_path)
            
            logger.info(f"Saved model to {model_path}")
            
            feature_importance = None
            best_model = serializable_results["best_model"]
            
            if hasattr(best_model, "feature_importance_"):
                fi = best_model.feature_importance_
                feature_names = serializable_results.get("feature_names", [f"feature_{i}" for i in range(len(fi))])
                feature_importance = convert_to_serializable(dict(zip(feature_names, fi.tolist() if hasattr(fi, 'tolist') else list(fi))))
            elif hasattr(best_model, "model") and hasattr(best_model.model, "feature_importances_"):
                fi = best_model.model.feature_importances_
                feature_names = serializable_results.get("feature_names", [f"feature_{i}" for i in range(len(fi))])
                feature_importance = convert_to_serializable(dict(zip(feature_names, fi.tolist() if hasattr(fi, 'tolist') else list(fi))))
            
            hyperparameters = None
            best_result = next((r for r in serializable_results.get("all_results", []) if r.get("algorithm") == serializable_results["best_algorithm"]), None)
            if best_result:
                hyperparameters = convert_to_serializable(best_result.get("best_params", {}))
            
            trained_model = TrainedModel(
                dataset_id=dataset_id,
                name=f"AutoML Best - {serializable_results['best_algorithm']}",
                algorithm=serializable_results["best_algorithm"],
                hyperparameters=hyperparameters,
                feature_importance=feature_importance,
                mape=float(serializable_results.get("best_score", 0)) if serializable_results.get("best_score") else None,
                training_time_seconds=float(serializable_results.get("total_time", 0)) if serializable_results.get("total_time") else None,
                model_path=model_path,
                status=ModelStatus.COMPLETED,
                is_best_model=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(trained_model)
            db.commit()
            db.refresh(trained_model)
            
            logger.info(f"Created trained model record with ID {trained_model.id}")
            
            automl_run = db.query(AutoMLRun).filter(AutoMLRun.id == automl_run_id).first()
            if automl_run:
                automl_run.best_algorithm = serializable_results["best_algorithm"]
                automl_run.best_model_id = trained_model.id
                automl_run.all_results = convert_to_serializable([
                    {
                        "algorithm": r.get("algorithm"),
                        "status": r.get("status"),
                        "mape": float(r.get("best_score")) if r.get("best_score") else None,
                        "training_time": float(r.get("training_time")) if r.get("training_time") else None
                    }
                    for r in serializable_results.get("all_results", [])
                ])
                automl_run.status = "completed"
                automl_run.total_time_seconds = float(serializable_results.get("total_time", 0))
                automl_run.completed_at = datetime.utcnow()
                db.commit()
                
            logger.info("AutoML run completed successfully")
        else:
            logger.warning("No model was trained successfully")
            automl_run = db.query(AutoMLRun).filter(AutoMLRun.id == automl_run_id).first()
            if automl_run:
                automl_run.status = "failed"
                automl_run.error_message = "No model was trained successfully"
                automl_run.completed_at = datetime.utcnow()
                db.commit()
                
    except Exception as e:
        logger.error(f"AutoML task failed: {str(e)}")
        import traceback
        traceback.print_exc()
        try:
            db.rollback()
            automl_run = db.query(AutoMLRun).filter(AutoMLRun.id == automl_run_id).first()
            if automl_run:
                automl_run.status = "failed"
                automl_run.error_message = str(e)
                automl_run.completed_at = datetime.utcnow()
                db.commit()
        except Exception as e2:
            logger.error(f"Failed to update AutoML run status: {str(e2)}")
    finally:
        db.close()


@router.get("/automl/{run_id}", response_model=AutoMLRunResponse)
async def get_automl_run(run_id: int, db: AsyncSession = Depends(get_db)):
    """Get AutoML run status and results."""
    from sqlalchemy import select
    result = await db.execute(select(AutoMLRun).filter(AutoMLRun.id == run_id))
    automl_run = result.scalar_one_or_none()
    
    if not automl_run:
        raise HTTPException(status_code=404, detail="AutoML run not found")
    
    return automl_run


@router.get("/models", response_model=List[TrainedModelResponse])
async def list_models(
    dataset_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
):
    """List all trained models."""
    from sqlalchemy import select
    query = select(TrainedModel)
    if dataset_id:
        query = query.filter(TrainedModel.dataset_id == dataset_id)
    
    result = await db.execute(query)
    models = result.scalars().all()
    return models


@router.get("/models/{model_id}", response_model=TrainedModelResponse)
async def get_model(model_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific trained model."""
    from sqlalchemy import select
    result = await db.execute(select(TrainedModel).filter(TrainedModel.id == model_id))
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return model


@router.get("/models/compare/{dataset_id}", response_model=ModelComparisonResponse)
async def compare_models(dataset_id: int, db: AsyncSession = Depends(get_db)):
    """Compare all models trained on a dataset."""
    from sqlalchemy import select
    result = await db.execute(
        select(TrainedModel).filter(TrainedModel.dataset_id == dataset_id)
    )
    models = result.scalars().all()
    
    if not models:
        raise HTTPException(status_code=404, detail="No models found for this dataset")
    
    best_model = min(models, key=lambda m: m.mape if m.mape else float('inf'))
    
    comparison_metrics = {}
    for model in models:
        comparison_metrics[model.algorithm] = {
            "mape": model.mape,
            "rmse": model.rmse,
            "mae": model.mae,
            "r2": model.r2_score,
            "training_time": model.training_time_seconds
        }
    
    return {
        "models": models,
        "best_model_id": best_model.id,
        "comparison_metrics": comparison_metrics
    }
