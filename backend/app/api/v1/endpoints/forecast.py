from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

from app.core.database import get_db
from app.models import Dataset, TrainedModel, Forecast, ModelStatus
from app.schemas import ForecastRequest, ForecastResponse, PredictionPoint
from app.services.feature_engineering import TimeSeriesFeatureEngineer

router = APIRouter()


@router.post("/", response_model=ForecastResponse)
async def generate_forecast(
    request: ForecastRequest,
    db: AsyncSession = Depends(get_db)
):
    """Generate forecast using a trained model."""
    # Get trained model
    result = await db.execute(
        select(TrainedModel).where(TrainedModel.id == request.model_id)
    )
    trained_model = result.scalar_one_or_none()
    
    if not trained_model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trained model not found"
        )
    
    if trained_model.status != ModelStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model is not ready for forecasting"
        )
    
    # Get dataset
    result = await db.execute(
        select(Dataset).where(Dataset.id == trained_model.dataset_id)
    )
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    try:
        # Load the saved model (it's a forecaster object saved with joblib)
        model = joblib.load(trained_model.model_path)
        
        # Load dataset to get the last date and historical data
        df = pd.read_csv(dataset.file_path)
        df[dataset.date_column] = pd.to_datetime(df[dataset.date_column])
        df = df.sort_values(dataset.date_column)
        last_date = df[dataset.date_column].max()
        
        # Generate future dates
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=request.forecast_horizon,
            freq='D'
        )
        
        # Generate predictions based on algorithm type
        if trained_model.algorithm in ["prophet", "arima"]:
            # Statistical models - create future dataframe
            X_future = pd.DataFrame({dataset.date_column: future_dates})
            
            if hasattr(model, 'predict_interval'):
                predictions, lower, upper = model.predict_interval(
                    X_future, confidence=request.confidence_level
                )
            else:
                predictions = model.predict(X_future)
                # Generate simple confidence bounds
                std = np.std(df[dataset.target_column].values) * 0.1
                lower = predictions - 1.96 * std
                upper = predictions + 1.96 * std
        else:
            # ML models (XGBoost, LightGBM) - need engineered features
            # Create a temporary dataframe with future dates
            future_df = pd.DataFrame({
                dataset.date_column: future_dates,
                dataset.target_column: [np.nan] * len(future_dates)
            })
            
            # Combine with historical data for feature engineering
            combined_df = pd.concat([df, future_df], ignore_index=True)
            
            # Engineer features
            feature_engineer = TimeSeriesFeatureEngineer(
                combined_df, dataset.date_column, dataset.target_column
            )
            df_features = feature_engineer.create_all_features()
            
            # Get future rows
            X_future = df_features.tail(request.forecast_horizon)
            feature_cols = feature_engineer.get_feature_names()
            
            # Filter to only use columns that exist
            available_cols = [c for c in feature_cols if c in X_future.columns]
            X_future = X_future[available_cols].fillna(0)
            
            # Make predictions
            if hasattr(model, 'predict_interval'):
                predictions, lower, upper = model.predict_interval(
                    X_future, confidence=request.confidence_level
                )
            elif hasattr(model, 'predict'):
                predictions = model.predict(X_future)
                # Generate simple confidence bounds based on historical std
                std = np.std(df[dataset.target_column].values) * 0.1
                z_score = 1.96 if request.confidence_level >= 0.95 else 1.645
                lower = predictions - z_score * std
                upper = predictions + z_score * std
            elif hasattr(model, 'model') and hasattr(model.model, 'predict'):
                # The forecaster wraps the actual model
                predictions = model.model.predict(X_future)
                std = np.std(df[dataset.target_column].values) * 0.1
                z_score = 1.96 if request.confidence_level >= 0.95 else 1.645
                lower = predictions - z_score * std
                upper = predictions + z_score * std
            else:
                raise ValueError(f"Model doesn't have predict method")
        
        # Ensure predictions are numpy arrays
        predictions = np.array(predictions).flatten()
        lower = np.array(lower).flatten() if lower is not None else predictions * 0.9
        upper = np.array(upper).flatten() if upper is not None else predictions * 1.1
        
        # Create prediction points
        prediction_points = []
        for i, date in enumerate(future_dates):
            prediction_points.append(PredictionPoint(
                date=date.strftime('%Y-%m-%d'),
                value=float(predictions[i]),
                lower_bound=float(lower[i]) if lower is not None else None,
                upper_bound=float(upper[i]) if upper is not None else None
            ))
        
        # Save forecast to database
        forecast = Forecast(
            dataset_id=dataset.id,
            model_id=trained_model.id,
            forecast_horizon=request.forecast_horizon,
            confidence_level=request.confidence_level,
            predictions=[p.model_dump() for p in prediction_points]
        )
        
        db.add(forecast)
        await db.commit()
        await db.refresh(forecast)
        
        return ForecastResponse(
            id=forecast.id,
            dataset_id=forecast.dataset_id,
            model_id=forecast.model_id,
            forecast_horizon=forecast.forecast_horizon,
            confidence_level=forecast.confidence_level,
            predictions=prediction_points,
            created_at=forecast.created_at
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecast generation failed: {str(e)}"
        )


@router.get("/{forecast_id}", response_model=ForecastResponse)
async def get_forecast(
    forecast_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific forecast by ID."""
    result = await db.execute(select(Forecast).where(Forecast.id == forecast_id))
    forecast = result.scalar_one_or_none()
    
    if not forecast:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Forecast not found"
        )
    
    # Convert stored predictions to PredictionPoint objects
    predictions = [
        PredictionPoint(
            date=p["date"],
            value=p["value"],
            lower_bound=p.get("lower_bound"),
            upper_bound=p.get("upper_bound")
        )
        for p in forecast.predictions
    ]
    
    return ForecastResponse(
        id=forecast.id,
        dataset_id=forecast.dataset_id,
        model_id=forecast.model_id,
        forecast_horizon=forecast.forecast_horizon,
        confidence_level=forecast.confidence_level,
        predictions=predictions,
        created_at=forecast.created_at
    )


@router.get("/dataset/{dataset_id}", response_model=List[ForecastResponse])
async def list_forecasts_by_dataset(
    dataset_id: int,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """List all forecasts for a dataset."""
    result = await db.execute(
        select(Forecast)
        .where(Forecast.dataset_id == dataset_id)
        .offset(skip)
        .limit(limit)
        .order_by(Forecast.created_at.desc())
    )
    forecasts = result.scalars().all()
    
    response = []
    for forecast in forecasts:
        predictions = [
            PredictionPoint(
                date=p["date"],
                value=p["value"],
                lower_bound=p.get("lower_bound"),
                upper_bound=p.get("upper_bound")
            )
            for p in forecast.predictions
        ]
        response.append(ForecastResponse(
            id=forecast.id,
            dataset_id=forecast.dataset_id,
            model_id=forecast.model_id,
            forecast_horizon=forecast.forecast_horizon,
            confidence_level=forecast.confidence_level,
            predictions=predictions,
            created_at=forecast.created_at
        ))
    
    return response
