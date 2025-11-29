from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import pandas as pd
import numpy as np
import io
import joblib
from typing import Optional

from app.core.database import get_db
from app.models import Dataset, TrainedModel, ModelStatus
from app.services.feature_engineering import TimeSeriesFeatureEngineer

router = APIRouter()


@router.post("/predict/{model_id}")
async def batch_predict(
    model_id: int,
    file: UploadFile = File(...),
    date_column: str = Form(...),
    confidence_level: float = Form(0.95),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate batch predictions from an uploaded CSV file with future dates.
    
    The CSV should contain at minimum a date column with the dates to predict.
    """
    # Get the model
    result = await db.execute(select(TrainedModel).where(TrainedModel.id == model_id))
    model_record = result.scalar_one_or_none()
    
    if not model_record:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if model_record.status != ModelStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Model is not ready")
    
    # Get the dataset for feature engineering reference
    result = await db.execute(select(Dataset).where(Dataset.id == model_record.dataset_id))
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Original dataset not found")
    
    try:
        # Read uploaded file
        contents = await file.read()
        future_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if date_column not in future_df.columns:
            raise HTTPException(status_code=400, detail=f"Date column '{date_column}' not found in uploaded file")
        
        future_df[date_column] = pd.to_datetime(future_df[date_column])
        future_df = future_df.sort_values(date_column)
        
        # Load the trained model
        model = joblib.load(model_record.model_path)
        
        # Load original dataset for feature engineering
        original_df = pd.read_csv(dataset.file_path)
        original_df[dataset.date_column] = pd.to_datetime(original_df[dataset.date_column])
        original_df = original_df.sort_values(dataset.date_column)
        
        # Prepare predictions based on algorithm
        if model_record.algorithm in ["prophet", "arima"]:
            # Statistical models
            X_future = future_df[[date_column]].copy()
            X_future.columns = [dataset.date_column]
            
            if hasattr(model, 'predict_interval'):
                predictions, lower, upper = model.predict_interval(X_future, confidence=confidence_level)
            else:
                predictions = model.predict(X_future)
                std = np.std(original_df[dataset.target_column].values) * 0.1
                lower = predictions - 1.96 * std
                upper = predictions + 1.96 * std
        else:
            # ML models - need feature engineering
            future_df[dataset.target_column] = np.nan
            
            # Rename date column if needed
            if date_column != dataset.date_column:
                future_df = future_df.rename(columns={date_column: dataset.date_column})
            
            # Combine with historical data
            combined_df = pd.concat([original_df, future_df], ignore_index=True)
            
            # Feature engineering
            feature_engineer = TimeSeriesFeatureEngineer(
                combined_df, dataset.date_column, dataset.target_column
            )
            df_features = feature_engineer.create_all_features()
            
            # Get future rows
            X_future = df_features.tail(len(future_df))
            feature_cols = feature_engineer.get_feature_names()
            available_cols = [c for c in feature_cols if c in X_future.columns]
            X_future = X_future[available_cols].fillna(0)
            
            # Predict
            if hasattr(model, 'predict'):
                predictions = model.predict(X_future)
            elif hasattr(model, 'model') and hasattr(model.model, 'predict'):
                predictions = model.model.predict(X_future)
            else:
                raise ValueError("Model doesn't have predict method")
            
            # Generate confidence bounds
            std = np.std(original_df[dataset.target_column].values) * 0.1
            z_score = 1.96 if confidence_level >= 0.95 else 1.645
            lower = predictions - z_score * std
            upper = predictions + z_score * std
        
        # Ensure arrays
        predictions = np.array(predictions).flatten()
        lower = np.array(lower).flatten()
        upper = np.array(upper).flatten()
        
        # Create output DataFrame
        output_df = pd.DataFrame({
            'date': future_df[date_column if date_column in future_df.columns else dataset.date_column].values,
            'forecast': predictions,
            'lower_bound': lower,
            'upper_bound': upper
        })
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        output_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        return StreamingResponse(
            csv_buffer,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=batch_predictions_{model_id}.csv"
            }
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.get("/template")
async def download_batch_template():
    """Download a template CSV for batch predictions."""
    template = """date
2024-01-01
2024-01-02
2024-01-03
2024-01-04
2024-01-05
2024-01-06
2024-01-07
"""
    return StreamingResponse(
        io.StringIO(template),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=batch_prediction_template.csv"
        }
    )
