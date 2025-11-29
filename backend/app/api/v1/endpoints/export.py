from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import os
import json
import joblib
import io
import zipfile
from datetime import datetime

from app.core.database import get_db
from app.models import Dataset, TrainedModel, Forecast

router = APIRouter()


@router.get("/model/{model_id}")
async def export_model(
    model_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Export a trained model as a downloadable file."""
    result = await db.execute(select(TrainedModel).where(TrainedModel.id == model_id))
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not model.model_path or not os.path.exists(model.model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return FileResponse(
        path=model.model_path,
        filename=f"{model.name.replace(' ', '_')}_{model.algorithm}.joblib",
        media_type="application/octet-stream"
    )


@router.get("/model/{model_id}/metadata")
async def export_model_metadata(
    model_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Export model metadata as JSON."""
    result = await db.execute(select(TrainedModel).where(TrainedModel.id == model_id))
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    metadata = {
        "model_id": model.id,
        "name": model.name,
        "algorithm": model.algorithm,
        "hyperparameters": model.hyperparameters,
        "feature_importance": model.feature_importance,
        "metrics": {
            "mape": model.mape,
            "rmse": model.rmse,
            "mae": model.mae,
            "r2_score": model.r2_score
        },
        "training_time_seconds": model.training_time_seconds,
        "created_at": model.created_at.isoformat() if model.created_at else None,
        "status": model.status.value if model.status else None
    }
    
    return metadata


@router.get("/model/{model_id}/package")
async def export_model_package(
    model_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Export model with metadata as a ZIP package."""
    result = await db.execute(select(TrainedModel).where(TrainedModel.id == model_id))
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not model.model_path or not os.path.exists(model.model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add model file
        zip_file.write(model.model_path, f"model.joblib")
        
        # Add metadata
        metadata = {
            "model_id": model.id,
            "name": model.name,
            "algorithm": model.algorithm,
            "hyperparameters": model.hyperparameters,
            "feature_importance": model.feature_importance,
            "metrics": {
                "mape": model.mape,
                "rmse": model.rmse,
                "mae": model.mae,
                "r2_score": model.r2_score
            },
            "training_time_seconds": model.training_time_seconds,
            "created_at": model.created_at.isoformat() if model.created_at else None
        }
        zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))
        
        # Add README
        readme = f"""# Exported Model: {model.name}

## Algorithm: {model.algorithm}

## Metrics
- MAPE: {model.mape}
- RMSE: {model.rmse}
- MAE: {model.mae}

## Usage
```python
import joblib

# Load the model
model = joblib.load('model.joblib')

# Make predictions
predictions = model.predict(X_new)
```

## Exported: {datetime.now().isoformat()}
"""
        zip_file.writestr("README.md", readme)
    
    zip_buffer.seek(0)
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={model.name.replace(' ', '_')}_package.zip"
        }
    )


@router.get("/forecast/{forecast_id}/csv")
async def export_forecast_csv(
    forecast_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Export forecast results as CSV."""
    result = await db.execute(select(Forecast).where(Forecast.id == forecast_id))
    forecast = result.scalar_one_or_none()
    
    if not forecast:
        raise HTTPException(status_code=404, detail="Forecast not found")
    
    # Create CSV content
    csv_lines = ["date,forecast,lower_bound,upper_bound"]
    for pred in forecast.predictions:
        csv_lines.append(f"{pred['date']},{pred['value']},{pred.get('lower_bound', '')},{pred.get('upper_bound', '')}")
    
    csv_content = "\n".join(csv_lines)
    
    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=forecast_{forecast_id}.csv"
        }
    )
