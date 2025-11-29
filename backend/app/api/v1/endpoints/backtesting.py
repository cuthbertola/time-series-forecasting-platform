from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
import pandas as pd

from app.core.database import get_db
from app.models import Dataset
from app.services.backtesting import run_backtest
from app.ml.feature_engineering import TimeSeriesFeatureEngineer

router = APIRouter()

class BacktestRequest(BaseModel):
    dataset_id: int
    model_type: str = "xgboost"
    initial_train_days: int = 365
    test_days: int = 30
    step_days: int = 30

@router.post("/run")
async def run_backtest_endpoint(request: BacktestRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Dataset).where(Dataset.id == request.dataset_id))
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if request.model_type not in ["xgboost", "lightgbm"]:
        raise HTTPException(status_code=400, detail="Only xgboost and lightgbm supported")
    
    try:
        df = pd.read_csv(dataset.file_path)
        date_column = dataset.date_column or 'date'
        target_column = dataset.target_column or 'sales'
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)
        
        feature_engineer = TimeSeriesFeatureEngineer()
        X, y, feature_names = feature_engineer.create_features(df, date_column=date_column, target_column=target_column)
        dates = df[date_column].iloc[feature_engineer.lookback:]
        
        min_required = request.initial_train_days + request.test_days
        if len(X) < min_required:
            raise HTTPException(status_code=400, detail=f"Not enough data. Need {min_required}, have {len(X)}")
        
        results = run_backtest(request.model_type, X, y, dates, request.initial_train_days, request.test_days, request.step_days)
        
        return {
            "dataset_id": request.dataset_id,
            "dataset_name": dataset.name,
            "model_type": request.model_type,
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Backtest error: {str(e)}")
