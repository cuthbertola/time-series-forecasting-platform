from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from app.core.database import get_db
from app.models import Dataset

router = APIRouter()


@router.get("/historical/{dataset_id}")
async def get_historical_data(
    dataset_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get historical time series data for visualization."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        df = pd.read_csv(dataset.file_path)
        df[dataset.date_column] = pd.to_datetime(df[dataset.date_column])
        df = df.sort_values(dataset.date_column)
        
        # Prepare data for chart
        chart_data = []
        for _, row in df.iterrows():
            chart_data.append({
                "date": row[dataset.date_column].strftime('%Y-%m-%d'),
                "value": float(row[dataset.target_column]) if pd.notna(row[dataset.target_column]) else None
            })
        
        return {
            "dataset_name": dataset.name,
            "date_column": dataset.date_column,
            "target_column": dataset.target_column,
            "data": chart_data,
            "total_points": len(chart_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")


@router.get("/statistics/{dataset_id}")
async def get_dataset_statistics(
    dataset_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get statistical summary of the dataset."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        df = pd.read_csv(dataset.file_path)
        df[dataset.date_column] = pd.to_datetime(df[dataset.date_column])
        target = df[dataset.target_column]
        
        # Basic statistics
        stats = {
            "count": int(len(target)),
            "mean": float(target.mean()),
            "std": float(target.std()),
            "min": float(target.min()),
            "max": float(target.max()),
            "median": float(target.median()),
            "q25": float(target.quantile(0.25)),
            "q75": float(target.quantile(0.75)),
        }
        
        # Date range
        date_stats = {
            "start_date": df[dataset.date_column].min().strftime('%Y-%m-%d'),
            "end_date": df[dataset.date_column].max().strftime('%Y-%m-%d'),
            "total_days": int((df[dataset.date_column].max() - df[dataset.date_column].min()).days)
        }
        
        return {
            "dataset_name": dataset.name,
            "statistics": stats,
            "date_range": date_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating statistics: {str(e)}")


@router.get("/seasonality/{dataset_id}")
async def get_seasonality_analysis(
    dataset_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Analyze seasonality patterns in the data."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        df = pd.read_csv(dataset.file_path)
        df[dataset.date_column] = pd.to_datetime(df[dataset.date_column])
        df = df.sort_values(dataset.date_column)
        
        # Day of week analysis
        df['day_of_week'] = df[dataset.date_column].dt.dayofweek
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_pattern = df.groupby('day_of_week')[dataset.target_column].mean()
        weekly_data = [{"day": day_names[i], "value": float(weekly_pattern.get(i, 0))} for i in range(7)]
        
        # Monthly analysis
        df['month'] = df[dataset.date_column].dt.month
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_pattern = df.groupby('month')[dataset.target_column].mean()
        monthly_data = [{"month": month_names[i-1], "value": float(monthly_pattern.get(i, 0))} for i in range(1, 13) if i in monthly_pattern.index]
        
        # Trend (rolling average)
        df['rolling_avg'] = df[dataset.target_column].rolling(window=7, min_periods=1).mean()
        trend_data = []
        for _, row in df.iterrows():
            trend_data.append({
                "date": row[dataset.date_column].strftime('%Y-%m-%d'),
                "value": float(row[dataset.target_column]) if pd.notna(row[dataset.target_column]) else None,
                "trend": float(row['rolling_avg']) if pd.notna(row['rolling_avg']) else None
            })
        
        return {
            "weekly_pattern": weekly_data,
            "monthly_pattern": monthly_data,
            "trend_data": trend_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing seasonality: {str(e)}")
