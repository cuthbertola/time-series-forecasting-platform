from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import pandas as pd
import numpy as np
import os

from app.core.database import get_db
from app.models import Dataset

router = APIRouter()


def detect_date_column(df: pd.DataFrame) -> str:
    """Auto-detect the date column in a dataframe."""
    common_names = ['date', 'Date', 'DATE', 'datetime', 'Datetime', 'timestamp', 'Timestamp', 'time', 'Time']
    for name in common_names:
        if name in df.columns:
            try:
                pd.to_datetime(df[name])
                return name
            except:
                pass
    
    for col in df.columns:
        try:
            pd.to_datetime(df[col])
            return col
        except:
            pass
    
    raise ValueError("No date column found in dataset")


def detect_target_column(df: pd.DataFrame, date_column: str) -> str:
    """Auto-detect the target column (first numeric column that isn't the date)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    common_names = ['sales', 'Sales', 'SALES', 'value', 'Value', 'VALUE', 'target', 'Target', 'amount', 'Amount', 'price', 'Price', 'revenue', 'Revenue']
    for name in common_names:
        if name in numeric_cols:
            return name
    
    if numeric_cols:
        return numeric_cols[0]
    
    raise ValueError("No numeric target column found in dataset")


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
        # Debug: print file path
        file_path = dataset.file_path
        print(f"DEBUG: Dataset ID={dataset_id}, file_path={file_path}")
        
        if not file_path:
            raise HTTPException(status_code=400, detail="Dataset has no file path")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        print(f"DEBUG: Loaded dataframe with columns: {df.columns.tolist()}")
        
        # Use stored columns or auto-detect
        date_column = dataset.date_column if dataset.date_column else detect_date_column(df)
        target_column = dataset.target_column if dataset.target_column else detect_target_column(df, date_column)
        
        print(f"DEBUG: Using date_column={date_column}, target_column={target_column}")
        
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)
        
        # Prepare data for chart
        chart_data = []
        for _, row in df.iterrows():
            chart_data.append({
                "date": row[date_column].strftime('%Y-%m-%d'),
                "value": float(row[target_column]) if pd.notna(row[target_column]) else None
            })
        
        return {
            "dataset_name": dataset.name,
            "date_column": date_column,
            "target_column": target_column,
            "data": chart_data,
            "total_points": len(chart_data)
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
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
        file_path = dataset.file_path
        if not file_path:
            raise HTTPException(status_code=400, detail="Dataset has no file path")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        date_column = dataset.date_column if dataset.date_column else detect_date_column(df)
        target_column = dataset.target_column if dataset.target_column else detect_target_column(df, date_column)
        
        df[date_column] = pd.to_datetime(df[date_column])
        target = df[target_column]
        
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
        
        date_stats = {
            "start_date": df[date_column].min().strftime('%Y-%m-%d'),
            "end_date": df[date_column].max().strftime('%Y-%m-%d'),
            "total_days": int((df[date_column].max() - df[date_column].min()).days)
        }
        
        return {
            "dataset_name": dataset.name,
            "statistics": stats,
            "date_range": date_stats
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
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
        file_path = dataset.file_path
        if not file_path:
            raise HTTPException(status_code=400, detail="Dataset has no file path")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        date_column = dataset.date_column if dataset.date_column else detect_date_column(df)
        target_column = dataset.target_column if dataset.target_column else detect_target_column(df, date_column)
        
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)
        
        # Day of week analysis
        df['day_of_week'] = df[date_column].dt.dayofweek
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_pattern = df.groupby('day_of_week')[target_column].mean()
        weekly_data = [{"day": day_names[i], "value": float(weekly_pattern.get(i, 0))} for i in range(7)]
        
        # Monthly analysis
        df['month'] = df[date_column].dt.month
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_pattern = df.groupby('month')[target_column].mean()
        monthly_data = [{"month": month_names[i-1], "value": float(monthly_pattern.get(i, 0))} for i in range(1, 13) if i in monthly_pattern.index]
        
        # Trend (rolling average)
        df['rolling_avg'] = df[target_column].rolling(window=7, min_periods=1).mean()
        trend_data = []
        for _, row in df.iterrows():
            trend_data.append({
                "date": row[date_column].strftime('%Y-%m-%d'),
                "value": float(row[target_column]) if pd.notna(row[target_column]) else None,
                "trend": float(row['rolling_avg']) if pd.notna(row['rolling_avg']) else None
            })
        
        return {
            "weekly_pattern": weekly_data,
            "monthly_pattern": monthly_data,
            "trend_data": trend_data
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error analyzing seasonality: {str(e)}")
