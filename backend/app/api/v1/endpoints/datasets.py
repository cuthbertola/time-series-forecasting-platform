from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List
import pandas as pd
import os
import uuid
from datetime import datetime

from app.core.database import get_db
from app.core.config import settings
from app.models import Dataset, DatasetStatus
from app.schemas import DatasetCreate, DatasetResponse, DatasetListResponse

router = APIRouter()

# Create uploads directory
UPLOAD_DIR = "data/raw"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    date_column: Optional[str] = Form(None),
    target_column: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """Upload a new time series dataset."""
    # Validate file extension
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Allowed types: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # Generate unique filename
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save file
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )
    
    # Read and analyze the dataset
    try:
        if file_ext == "csv":
            df = pd.read_csv(file_path)
        elif file_ext in ["xlsx", "xls"]:
            df = pd.read_excel(file_path)
        elif file_ext == "json":
            df = pd.read_json(file_path)
        else:
            df = pd.read_csv(file_path)
        
        row_count = len(df)
        column_count = len(df.columns)
        feature_columns = list(df.columns)
        
        # Auto-detect date column if not provided
        if not date_column:
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col])
                        date_column = col
                        break
                    except:
                        continue
        
        # Detect frequency and date range
        frequency = None
        start_date = None
        end_date = None
        
        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            start_date = df[date_column].min()
            end_date = df[date_column].max()
            
            # Detect frequency
            diffs = df[date_column].diff().dropna()
            if len(diffs) > 0:
                median_diff = diffs.median()
                if median_diff <= pd.Timedelta(days=1):
                    frequency = "daily"
                elif median_diff <= pd.Timedelta(days=7):
                    frequency = "weekly"
                elif median_diff <= pd.Timedelta(days=31):
                    frequency = "monthly"
                else:
                    frequency = "yearly"
        
        dataset_status = DatasetStatus.READY
        error_message = None
        
    except Exception as e:
        row_count = None
        column_count = None
        feature_columns = None
        frequency = None
        start_date = None
        end_date = None
        dataset_status = DatasetStatus.FAILED
        error_message = str(e)
    
    # Create database record
    dataset = Dataset(
        name=name,
        description=description,
        filename=file.filename,
        file_path=file_path,
        row_count=row_count,
        column_count=column_count,
        date_column=date_column,
        target_column=target_column,
        feature_columns=feature_columns,
        frequency=frequency,
        start_date=start_date,
        end_date=end_date,
        status=dataset_status,
        error_message=error_message
    )
    
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    
    return dataset


@router.get("/", response_model=DatasetListResponse)
async def list_datasets(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """List all datasets."""
    result = await db.execute(
        select(Dataset).offset(skip).limit(limit).order_by(Dataset.created_at.desc())
    )
    datasets = result.scalars().all()
    
    total_result = await db.execute(select(Dataset))
    total = len(total_result.scalars().all())
    
    return DatasetListResponse(datasets=datasets, total=total)


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific dataset by ID."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    return dataset


@router.get("/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: int,
    rows: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """Preview the first N rows of a dataset."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    try:
        file_ext = dataset.filename.split(".")[-1].lower()
        if file_ext == "csv":
            df = pd.read_csv(dataset.file_path, nrows=rows)
        elif file_ext in ["xlsx", "xls"]:
            df = pd.read_excel(dataset.file_path, nrows=rows)
        else:
            df = pd.read_csv(dataset.file_path, nrows=rows)
        
        return {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "data": df.to_dict(orient="records"),
            "total_rows": dataset.row_count
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read dataset: {str(e)}"
        )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete a dataset."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Delete file
    if os.path.exists(dataset.file_path):
        os.remove(dataset.file_path)
    
    await db.delete(dataset)
    await db.commit()
