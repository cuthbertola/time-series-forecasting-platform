from fastapi import APIRouter
from app.api.v1.endpoints import datasets, training, forecast, visualization, export, batch

api_router = APIRouter()

api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(training.router, prefix="/training", tags=["training"])
api_router.include_router(forecast.router, prefix="/forecast", tags=["forecast"])
api_router.include_router(visualization.router, prefix="/visualization", tags=["visualization"])
api_router.include_router(export.router, prefix="/export", tags=["export"])
api_router.include_router(batch.router, prefix="/batch", tags=["batch"])
