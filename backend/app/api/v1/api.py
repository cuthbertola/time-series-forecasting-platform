from fastapi import APIRouter
from app.api.v1.endpoints import datasets, training, forecast, health, visualization, export, batch, explanations, backtesting

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(training.router, prefix="/training", tags=["training"])
api_router.include_router(forecast.router, prefix="/forecast", tags=["forecast"])
api_router.include_router(visualization.router, prefix="/visualization", tags=["visualization"])
api_router.include_router(export.router, prefix="/export", tags=["export"])
api_router.include_router(batch.router, prefix="/batch", tags=["batch"])
api_router.include_router(explanations.router, prefix="/explanations", tags=["explanations"])
api_router.include_router(backtesting.router, prefix="/backtesting", tags=["backtesting"])
