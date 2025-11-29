from fastapi import APIRouter
from app.api.v1.endpoints import datasets, training, forecast

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    datasets.router,
    prefix="/datasets",
    tags=["datasets"]
)

api_router.include_router(
    training.router,
    prefix="/training",
    tags=["training"]
)

api_router.include_router(
    forecast.router,
    prefix="/forecast",
    tags=["forecast"]
)
