from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.core.config import settings
from app.core.database import async_engine, Base
from app.api.v1.api import api_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Time Series Forecasting Platform...")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created successfully")
    yield
    # Shutdown
    logger.info("Shutting down Time Series Forecasting Platform...")
    await async_engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    description="""
## Time Series Forecasting Platform with AutoML

A comprehensive platform for time series forecasting featuring:

* **AutoML**: Automatic model selection and hyperparameter tuning
* **Multiple Algorithms**: Prophet, ARIMA, XGBoost, LightGBM
* **Feature Engineering**: Automated lag, rolling, and calendar features
* **Backtesting**: Walk-forward validation for robust evaluation
* **Confidence Intervals**: Uncertainty quantification in predictions
* **Model Comparison**: Compare multiple models side by side

### Key Features:
- Upload time series data (CSV, Excel, JSON)
- Automatic data preprocessing and feature engineering
- Train and compare 5+ forecasting algorithms
- Generate forecasts with confidence intervals
- Track experiments with MLflow

### Target Metrics:
- MAPE: <8%
- Training Time: <5 minutes
- Forecast Horizon: 30-90 days
    """,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS middleware - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return {"detail": str(exc)}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from datetime import datetime
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "api": f"{settings.API_V1_PREFIX}"
    }


# Include API router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
