from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError
import logging

from src.api.routes import tabular, live_preview, health
from src.utils.error_handlers import (
    validation_error_handler,
    custom_exception_handler,
    generic_exception_handler
)
from src.utils.exceptions import BaseAPIException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Exoplanet Detection API",
    version="1.0.0",
    description="API for predicting exoplanet detection probability using machine learning models"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["Content-Type", "Accept"],
    max_age=86400
)

app.include_router(tabular.router, prefix="/api/v1")
app.include_router(live_preview.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")

app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(PydanticValidationError, validation_error_handler)
app.add_exception_handler(BaseAPIException, custom_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Exoplanet Detection API Starting Up")
    logger.info("=" * 60)
    logger.info("Application started - using mock predictions")
    logger.info("Mock predictor functions are located in: src/ml/predictor.py")
    logger.info("Replace mock functions with actual model inference when ready")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Exoplanet Detection API shutting down")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
