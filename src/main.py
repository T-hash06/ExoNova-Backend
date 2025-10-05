from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError
import logging

from src.api.routes import tabular, live_preview, health
from src.utils.error_handlers import (
    validation_error_handler,
    custom_exception_handler,
    generic_exception_handler,
)
from src.utils.exceptions import BaseAPIException

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Exoplanet Detection API",
    version="1.0.0",
    description="API for predicting exoplanet detection probability using machine learning models",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["Content-Type", "Accept"],
    max_age=86400,
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

    # Pre-load ML models at startup
    try:
        from src.ml.tabular_classifier import get_classifier

        logger.info("Loading ML models...")
        classifier = get_classifier()
        # Ensure models are loaded without issuing a fake prediction
        classifier.ensure_loaded()

        logger.info("✓ ML models loaded successfully")
        logger.info("  - Prediction model: models/preview.pkl")
        logger.info("  - SHAP explainer: models/shap_explainer.pkl")
        logger.info("✓ Models are ready for predictions")

    except FileNotFoundError as e:
        logger.warning("=" * 60)
        logger.warning("ML models not found - falling back to mock predictions")
        logger.warning(f"Error: {e}")
        logger.warning("Place model files in the 'models/' directory:")
        logger.warning("  - models/preview.pkl (prediction model)")
        logger.warning("  - models/shap_explainer.pkl (SHAP explainer)")
        logger.warning("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error("Failed to load ML models - falling back to mock predictions")
        logger.error(f"Error: {e}")
        logger.error("=" * 60)

    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Exoplanet Detection API shutting down")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
