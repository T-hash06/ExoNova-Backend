from fastapi import APIRouter
from pydantic import BaseModel
import logging

from ...utils.request_utils import get_timestamp

router = APIRouter(tags=["health"])

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    timestamp = get_timestamp()
    
    logger.info(f"Health check request at {timestamp}")
    
    return HealthResponse(
        status="healthy",
        model_loaded=False,
        timestamp=timestamp
    )
