from fastapi import APIRouter, HTTPException, status
import logging

from ..schemas.live_preview import LivePreviewRequest, LivePreviewResponse
from ..schemas.base import BaseResponse
from ...ml.predictor import predict_live_preview_mock
from ...utils.request_utils import generate_request_id, get_timestamp
from ...utils.exceptions import ModelError

router = APIRouter(prefix="/predict", tags=["predictions"])

logger = logging.getLogger(__name__)


@router.post("/live-preview", response_model=BaseResponse[LivePreviewResponse])
async def predict_live_preview(request: LivePreviewRequest) -> BaseResponse[LivePreviewResponse]:
    request_id = generate_request_id()
    timestamp = get_timestamp()
    
    logger.info(f"Live preview prediction request {request_id}: {request.model_dump()}")
    
    try:
        input_data = request.model_dump()
        
        probability = predict_live_preview_mock(input_data)
        
        response_data = LivePreviewResponse(
            probability=probability
        )
        
        logger.info(f"Live preview prediction response {request_id}: probability={probability:.2f}%")
        
        return BaseResponse[LivePreviewResponse](
            data=response_data,
            timestamp=timestamp,
            requestId=request_id
        )
        
    except Exception as e:
        logger.error(f"Error in live preview prediction {request_id}: {str(e)}")
        raise ModelError(f"Prediction failed: {str(e)}")
