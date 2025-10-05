from fastapi import APIRouter, HTTPException, status
from typing import Dict
import logging

from ..schemas.tabular import TabularPredictionRequest, TabularPredictionResponse
from ..schemas.base import BaseResponse
from ...ml.predictor import predict_tabular_mock
from ...utils.request_utils import generate_request_id, get_timestamp
from ...utils.exceptions import ModelError

router = APIRouter(prefix="/predict", tags=["predictions"])

logger = logging.getLogger(__name__)


@router.post("/tabular", response_model=BaseResponse[TabularPredictionResponse])
async def predict_tabular(request: TabularPredictionRequest) -> BaseResponse[TabularPredictionResponse]:
    request_id = generate_request_id()
    timestamp = get_timestamp()
    
    logger.info(f"Tabular prediction request {request_id}: {request.model_dump()}")
    
    try:
        input_data = request.model_dump()
        
        predicted_value, confidence, attribute_weights = predict_tabular_mock(input_data)
        
        response_data = TabularPredictionResponse(
            predictedValue=predicted_value,
            confidence=confidence,
            attributeWeights=attribute_weights
        )
        
        logger.info(f"Tabular prediction response {request_id}: predictedValue={predicted_value:.4f}, confidence={confidence:.4f}")
        
        return BaseResponse[TabularPredictionResponse](
            data=response_data,
            timestamp=timestamp,
            requestId=request_id
        )
        
    except Exception as e:
        logger.error(f"Error in tabular prediction {request_id}: {str(e)}")
        raise ModelError(f"Prediction failed: {str(e)}")
