from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError
import logging
from typing import Union

from src.api.schemas.errors import ErrorResponse, ErrorInfo, ErrorDetail
from src.utils.exceptions import BaseAPIException
from src.utils.request_utils import generate_request_id

logger = logging.getLogger(__name__)


async def validation_error_handler(
    request: Request,
    exc: Union[RequestValidationError, PydanticValidationError]
) -> JSONResponse:
    request_id = generate_request_id()
    
    errors = exc.errors()
    
    if errors:
        first_error = errors[0]
        error_type = first_error.get("type", "validation_error")
        error_msg = first_error.get("msg", "Validation error")
        error_loc = first_error.get("loc", [])
        
        field_name = None
        if len(error_loc) >= 2:
            field_name = str(error_loc[-1])
        elif len(error_loc) == 1:
            field_name = str(error_loc[0])
        
        constraint = None
        input_value = first_error.get("input")
        
        if "missing" in error_type:
            error_code = "MISSING_FIELD"
            message = f"Required field missing"
            constraint = "field is required"
        elif "type" in error_type or "string" in error_type or "float" in error_type or "int" in error_type:
            error_code = "INVALID_TYPE"
            message = "Invalid parameter type"
            constraint = error_msg
        elif "less_than" in error_type or "greater_than" in error_type or "range" in error_type:
            error_code = "OUT_OF_RANGE"
            message = "Invalid parameter value"
            constraint = error_msg
        else:
            error_code = "VALIDATION_ERROR"
            message = "Invalid parameter value"
            constraint = error_msg
        
        error_detail = ErrorDetail(
            field=field_name,
            constraint=constraint,
            value=input_value,
            received=input_value
        )
    else:
        error_code = "VALIDATION_ERROR"
        message = "Validation error"
        error_detail = None
    
    error_response = ErrorResponse(
        error=ErrorInfo(
            code=error_code,
            message=message,
            details=error_detail
        ),
        requestId=request_id
    )
    
    logger.warning(
        f"Validation error: {error_code} - {message}",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "field": field_name
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_response.model_dump()
    )


async def custom_exception_handler(
    request: Request,
    exc: BaseAPIException
) -> JSONResponse:
    request_id = generate_request_id()
    
    error_detail = None
    if exc.field or exc.constraint or exc.value or exc.reason:
        error_detail = ErrorDetail(
            field=exc.field,
            constraint=exc.constraint,
            value=exc.value,
            received=exc.value,
            reason=exc.reason
        )
    
    error_response = ErrorResponse(
        error=ErrorInfo(
            code=exc.error_code,
            message=exc.message,
            details=error_detail
        ),
        requestId=request_id
    )
    
    log_level = logging.ERROR if exc.status_code >= 500 else logging.WARNING
    logger.log(
        log_level,
        f"{exc.error_code}: {exc.message}",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "status_code": exc.status_code
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    request_id = generate_request_id()
    
    error_response = ErrorResponse(
        error=ErrorInfo(
            code="INTERNAL_ERROR",
            message="An unexpected error occurred",
            details=ErrorDetail(
                reason="Internal server error"
            )
        ),
        requestId=request_id
    )
    
    logger.error(
        f"Unexpected error: {type(exc).__name__}: {str(exc)}",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "exception_type": type(exc).__name__
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump()
    )
