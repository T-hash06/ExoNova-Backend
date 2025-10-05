from typing import Optional, Dict, Any


class BaseAPIException(Exception):
    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        field: Optional[str] = None,
        constraint: Optional[str] = None,
        value: Optional[Any] = None,
        reason: Optional[str] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.field = field
        self.constraint = constraint
        self.value = value
        self.reason = reason
        super().__init__(self.message)


class ValidationError(BaseAPIException):
    def __init__(
        self,
        message: str = "Validation error",
        field: Optional[str] = None,
        constraint: Optional[str] = None,
        value: Optional[Any] = None
    ):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            field=field,
            constraint=constraint,
            value=value
        )


class MissingFieldError(BaseAPIException):
    def __init__(
        self,
        field: str,
        message: Optional[str] = None
    ):
        if message is None:
            message = f"Required field missing: {field}"
        super().__init__(
            message=message,
            error_code="MISSING_FIELD",
            status_code=400,
            field=field,
            reason="Field is required"
        )


class InvalidTypeError(BaseAPIException):
    def __init__(
        self,
        field: str,
        expected_type: str,
        received_value: Any,
        message: Optional[str] = None
    ):
        if message is None:
            message = f"Field '{field}' has invalid type. Expected {expected_type}"
        super().__init__(
            message=message,
            error_code="INVALID_TYPE",
            status_code=400,
            field=field,
            value=received_value,
            constraint=f"must be of type {expected_type}"
        )


class OutOfRangeError(BaseAPIException):
    def __init__(
        self,
        field: str,
        value: Any,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        message: Optional[str] = None
    ):
        if message is None:
            if min_value is not None and max_value is not None:
                constraint = f"must be between {min_value} and {max_value}"
            elif min_value is not None:
                constraint = f"must be at least {min_value}"
            elif max_value is not None:
                constraint = f"must be at most {max_value}"
            else:
                constraint = "value out of valid range"
            message = f"Value for '{field}' is out of range"
        else:
            constraint = "value out of valid range"
        
        super().__init__(
            message=message,
            error_code="OUT_OF_RANGE",
            status_code=400,
            field=field,
            constraint=constraint,
            value=value
        )


class ModelError(BaseAPIException):
    def __init__(
        self,
        message: str = "Model prediction failed",
        reason: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="MODEL_ERROR",
            status_code=500,
            reason=reason or "Model inference failed"
        )


class ModelNotLoadedError(BaseAPIException):
    def __init__(
        self,
        message: str = "Model not initialized",
        reason: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="MODEL_NOT_LOADED",
            status_code=503,
            reason=reason or "Model has not been loaded"
        )


class InternalError(BaseAPIException):
    def __init__(
        self,
        message: str = "Internal server error",
        reason: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="INTERNAL_ERROR",
            status_code=500,
            reason=reason or "An unexpected error occurred"
        )
