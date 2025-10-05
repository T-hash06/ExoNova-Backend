from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ErrorDetail(BaseModel):
    field: Optional[str] = Field(None, description="Specific field with error")
    constraint: Optional[str] = Field(None, description="Validation constraint violated")
    value: Optional[Any] = Field(None, description="Value that caused the error")
    received: Optional[Any] = Field(None, description="Received value")
    reason: Optional[str] = Field(None, description="Reason for the error")

    class Config:
        json_schema_extra = {
            "example": {
                "field": "pl_orbper",
                "constraint": "must be between 0 and 100",
                "value": 150.5,
                "received": 150.5
            }
        }


class ErrorInfo(BaseModel):
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[ErrorDetail] = Field(None, description="Additional error details")

    class Config:
        json_schema_extra = {
            "example": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid parameter value",
                "details": {
                    "field": "pl_orbper",
                    "value": 150.5,
                    "constraint": "must be between 0 and 100",
                    "received": 150.5
                }
            }
        }


class ErrorResponse(BaseModel):
    error: ErrorInfo
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    requestId: str

    class Config:
        json_schema_extra = {
            "example": {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid parameter value",
                    "details": {
                        "field": "pl_orbper",
                        "value": 150.5,
                        "constraint": "must be between 0 and 100",
                        "received": 150.5
                    }
                },
                "timestamp": "2025-10-06T12:34:56.789Z",
                "requestId": "req_abc123def456"
            }
        }
