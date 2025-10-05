from typing import Generic, TypeVar
from pydantic import BaseModel, Field
from datetime import datetime

T = TypeVar('T')


class BaseResponse(BaseModel, Generic[T]):
    data: T
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    requestId: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": {},
                "timestamp": "2025-10-06T12:34:56.789Z",
                "requestId": "req_abc123def456"
            }
        }
