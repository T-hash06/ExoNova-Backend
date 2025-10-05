from pydantic import BaseModel, Field, field_validator


class LivePreviewRequest(BaseModel):
    plTranmid: float = Field(..., ge=0.0, le=1.0, description="Transit Midpoint - Normalized [0.0 - 1.0]")
    stPmdec: float = Field(..., ge=0.0, le=1.0, description="Stellar Proper Motion (Dec) - Normalized [0.0 - 1.0]")
    stTmag: float = Field(..., ge=0.0, le=1.0, description="TESS Magnitude - Normalized [0.0 - 1.0]")
    stRade: float = Field(..., ge=0.0, le=1.0, description="Stellar Radius - Normalized [0.0 - 1.0]")
    stDist: float = Field(..., ge=0.0, le=1.0, description="Distance to Star - Normalized [0.0 - 1.0]")
    plRade: float = Field(..., ge=0.0, le=1.0, description="Planetary Radius - Normalized [0.0 - 1.0]")

    @field_validator('*', mode='before')
    @classmethod
    def check_numeric_value(cls, v):
        if v is None:
            raise ValueError("Value cannot be null")
        if not isinstance(v, (int, float)):
            raise ValueError("Value must be numeric")
        if not (-1e308 < v < 1e308):
            raise ValueError("Value cannot be infinite")
        return float(v)

    class Config:
        json_schema_extra = {
            "example": {
                "plTranmid": 0.5,
                "stPmdec": 0.5,
                "stTmag": 0.5,
                "stRade": 0.5,
                "stDist": 0.5,
                "plRade": 0.5
            }
        }


class LivePreviewResponse(BaseModel):
    probability: float = Field(..., ge=0, le=100, description="Detection probability as percentage [0 - 100]")

    @field_validator('probability')
    @classmethod
    def validate_probability(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError("Probability must be numeric")
        if not (0 <= v <= 100):
            raise ValueError("Probability must be between 0 and 100")
        return float(v)

    class Config:
        json_schema_extra = {
            "example": {
                "probability": 67.5
            }
        }
