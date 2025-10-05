from pydantic import BaseModel, Field, field_validator
from typing import Dict


class TabularPredictionRequest(BaseModel):
    pl_orbper: float = Field(..., ge=0, le=100, description="Orbital Period (days)")
    pl_orbsmax: float = Field(..., ge=0, le=0.5, description="Orbit Semi-Major Axis (AU)")
    pl_eqt: float = Field(..., ge=0, le=4000, description="Equilibrium Temperature (K)")
    pl_insol: float = Field(..., ge=0, le=7000, description="Insolation Flux (Earth flux)")
    pl_imppar: float = Field(..., ge=-1, le=2, description="Impact Parameter")
    pl_trandep: float = Field(..., ge=0, le=6, description="Transit Depth (%)")
    pl_trandur: float = Field(..., ge=0, le=15, description="Transit Duration (hours)")
    pl_ratdor: float = Field(..., ge=0, le=100, description="Distance-to-Radius Ratio")
    pl_ratror: float = Field(..., ge=0, le=1, description="Planet-Star Radius Ratio")
    st_teff: float = Field(..., ge=3000, le=8000, description="Stellar Temperature (K)")
    st_rad: float = Field(..., ge=0, le=3, description="Stellar Radius (R☉)")
    st_mass: float = Field(..., ge=0, le=2, description="Stellar Mass (M☉)")
    st_met: float = Field(..., ge=-1, le=0.5, description="Stellar Metallicity [Fe/H] (dex)")
    st_logg: float = Field(..., ge=3, le=5.5, description="Surface Gravity (log(cm/s²))")
    sy_gmag: float = Field(..., ge=10, le=20, description="Gaia G-band magnitude")
    sy_rmag: float = Field(..., ge=10, le=19, description="r-band magnitude")
    sy_imag: float = Field(..., ge=10, le=18, description="i-band magnitude")
    sy_zmag: float = Field(..., ge=10, le=18, description="z-band magnitude")
    sy_jmag: float = Field(..., ge=6, le=17, description="2MASS J-band magnitude")
    sy_hmag: float = Field(..., ge=6, le=17, description="2MASS H-band magnitude")
    sy_kmag: float = Field(..., ge=6, le=17, description="2MASS K-band magnitude")

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
                "pl_orbper": 10.5,
                "pl_orbsmax": 0.06,
                "pl_eqt": 1000,
                "pl_insol": 500,
                "pl_imppar": 0.55,
                "pl_trandep": 0.2,
                "pl_trandur": 4.0,
                "pl_ratdor": 15.0,
                "pl_ratror": 0.1,
                "st_teff": 5700,
                "st_rad": 1.0,
                "st_mass": 0.96,
                "st_met": -0.05,
                "st_logg": 4.45,
                "sy_gmag": 15.0,
                "sy_rmag": 14.4,
                "sy_imag": 14.2,
                "sy_zmag": 14.2,
                "sy_jmag": 12.8,
                "sy_hmag": 12.5,
                "sy_kmag": 12.4
            }
        }


class TabularPredictionResponse(BaseModel):
    predictedValue: float = Field(..., ge=0.0, le=1.0, description="Probability [0.0 - 1.0]")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level [0.0 - 1.0]")
    attributeWeights: Dict[str, float] = Field(
        ..., 
        description="Feature importance weights for 15-20 most important features"
    )

    @field_validator('attributeWeights')
    @classmethod
    def validate_weights(cls, v):
        if not v:
            raise ValueError("attributeWeights cannot be empty")
        for key, value in v.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Weight for {key} must be numeric")
            if not (-1.0 <= value <= 1.0):
                raise ValueError(f"Weight for {key} must be between -1.0 and 1.0")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "predictedValue": 0.7823,
                "confidence": 0.8945,
                "attributeWeights": {
                    "pl_trandep": 0.234,
                    "pl_orbper": -0.156,
                    "pl_trandur": 0.189,
                    "st_teff": 0.145,
                    "sy_gmag": -0.178,
                    "pl_ratror": 0.167,
                    "st_rad": 0.123,
                    "pl_insol": 0.112,
                    "st_mass": 0.098,
                    "sy_jmag": -0.134,
                    "sy_hmag": -0.128,
                    "sy_kmag": -0.125,
                    "pl_eqt": 0.089,
                    "st_met": -0.076,
                    "pl_ratdor": 0.067,
                    "st_logg": -0.054,
                    "pl_imppar": -0.043,
                    "sy_rmag": -0.123
                }
            }
        }
