from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, Optional


class TabularPredictionRequest(BaseModel):
    """
    Flexible payload: accept up to 21 known fields, require at least 15 non-null.

    The backend will forward ONLY non-null values to the model. Missing expected
    features are passed as NaN, and pl_trandur is mapped to pl_trandurh.
    """

    # Planet/orbital
    pl_tranmid: Optional[float] = Field(None, description="Transit Midpoint")
    pl_orbper: Optional[float] = Field(None, ge=0, description="Orbital Period")
    pl_trandurh: Optional[float] = Field(
        None, ge=0, description="Transit Duration (hours)"
    )
    pl_trandep: Optional[float] = Field(None, ge=0, description="Transit Depth")
    pl_rade: Optional[float] = Field(None, ge=0, description="Planetary Radius")
    pl_insol: Optional[float] = Field(None, ge=0, description="Insolation Flux")
    pl_eqt: Optional[float] = Field(None, ge=0, description="Equilibrium Temperature")
    # Alternate names from legacy payload
    pl_trandur: Optional[float] = Field(
        None, ge=0, description="Transit Duration (hours) [legacy]"
    )
    pl_orbsmax: Optional[float] = Field(None, ge=0, description="Semi-major axis")
    pl_imppar: Optional[float] = Field(None, description="Impact parameter")
    pl_ratdor: Optional[float] = Field(None, ge=0, description="a/R*")
    pl_ratror: Optional[float] = Field(None, ge=0, le=1, description="Rp/R*")

    # Stellar
    st_tmag: Optional[float] = Field(None, description="TESS Magnitude")
    st_dist: Optional[float] = Field(None, ge=0, description="Distance to Star")
    st_teff: Optional[float] = Field(None, ge=0, description="Stellar Temperature")
    st_logg: Optional[float] = Field(None, description="Surface Gravity")
    st_rad: Optional[float] = Field(None, ge=0, description="Stellar Radius")
    st_pmra: Optional[float] = Field(None, description="Proper Motion RA")
    st_pmdec: Optional[float] = Field(None, description="Proper Motion Dec")
    st_mass: Optional[float] = Field(None, ge=0, description="Stellar Mass")
    st_met: Optional[float] = Field(None, description="Metallicity")

    # Photometry
    sy_gmag: Optional[float] = Field(None, ge=0, description="Gaia G mag")
    sy_rmag: Optional[float] = Field(None, ge=0, description="R mag")
    sy_imag: Optional[float] = Field(None, ge=0, description="I mag")
    sy_zmag: Optional[float] = Field(None, ge=0, description="Z mag")
    sy_jmag: Optional[float] = Field(None, ge=0, description="J mag")
    sy_hmag: Optional[float] = Field(None, ge=0, description="H mag")
    sy_kmag: Optional[float] = Field(None, ge=0, description="K mag")

    @field_validator("*", mode="before")
    @classmethod
    def numeric_or_none(cls, v):
        if v is None:
            return None
        if not isinstance(v, (int, float)):
            raise ValueError("Value must be numeric or null")
        if not (-1e308 < v < 1e308):
            raise ValueError("Value cannot be infinite")
        return float(v)

    @model_validator(mode="after")
    def ensure_minimum_features(self):
        non_null = sum(1 for v in self.model_dump().values() if v is not None)
        if non_null < 15:
            raise ValueError(
                f"At least 15 non-null fields are required, received {non_null}"
            )
        return self


class TabularPredictionResponse(BaseModel):
    predictedValue: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    attributeWeights: Dict[str, float]
