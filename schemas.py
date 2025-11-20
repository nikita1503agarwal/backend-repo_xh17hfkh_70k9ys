"""
Database Schemas for Size Recommendation App

Each Pydantic model represents a collection in MongoDB. The collection name is the lowercase of the class name.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class User(BaseModel):
    email: str = Field(..., description="Email address")
    consent: bool = Field(False, description="User consent for data processing")
    country: Optional[str] = Field(None, description="Country for privacy jurisdiction checks")


class BodyProfile(BaseModel):
    user_id: Optional[str] = Field(None, description="Reference to user")
    height_cm: Optional[float] = Field(None, ge=50, le=250)
    weight_kg: Optional[float] = Field(None, ge=20, le=300)
    chest_cm: Optional[float] = Field(None, ge=40, le=200)
    waist_cm: Optional[float] = Field(None, ge=30, le=180)
    hips_cm: Optional[float] = Field(None, ge=40, le=200)
    inseam_cm: Optional[float] = Field(None, ge=30, le=120)
    shoulder_cm: Optional[float] = Field(None, ge=20, le=80)
    computed_from_image: bool = Field(False)


class Garment(BaseModel):
    brand: str
    category: Literal[
        "tshirt",
        "shirt",
        "hoodie",
        "jacket",
        "pants",
        "jeans",
        "dress",
        "skirt",
    ]
    size_chart: dict = Field(
        ..., description="Map size label -> required body measurements (cm)"
    )


class Recommendation(BaseModel):
    user_id: Optional[str] = None
    brand: str
    category: str
    suggested_size: str
    confidence: float = Field(..., ge=0, le=1)
    details: dict
