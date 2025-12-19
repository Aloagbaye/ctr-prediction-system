"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request model for single CTR prediction"""
    user_id: str = Field(..., description="User identifier")
    ad_id: str = Field(..., description="Advertisement identifier")
    device: str = Field(..., description="Device type: mobile, desktop, or tablet")
    placement: str = Field(..., description="Ad placement: header, sidebar, footer, in_content, or popup")
    timestamp: Optional[datetime] = Field(None, description="Timestamp of impression (optional)")
    hour: Optional[int] = Field(None, ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: Optional[int] = Field(None, ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "ad_id": "ad_456",
                "device": "mobile",
                "placement": "header",
                "timestamp": "2024-01-15T14:30:00",
                "hour": 14,
                "day_of_week": 0
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch CTR predictions"""
    impressions: List[PredictionRequest] = Field(..., description="List of impression requests")
    
    class Config:
        json_schema_extra = {
            "example": {
                "impressions": [
                    {
                        "user_id": "user_123",
                        "ad_id": "ad_456",
                        "device": "mobile",
                        "placement": "header"
                    },
                    {
                        "user_id": "user_456",
                        "ad_id": "ad_789",
                        "device": "desktop",
                        "placement": "sidebar"
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Response model for CTR prediction"""
    predicted_ctr: float = Field(..., ge=0.0, le=1.0, description="Predicted click-through rate")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of click")
    model_name: str = Field(..., description="Name of the model used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_ctr": 0.0234,
                "probability": 0.0234,
                "model_name": "xgboost"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total: int = Field(..., description="Total number of predictions")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type")
    features: List[str] = Field(..., description="List of feature names")
    num_features: int = Field(..., description="Number of features")
    loaded: bool = Field(..., description="Whether model is loaded")
    load_time: Optional[float] = Field(None, description="Model load time in seconds")

