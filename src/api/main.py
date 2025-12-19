"""
FastAPI Application for CTR Prediction Service

Main API endpoints:
- /predict_ctr: Single prediction
- /batch_predict: Batch predictions
- /health: Health check
- /model_info: Model information
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import logging
from typing import Optional
import time

from .models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from .predictor import ModelPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CTR Prediction API",
    description="API for predicting Click-Through Rate of ad impressions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model predictor (loads models on startup)
predictor = ModelPredictor(models_dir="models", default_model="xgboost")

# Load models on startup
@app.on_event("startup")
async def startup_event():
    """Load models when API starts"""
    logger.info("Starting CTR Prediction API...")
    logger.info("Loading models...")
    
    results = predictor.load_all_models()
    loaded_models = [name for name, success in results.items() if success]
    
    if loaded_models:
        logger.info(f"✓ Loaded models: {', '.join(loaded_models)}")
        logger.info(f"✓ Default model: {predictor.default_model}")
    else:
        logger.warning("⚠️  No models loaded! API will not work properly.")
        logger.warning("Please train models first: python scripts/train_models.py")
    
    logger.info("API startup complete!")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "CTR Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint
    
    Returns API status and timestamp
    """
    return HealthResponse(
        status="healthy" if predictor.models else "degraded",
        timestamp=datetime.now(),
        version="1.0.0"
    )


@app.get("/model_info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info(model_name: Optional[str] = None):
    """
    Get information about the prediction model
    
    Args:
        model_name: Name of model (logistic, xgboost, lightgbm). Uses default if not specified.
    
    Returns:
        Model information including features and status
    """
    try:
        info = predictor.get_model_info(model_name)
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving model info: {str(e)}"
        )


@app.post("/predict_ctr", response_model=PredictionResponse, tags=["Prediction"])
async def predict_ctr(request: PredictionRequest, model_name: Optional[str] = None):
    """
    Predict CTR for a single ad impression
    
    Args:
        request: Prediction request with impression details
        model_name: Model to use (logistic, xgboost, lightgbm). Uses default if not specified.
    
    Returns:
        Predicted CTR and probability
    
    Example:
        ```json
        {
            "user_id": "user_123",
            "ad_id": "ad_456",
            "device": "mobile",
            "placement": "header",
            "hour": 14,
            "day_of_week": 0
        }
        ```
    """
    if not predictor.models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No models loaded. Please train models first."
        )
    
    try:
        start_time = time.time()
        
        # Convert request to dict
        request_dict = request.dict()
        
        # Make prediction
        result = predictor.predict(request_dict, model_name)
        
        # Log prediction time
        prediction_time = (time.time() - start_time) * 1000  # Convert to ms
        logger.info(f"Prediction completed in {prediction_time:.2f}ms")
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest, model_name: Optional[str] = None):
    """
    Predict CTR for multiple ad impressions (batch)
    
    Args:
        request: Batch prediction request with list of impressions
        model_name: Model to use. Uses default if not specified.
    
    Returns:
        List of predictions with processing time
    
    Example:
        ```json
        {
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
        ```
    """
    if not predictor.models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No models loaded. Please train models first."
        )
    
    if len(request.impressions) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty impressions list"
        )
    
    if len(request.impressions) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size too large. Maximum 1000 impressions per batch."
        )
    
    try:
        # Convert requests to dicts
        request_dicts = [imp.dict() for imp in request.impressions]
        
        # Make batch predictions
        result = predictor.predict_batch(request_dicts, model_name)
        
        # Convert predictions to response format
        predictions = [
            PredictionResponse(**pred) for pred in result['predictions']
        ]
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=result['total'],
            processing_time_ms=result['processing_time_ms']
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

