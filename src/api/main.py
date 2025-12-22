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
from typing import Optional, List
import time
import os

from .models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ABTestCreateRequest,
    ABTestResponse,
    ABTestStatsResponse,
    ABTestPredictionResponse
)
from .predictor import ModelPredictor
from .bigquery_logger import BigQueryLogger
from .ab_test_manager import ABTestManager
from .monitoring import MonitoringManager

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
# Check for GCS bucket or use local models directory
models_dir = os.getenv("MODELS_DIR", "models")
predictor = ModelPredictor(models_dir=models_dir, default_model="xgboost")

# Initialize BigQuery logger (optional)
bigquery_logger = BigQueryLogger()

# Initialize A/B Test Manager with MLflow integration
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
ab_test_manager = ABTestManager(
    storage_path="data/ab_tests",
    mlflow_tracking_uri=mlflow_tracking_uri,
    mlflow_experiment_name="ctr_prediction_ab_tests",
    enable_mlflow=True
)

# Initialize Monitoring Manager with MLflow
monitoring_manager = MonitoringManager(
    mlflow_tracking_uri=mlflow_tracking_uri,
    experiment_name="ctr_prediction_production",
    enable_mlflow=True
)

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
async def predict_ctr(
    request: PredictionRequest,
    model_name: Optional[str] = None,
    ab_test_id: Optional[str] = None
):
    """
    Predict CTR for a single ad impression
    
    Args:
        request: Prediction request with impression details
        model_name: Model to use (logistic, xgboost, lightgbm). Uses default if not specified.
        ab_test_id: A/B test identifier. If provided, traffic will be split between models.
    
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
        
        # A/B testing: route to appropriate model if test_id is provided
        test_variant = None
        if ab_test_id:
            try:
                selected_model, variant = ab_test_manager.get_model_for_request(
                    ab_test_id, request.user_id
                )
                model_name = selected_model
                test_variant = variant
            except ValueError as e:
                logger.warning(f"A/B test routing failed: {e}, using default model")
        
        # Make prediction
        result = predictor.predict(request_dict, model_name)
        
        # Log prediction time
        prediction_time = (time.time() - start_time) * 1000  # Convert to ms
        logger.info(f"Prediction completed in {prediction_time:.2f}ms using model: {result['model_name']}")
        
        # Record metrics for A/B testing
        if ab_test_id and test_variant:
            try:
                ab_test_manager.record_prediction(
                    ab_test_id,
                    result['model_name'],
                    prediction_time,
                    error=False
                )
            except Exception as e:
                logger.warning(f"Failed to record A/B test metrics: {e}")
        
        # Record metrics for monitoring
        try:
            monitoring_manager.record_prediction(
                model_name=result['model_name'],
                latency_ms=prediction_time,
                error=False,
                prediction_value=result.get('predicted_ctr'),
                request_id=str(hash(str(request_dict)))
            )
        except Exception as e:
            logger.warning(f"Failed to record monitoring metrics: {e}")
        
        # Log to BigQuery (async, don't block response)
        try:
            bigquery_logger.log_prediction(request_dict, result)
        except Exception as e:
            logger.warning(f"Failed to log to BigQuery: {e}")
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        # Record error in A/B test metrics
        if ab_test_id:
            try:
                ab_test_manager.record_prediction(
                    ab_test_id,
                    model_name or predictor.default_model,
                    (time.time() - start_time) * 1000,
                    error=True
                )
            except:
                pass
        
        # Record error in monitoring
        try:
            monitoring_manager.record_prediction(
                model_name=model_name or predictor.default_model,
                latency_ms=(time.time() - start_time) * 1000,
                error=True
            )
        except:
            pass
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Record error in A/B test metrics
        if ab_test_id:
            try:
                ab_test_manager.record_prediction(
                    ab_test_id,
                    model_name or predictor.default_model,
                    (time.time() - start_time) * 1000,
                    error=True
                )
            except:
                pass
        
        # Record error in monitoring
        try:
            monitoring_manager.record_prediction(
                model_name=model_name or predictor.default_model,
                latency_ms=(time.time() - start_time) * 1000,
                error=True
            )
        except:
            pass
        
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
        
        # Log to BigQuery (async, don't block response)
        try:
            bigquery_logger.log_batch_predictions(request_dicts, result['predictions'])
        except Exception as e:
            logger.warning(f"Failed to log batch to BigQuery: {e}")
        
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


# ============================================================================
# A/B Testing Endpoints
# ============================================================================

@app.post("/ab_test/create", response_model=ABTestResponse, tags=["A/B Testing"])
async def create_ab_test(request: ABTestCreateRequest):
    """
    Create a new A/B test
    
    Args:
        request: A/B test configuration
    
    Returns:
        Created A/B test configuration
    
    Example:
        ```json
        {
            "test_id": "xgboost_vs_lightgbm",
            "model_a": "xgboost",
            "model_b": "lightgbm",
            "traffic_split": 0.5,
            "duration_hours": 24,
            "description": "Compare XGBoost vs LightGBM performance"
        }
        ```
    """
    try:
        # Validate models exist
        if request.model_a not in predictor.models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model {request.model_a} not loaded"
            )
        if request.model_b not in predictor.models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model {request.model_b} not loaded"
            )
        
        config = ab_test_manager.create_test(
            test_id=request.test_id,
            model_a=request.model_a,
            model_b=request.model_b,
            traffic_split=request.traffic_split,
            duration_hours=request.duration_hours,
            description=request.description
        )
        
        return ABTestResponse(**config.to_dict())
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating A/B test: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create A/B test: {str(e)}"
        )


@app.get("/ab_test/list", response_model=List[ABTestResponse], tags=["A/B Testing"])
async def list_ab_tests():
    """
    List all active A/B tests
    
    Returns:
        List of A/B test configurations
    """
    try:
        tests = ab_test_manager.list_tests()
        return [ABTestResponse(**test) for test in tests]
    except Exception as e:
        logger.error(f"Error listing A/B tests: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list A/B tests: {str(e)}"
        )


@app.get("/ab_test/stats/{test_id}", response_model=ABTestStatsResponse, tags=["A/B Testing"])
async def get_ab_test_stats(test_id: str):
    """
    Get statistics for an A/B test
    
    Args:
        test_id: A/B test identifier
    
    Returns:
        A/B test statistics including performance comparison
    """
    try:
        stats = ab_test_manager.get_test_stats(test_id)
        return ABTestStatsResponse(**stats)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting A/B test stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get A/B test stats: {str(e)}"
        )


@app.post("/ab_test/stop/{test_id}", tags=["A/B Testing"])
async def stop_ab_test(test_id: str):
    """
    Stop an A/B test (disables traffic splitting)
    
    Args:
        test_id: A/B test identifier
    
    Returns:
        Success message
    """
    try:
        ab_test_manager.stop_test(test_id)
        return {"message": f"A/B test {test_id} stopped successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error stopping A/B test: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop A/B test: {str(e)}"
        )


@app.delete("/ab_test/{test_id}", tags=["A/B Testing"])
async def delete_ab_test(test_id: str):
    """
    Delete an A/B test and its metrics
    
    Args:
        test_id: A/B test identifier
    
    Returns:
        Success message
    """
    try:
        ab_test_manager.delete_test(test_id)
        return {"message": f"A/B test {test_id} deleted successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error deleting A/B test: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete A/B test: {str(e)}"
        )


# ============================================================================
# Monitoring Endpoints
# ============================================================================

@app.get("/monitoring/health", tags=["Monitoring"])
async def get_monitoring_health():
    """
    Get overall system health status
    
    Returns:
        Health status with alerts and model statistics
    """
    try:
        health = monitoring_manager.get_health_status()
        return health
    except Exception as e:
        logger.error(f"Error getting monitoring health: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get health status: {str(e)}"
        )


@app.get("/monitoring/stats", tags=["Monitoring"])
async def get_monitoring_stats(model_name: Optional[str] = None):
    """
    Get monitoring statistics
    
    Args:
        model_name: Optional model name filter. Returns all models if not specified.
    
    Returns:
        Statistics for model(s)
    """
    try:
        if model_name:
            stats = monitoring_manager.get_model_stats(model_name)
            if stats is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model {model_name} not found in monitoring"
                )
            return stats.to_dict()
        else:
            return monitoring_manager.get_all_stats()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting monitoring stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get monitoring stats: {str(e)}"
        )


@app.get("/monitoring/drift/{model_name}", tags=["Monitoring"])
async def check_model_drift(model_name: str):
    """
    Check for model performance drift
    
    Args:
        model_name: Model name to check
    
    Returns:
        Drift detection results
    """
    try:
        drift_result = monitoring_manager.check_drift(model_name)
        return drift_result
    except Exception as e:
        logger.error(f"Error checking drift: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check drift: {str(e)}"
        )


@app.post("/monitoring/baseline", tags=["Monitoring"])
async def set_baseline(request: dict):
    """
    Set baseline statistics for drift detection
    
    Request body:
        {
            "model_name": "xgboost",
            "mean_prediction": 0.0234,
            "std_prediction": 0.0123,
            "mean_latency_ms": 45.6,
            "std_latency_ms": 12.3
        }
    """
    try:
        required_fields = ["model_name", "mean_prediction", "std_prediction", 
                          "mean_latency_ms", "std_latency_ms"]
        for field in required_fields:
            if field not in request:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required field: {field}"
                )
        
        monitoring_manager.set_baseline(
            model_name=request["model_name"],
            mean_prediction=float(request["mean_prediction"]),
            std_prediction=float(request["std_prediction"]),
            mean_latency_ms=float(request["mean_latency_ms"]),
            std_latency_ms=float(request["std_latency_ms"])
        )
        
        return {"message": f"Baseline set for model {request['model_name']}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting baseline: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set baseline: {str(e)}"
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

