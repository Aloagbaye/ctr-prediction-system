"""
BigQuery Logger for CTR Predictions

Logs predictions to BigQuery for analytics and monitoring
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

try:
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    logging.warning("google-cloud-bigquery not installed. BigQuery logging disabled.")

logger = logging.getLogger(__name__)


class BigQueryLogger:
    """Log predictions to BigQuery"""
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        table_name: str = "predictions"
    ):
        """
        Initialize BigQuery logger
        
        Args:
            project_id: GCP project ID (defaults to environment variable)
            dataset_name: BigQuery dataset name (defaults to environment variable)
            table_name: BigQuery table name
        """
        if not BIGQUERY_AVAILABLE:
            self.enabled = False
            logger.warning("BigQuery not available. Logging disabled.")
            return
        
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.dataset_name = dataset_name or os.getenv("BIGQUERY_DATASET", "ctr_predictions")
        self.table_name = table_name or os.getenv("BIGQUERY_TABLE", "predictions")
        
        if not self.project_id or not self.dataset_name:
            self.enabled = False
            logger.warning("BigQuery project_id or dataset_name not configured. Logging disabled.")
            return
        
        try:
            self.client = bigquery.Client(project=self.project_id)
            self.table_ref = self.client.dataset(self.dataset_name).table(self.table_name)
            self.enabled = True
            logger.info(f"BigQuery logger initialized: {self.project_id}.{self.dataset_name}.{self.table_name}")
        except Exception as e:
            self.enabled = False
            logger.error(f"Failed to initialize BigQuery logger: {e}")
    
    def log_prediction(
        self,
        request_data: Dict[str, Any],
        prediction_result: Dict[str, Any],
        actual_ctr: Optional[float] = None
    ) -> bool:
        """
        Log a prediction to BigQuery
        
        Args:
            request_data: Original request data
            prediction_result: Prediction result with predicted_ctr, model_name, etc.
            actual_ctr: Actual CTR if available (for evaluation)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Prepare row data
            row = {
                "user_id": request_data.get("user_id", ""),
                "ad_id": request_data.get("ad_id", ""),
                "device": request_data.get("device", ""),
                "placement": request_data.get("placement", ""),
                "predicted_ctr": float(prediction_result.get("predicted_ctr", 0.0)),
                "actual_ctr": float(actual_ctr) if actual_ctr is not None else None,
                "timestamp": datetime.utcnow().isoformat(),
                "model_name": prediction_result.get("model_name", "unknown"),
                "request_id": str(uuid.uuid4()),
                "features": json.dumps({
                    "hour": request_data.get("hour"),
                    "day_of_week": request_data.get("day_of_week"),
                    "timestamp": request_data.get("timestamp")
                })
            }
            
            # Insert row
            errors = self.client.insert_rows_json(self.table_ref, [row])
            
            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
                return False
            
            logger.debug(f"Logged prediction to BigQuery: {row['request_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging to BigQuery: {e}")
            return False
    
    def log_batch_predictions(
        self,
        requests: list[Dict[str, Any]],
        predictions: list[Dict[str, Any]],
        actual_ctrs: Optional[list[float]] = None
    ) -> int:
        """
        Log multiple predictions to BigQuery
        
        Args:
            requests: List of request data dictionaries
            predictions: List of prediction results
            actual_ctrs: Optional list of actual CTRs
            
        Returns:
            Number of successfully logged predictions
        """
        if not self.enabled:
            return 0
        
        if actual_ctrs is None:
            actual_ctrs = [None] * len(requests)
        
        success_count = 0
        for request, prediction, actual_ctr in zip(requests, predictions, actual_ctrs):
            if self.log_prediction(request, prediction, actual_ctr):
                success_count += 1
        
        logger.info(f"Logged {success_count}/{len(requests)} predictions to BigQuery")
        return success_count

