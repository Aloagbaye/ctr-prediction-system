"""
Model Predictor

Handles model loading and prediction logic
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict, Any, List
import time
import logging
from datetime import datetime
import sys
import os
import tempfile
import shutil

# Add src to path for feature engineering imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.features.pipeline import FeatureEngineeringPipeline
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Load and use trained models for prediction"""
    
    def __init__(self, models_dir: str = "models", default_model: str = "xgboost"):
        """
        Initialize model predictor
        
        Args:
            models_dir: Directory containing trained models (or local path for GCS models)
            default_model: Default model to use ('logistic', 'xgboost', 'lightgbm')
        """
        self.models_dir = Path(models_dir)
        self.default_model = default_model
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.model_info = {}
        self.feature_pipeline = None
        self.gcs_bucket = os.getenv("GCS_BUCKET")
        self._temp_dir = None
        
        # Download models from GCS if bucket is specified
        if self.gcs_bucket:
            logger.info(f"GCS bucket specified: {self.gcs_bucket}. Downloading models...")
            self._download_models_from_gcs()
        
        # Initialize feature engineering pipeline if available
        if FEATURE_ENGINEERING_AVAILABLE:
            try:
                self.feature_pipeline = FeatureEngineeringPipeline(
                    target_col='clicked',
                    create_basic_features=True,
                    create_advanced_features=False,  # Skip advanced features for speed
                    apply_feature_selection=False  # Features already selected during training
                )
                logger.info("Feature engineering pipeline initialized")
            except Exception as e:
                logger.warning(f"Could not initialize feature pipeline: {e}")
    
    def _download_models_from_gcs(self):
        """Download models from Google Cloud Storage"""
        try:
            from google.cloud import storage
            
            # Create temporary directory for models
            self._temp_dir = tempfile.mkdtemp(prefix="ctr_models_")
            self.models_dir = Path(self._temp_dir)
            logger.info(f"Created temporary directory for models: {self._temp_dir}")
            
            # Initialize GCS client
            client = storage.Client()
            bucket = client.bucket(self.gcs_bucket)
            
            # List and download model files
            model_files = ['logistic_model.pkl', 'xgboost_model.pkl', 'lightgbm_model.pkl',
                          'logistic_scaler.pkl', 'feature_names.pkl']
            
            downloaded = []
            for blob_name in model_files:
                blob = bucket.blob(blob_name)
                if blob.exists():
                    local_path = self.models_dir / blob_name
                    blob.download_to_filename(str(local_path))
                    downloaded.append(blob_name)
                    logger.info(f"Downloaded {blob_name} from GCS")
            
            if downloaded:
                logger.info(f"Successfully downloaded {len(downloaded)} model files from GCS")
            else:
                logger.warning("No model files found in GCS bucket")
                
        except ImportError:
            logger.error("google-cloud-storage not installed. Install with: pip install google-cloud-storage")
            raise
        except Exception as e:
            logger.error(f"Error downloading models from GCS: {e}")
            # Fall back to local models_dir if GCS download fails
            if self._temp_dir and Path(self._temp_dir).exists():
                shutil.rmtree(self._temp_dir)
            self.models_dir = Path(os.getenv("MODELS_DIR", "models"))
            logger.warning(f"Falling back to local models directory: {self.models_dir}")
    
    def __del__(self):
        """Cleanup temporary directory if created"""
        if self._temp_dir and Path(self._temp_dir).exists():
            try:
                shutil.rmtree(self._temp_dir)
                logger.info(f"Cleaned up temporary directory: {self._temp_dir}")
            except Exception as e:
                logger.warning(f"Error cleaning up temp directory: {e}")
        
    def load_model(self, model_name: str) -> bool:
        """
        Load a specific model
        
        Args:
            model_name: Name of model to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = self.models_dir / f"{model_name}_model.pkl"
            
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                return False
            
            start_time = time.time()
            self.models[model_name] = joblib.load(model_path)
            load_time = time.time() - start_time
            
            # Load scaler if exists (for logistic regression)
            if model_name == 'logistic':
                scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
                if scaler_path.exists():
                    self.scalers[model_name] = joblib.load(scaler_path)
            
            # Load feature names if not already loaded
            if self.feature_names is None:
                feature_path = self.models_dir / "feature_names.pkl"
                if feature_path.exists():
                    self.feature_names = joblib.load(feature_path)
            
            self.model_info[model_name] = {
                'loaded': True,
                'load_time': load_time,
                'model_type': model_name
            }
            
            logger.info(f"Loaded {model_name} model in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def load_all_models(self) -> Dict[str, bool]:
        """
        Load all available models
        
        Returns:
            Dictionary mapping model names to load success status
        """
        model_types = ['logistic', 'xgboost', 'lightgbm']
        results = {}
        
        for model_type in model_types:
            results[model_type] = self.load_model(model_type)
        
        return results
    
    def prepare_features(self, request_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare features from request data
        
        Args:
            request_data: Dictionary with request fields
            
        Returns:
            DataFrame with features ready for prediction
        """
        # Extract timestamp if provided
        timestamp = request_data.get('timestamp')
        if timestamp and isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()
        
        # Extract temporal features
        hour = request_data.get('hour')
        if hour is None:
            hour = timestamp.hour if timestamp else datetime.now().hour
        
        day_of_week = request_data.get('day_of_week')
        if day_of_week is None:
            day_of_week = timestamp.weekday() if timestamp else datetime.now().weekday()
        
        # Create base features DataFrame
        features = {
            'user_id': [request_data['user_id']],
            'ad_id': [request_data['ad_id']],
            'device': [request_data['device']],
            'placement': [request_data['placement']],
            'timestamp': [timestamp],
            'hour': [hour],
            'day_of_week': [day_of_week],
            'is_weekend': [1 if day_of_week >= 5 else 0],
            'date': [timestamp.date() if timestamp else datetime.now().date()]
        }
        
        df = pd.DataFrame(features)
        
        # Note: In production, you would apply the same feature engineering pipeline
        # For now, we'll use the model with pre-engineered features
        # This assumes features were already created during training
        
        return df
    
    def predict(
        self,
        request_data: Dict[str, Any],
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make a prediction
        
        Args:
            request_data: Request data dictionary
            model_name: Model to use (default: self.default_model)
            
        Returns:
            Dictionary with prediction results
        """
        if model_name is None:
            model_name = self.default_model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Prepare features
        # Note: This is simplified - in production, you'd need to apply full feature engineering
        # For now, we assume the model expects numeric features matching training features
        
        # Convert to DataFrame for feature engineering
        df = self.prepare_features(request_data)
        
        # Apply feature engineering pipeline if available
        if self.feature_pipeline is not None:
            try:
                # Add dummy target column (required by pipeline but not used)
                df['clicked'] = 0
                
                # Use fit_transform for now (in production, pipeline should be pre-fitted)
                # This creates features but may not match training exactly
                df_features = self.feature_pipeline.fit_transform(df)
                
                # Remove target column
                if 'clicked' in df_features.columns:
                    df_features = df_features.drop('clicked', axis=1)
                
                # Select only numeric features
                X = df_features.select_dtypes(include=[np.number])
                
            except Exception as e:
                logger.warning(f"Feature engineering failed: {e}. Using simplified features.")
                X = self._create_simple_features(df)
        else:
            # Fallback to simple feature creation
            X = self._create_simple_features(df)
        
        # Ensure feature order matches training
        if self.feature_names:
            # Create DataFrame with all expected features
            X_aligned = pd.DataFrame(0, index=X.index, columns=self.feature_names)
            # Fill in available features
            for col in X.columns:
                if col in X_aligned.columns:
                    X_aligned[col] = X[col].values
            X = X_aligned
        else:
            logger.warning("Feature names not available. Using available features.")
        
        # Scale if needed (for logistic regression)
        if model_name == 'logistic' and model_name in self.scalers:
            X = self.scalers[model_name].transform(X)
        
        # Make prediction
        try:
            probability = model.predict_proba(X)[0, 1]
        except Exception as e:
            # Fallback: use simple prediction if feature mismatch
            logger.warning(f"Prediction error: {e}. Using default probability.")
            probability = 0.02  # Default CTR estimate
        
        return {
            'predicted_ctr': float(probability),
            'probability': float(probability),
            'model_name': model_name
        }
    
    def _create_simple_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create simple features as fallback
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with numeric features
        """
        numeric_features = {}
        
        # Device encoding
        device_map = {'mobile': 0, 'desktop': 1, 'tablet': 2}
        numeric_features['device_encoded'] = device_map.get(df['device'].iloc[0], 0)
        
        # Placement encoding
        placement_map = {'header': 0, 'sidebar': 1, 'footer': 2, 'in_content': 3, 'popup': 4}
        numeric_features['placement_encoded'] = placement_map.get(df['placement'].iloc[0], 0)
        
        # Temporal features
        numeric_features['hour'] = df['hour'].iloc[0]
        numeric_features['day_of_week'] = df['day_of_week'].iloc[0]
        numeric_features['is_weekend'] = df['is_weekend'].iloc[0]
        
        return pd.DataFrame([numeric_features])
    
    def predict_batch(
        self,
        requests: List[Dict[str, Any]],
        model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Make batch predictions
        
        Args:
            requests: List of request dictionaries
            model_name: Model to use
            
        Returns:
            List of prediction results
        """
        start_time = time.time()
        
        predictions = []
        for request in requests:
            pred = self.predict(request, model_name)
            predictions.append(pred)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            'predictions': predictions,
            'total': len(predictions),
            'processing_time_ms': processing_time
        }
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a model
        
        Args:
            model_name: Model name (default: default_model)
            
        Returns:
            Dictionary with model information
        """
        if model_name is None:
            model_name = self.default_model
        
        info = {
            'model_name': model_name,
            'model_type': model_name,
            'features': self.feature_names if self.feature_names else [],
            'num_features': len(self.feature_names) if self.feature_names else 0,
            'loaded': model_name in self.models,
            'load_time': self.model_info.get(model_name, {}).get('load_time')
        }
        
        return info

