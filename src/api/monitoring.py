"""
Real-time Monitoring System with MLflow Integration

Features:
- Prediction latency tracking
- Model performance drift detection
- Error rate monitoring
- Request volume tracking
- MLflow metrics logging
"""

import os
import logging
import time
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, asdict
from threading import Lock
import json
from pathlib import Path

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not installed. Monitoring will work without MLflow tracking.")

logger = logging.getLogger(__name__)


@dataclass
class MonitoringMetrics:
    """Metrics for monitoring"""
    model_name: str
    timestamp: datetime
    latency_ms: float
    error: bool = False
    prediction_value: Optional[float] = None
    request_id: Optional[str] = None


@dataclass
class ModelStats:
    """Statistics for a model"""
    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    requests_per_minute: float = 0.0
    error_rate: float = 0.0
    last_updated: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        result = asdict(self)
        result['last_updated'] = self.last_updated.isoformat() if self.last_updated else None
        result['min_latency_ms'] = self.min_latency_ms if self.min_latency_ms != float('inf') else None
        return result


class DriftDetector:
    """
    Detects model performance drift
    
    Methods:
    - Statistical drift detection (mean shift, variance change)
    - Prediction distribution monitoring
    - Latency anomaly detection
    """
    
    def __init__(self, window_size: int = 1000, threshold: float = 2.0):
        """
        Initialize drift detector
        
        Args:
            window_size: Number of recent predictions to keep for comparison
            threshold: Z-score threshold for drift detection (default: 2.0 = 95% confidence)
        """
        self.window_size = window_size
        self.threshold = threshold
        
        # Store recent predictions and latencies per model
        self.prediction_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
        # Baseline statistics (from training/initial period)
        self.baseline_mean: Dict[str, float] = {}
        self.baseline_std: Dict[str, float] = {}
        self.baseline_latency_mean: Dict[str, float] = {}
        self.baseline_latency_std: Dict[str, float] = {}
        
        self._lock = Lock()
    
    def set_baseline(
        self,
        model_name: str,
        mean_prediction: float,
        std_prediction: float,
        mean_latency_ms: float,
        std_latency_ms: float
    ):
        """Set baseline statistics for a model"""
        with self._lock:
            self.baseline_mean[model_name] = mean_prediction
            self.baseline_std[model_name] = std_prediction
            self.baseline_latency_mean[model_name] = mean_latency_ms
            self.baseline_latency_std[model_name] = std_latency_ms
    
    def record_prediction(
        self,
        model_name: str,
        prediction_value: float,
        latency_ms: float
    ):
        """Record a prediction for drift detection"""
        with self._lock:
            self.prediction_history[model_name].append(prediction_value)
            self.latency_history[model_name].append(latency_ms)
    
    def detect_drift(self, model_name: str) -> Dict[str, Any]:
        """
        Detect if model has drifted from baseline
        
        Returns:
            Dictionary with drift detection results
        """
        with self._lock:
            if model_name not in self.prediction_history:
                return {
                    "drift_detected": False,
                    "reason": "Insufficient data"
                }
            
            predictions = list(self.prediction_history[model_name])
            latencies = list(self.latency_history[model_name])
            
            if len(predictions) < 100:  # Need minimum data
                return {
                    "drift_detected": False,
                    "reason": "Insufficient data for drift detection"
                }
            
            # Calculate current statistics
            import numpy as np
            current_mean = np.mean(predictions)
            current_std = np.std(predictions)
            current_latency_mean = np.mean(latencies)
            current_latency_std = np.std(latencies)
            
            drift_results = {
                "drift_detected": False,
                "model_name": model_name,
                "current_stats": {
                    "mean_prediction": float(current_mean),
                    "std_prediction": float(current_std),
                    "mean_latency_ms": float(current_latency_mean),
                    "std_latency_ms": float(current_latency_std)
                },
                "drifts": []
            }
            
            # Check baseline if available
            if model_name in self.baseline_mean:
                baseline_mean = self.baseline_mean[model_name]
                baseline_std = self.baseline_std[model_name]
                
                # Z-score for mean shift
                if baseline_std > 0:
                    z_score_mean = abs(current_mean - baseline_mean) / baseline_std
                    if z_score_mean > self.threshold:
                        drift_results["drift_detected"] = True
                        drift_results["drifts"].append({
                            "type": "mean_shift",
                            "severity": "high" if z_score_mean > 3.0 else "medium",
                            "z_score": float(z_score_mean),
                            "baseline_mean": float(baseline_mean),
                            "current_mean": float(current_mean)
                        })
                
                # Variance change
                if baseline_std > 0:
                    variance_ratio = current_std / baseline_std
                    if variance_ratio > 1.5 or variance_ratio < 0.67:
                        drift_results["drift_detected"] = True
                        drift_results["drifts"].append({
                            "type": "variance_change",
                            "severity": "high" if variance_ratio > 2.0 or variance_ratio < 0.5 else "medium",
                            "variance_ratio": float(variance_ratio),
                            "baseline_std": float(baseline_std),
                            "current_std": float(current_std)
                        })
            
            # Latency anomaly detection
            if model_name in self.baseline_latency_mean:
                baseline_latency_mean = self.baseline_latency_mean[model_name]
                baseline_latency_std = self.baseline_latency_std[model_name]
                
                if baseline_latency_std > 0:
                    z_score_latency = abs(current_latency_mean - baseline_latency_mean) / baseline_latency_std
                    if z_score_latency > self.threshold:
                        drift_results["drift_detected"] = True
                        drift_results["drifts"].append({
                            "type": "latency_anomaly",
                            "severity": "high" if z_score_latency > 3.0 else "medium",
                            "z_score": float(z_score_latency),
                            "baseline_latency_ms": float(baseline_latency_mean),
                            "current_latency_ms": float(current_latency_mean)
                        })
            
            return drift_results


class MonitoringManager:
    """
    Real-time monitoring manager with MLflow integration
    
    Features:
    - Metrics collection and aggregation
    - MLflow tracking integration
    - Drift detection
    - Performance statistics
    """
    
    def __init__(
        self,
        mlflow_tracking_uri: Optional[str] = None,
        experiment_name: str = "ctr_prediction_production",
        enable_mlflow: bool = True
    ):
        """
        Initialize monitoring manager
        
        Args:
            mlflow_tracking_uri: MLflow tracking URI (defaults to local file store)
            experiment_name: MLflow experiment name
            enable_mlflow: Whether to enable MLflow tracking
        """
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        
        # MLflow setup
        if self.enable_mlflow:
            try:
                if mlflow_tracking_uri:
                    mlflow.set_tracking_uri(mlflow_tracking_uri)
                else:
                    # Use local file store (default MLflow location)
                    # On Windows, use absolute path for compatibility
                    from pathlib import Path
                    import os
                    project_root = Path(__file__).parent.parent.parent
                    mlruns_dir = project_root / "mlruns"
                    mlruns_dir.mkdir(exist_ok=True)
                    # Use absolute path with file:// prefix
                    # On Windows, convert backslashes to forward slashes
                    abs_path = mlruns_dir.absolute()
                    if os.name == 'nt':  # Windows
                        tracking_uri = f"file:///{str(abs_path).replace(chr(92), '/')}"
                    else:
                        tracking_uri = f"file://{abs_path.as_posix()}"
                    mlflow.set_tracking_uri(tracking_uri)
                
                # Get or create experiment
                try:
                    experiment = mlflow.get_experiment_by_name(experiment_name)
                    if experiment is None:
                        mlflow.create_experiment(experiment_name)
                    mlflow.set_experiment(experiment_name)
                except Exception as e:
                    logger.warning(f"Could not set up MLflow experiment: {e}")
                    self.enable_mlflow = False
                
                logger.info(f"MLflow tracking enabled: {mlflow.get_tracking_uri()}")
            except Exception as e:
                logger.warning(f"MLflow initialization failed: {e}. Continuing without MLflow.")
                self.enable_mlflow = False
        
        # Metrics storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.model_stats: Dict[str, ModelStats] = {}
        
        # Time windows for rate calculations
        self.request_timestamps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Drift detector
        self.drift_detector = DriftDetector()
        
        # Thread lock
        self._lock = Lock()
        
        # MLflow run (one per model)
        self.mlflow_runs: Dict[str, str] = {}
        
        # Initialize MLflow runs for each model
        self._init_mlflow_runs()
    
    def _init_mlflow_runs(self):
        """Initialize MLflow runs for monitoring"""
        if not self.enable_mlflow:
            return
        
        try:
            # Create a run for production monitoring
            # Use a single active run that stays open
            run_name = f"production_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run = mlflow.start_run(run_name=run_name)
            self.mlflow_runs["production"] = run.info.run_id
            # Keep run active (don't end it)
            logger.info(f"Started MLflow run: {run.info.run_id} (experiment: {mlflow.get_experiment_by_name('ctr_prediction_production').experiment_id if mlflow.get_experiment_by_name('ctr_prediction_production') else 'default'})")
        except Exception as e:
            logger.warning(f"Could not initialize MLflow run: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def record_prediction(
        self,
        model_name: str,
        latency_ms: float,
        error: bool = False,
        prediction_value: Optional[float] = None,
        request_id: Optional[str] = None
    ):
        """
        Record a prediction for monitoring
        
        Args:
            model_name: Name of the model
            latency_ms: Prediction latency in milliseconds
            error: Whether an error occurred
            prediction_value: Predicted CTR value
            request_id: Optional request identifier
        """
        timestamp = datetime.now()
        
        with self._lock:
            # Initialize stats if needed
            if model_name not in self.model_stats:
                self.model_stats[model_name] = ModelStats(model_name=model_name)
            
            stats = self.model_stats[model_name]
            
            # Update statistics
            stats.total_requests += 1
            self.request_timestamps[model_name].append(timestamp)
            
            if error:
                stats.error_count += 1
            else:
                stats.successful_requests += 1
                
                # Update latency statistics
                self.metrics_history[model_name].append(latency_ms)
                latencies = list(self.metrics_history[model_name])
                
                if latencies:
                    import numpy as np
                    stats.avg_latency_ms = float(np.mean(latencies))
                    stats.p50_latency_ms = float(np.percentile(latencies, 50))
                    stats.p95_latency_ms = float(np.percentile(latencies, 95))
                    stats.p99_latency_ms = float(np.percentile(latencies, 99))
                    stats.min_latency_ms = float(min(latencies))
                    stats.max_latency_ms = float(max(latencies))
                
                # Record for drift detection
                if prediction_value is not None:
                    self.drift_detector.record_prediction(model_name, prediction_value, latency_ms)
            
            # Calculate error rate
            if stats.total_requests > 0:
                stats.error_rate = (stats.error_count / stats.total_requests) * 100
            
            # Calculate requests per minute
            recent_timestamps = list(self.request_timestamps[model_name])
            if len(recent_timestamps) > 1:
                time_span = (recent_timestamps[-1] - recent_timestamps[0]).total_seconds() / 60.0
                if time_span > 0:
                    stats.requests_per_minute = len(recent_timestamps) / time_span
            
            stats.last_updated = timestamp
            
            # Log to MLflow
            if self.enable_mlflow:
                self._log_to_mlflow(model_name, latency_ms, error, prediction_value)
    
    def _log_to_mlflow(
        self,
        model_name: str,
        latency_ms: float,
        error: bool,
        prediction_value: Optional[float]
    ):
        """Log metrics to MLflow"""
        if not self.enable_mlflow:
            return
            
        try:
            # Check if we have an active run
            active_run = mlflow.active_run()
            
            if not active_run:
                # No active run - create one
                run_id = self.mlflow_runs.get("production")
                if run_id:
                    # Try to resume the existing run
                    try:
                        mlflow.start_run(run_id=run_id)
                    except:
                        # Run might be ended, create a new one
                        run_name = f"production_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        run = mlflow.start_run(run_name=run_name)
                        run_id = run.info.run_id
                        self.mlflow_runs["production"] = run_id
                else:
                    # Create a new run
                    run_name = f"production_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    run = mlflow.start_run(run_name=run_name)
                    run_id = run.info.run_id
                    self.mlflow_runs["production"] = run_id
            
            # Now we have an active run - log metrics directly
            # Use request count as step for better visualization
            if model_name in self.model_stats:
                step = self.model_stats[model_name].total_requests
            else:
                step = int(time.time())
            
            # Log individual metrics
            mlflow.log_metric(f"{model_name}_latency_ms", latency_ms, step=step)
            mlflow.log_metric(f"{model_name}_error", 1 if error else 0, step=step)
            
            if prediction_value is not None:
                mlflow.log_metric(f"{model_name}_prediction", prediction_value, step=step)
            
            # Log aggregated metrics periodically (every 10 requests for better visibility)
            if model_name in self.model_stats:
                stats = self.model_stats[model_name]
                if stats.total_requests % 10 == 0:  # Log every 10 requests
                    mlflow.log_metrics({
                        f"{model_name}_avg_latency_ms": stats.avg_latency_ms,
                        f"{model_name}_p95_latency_ms": stats.p95_latency_ms,
                        f"{model_name}_error_rate": stats.error_rate,
                        f"{model_name}_requests_per_minute": stats.requests_per_minute,
                        f"{model_name}_total_requests": stats.total_requests
                    }, step=stats.total_requests)
                    
        except Exception as e:
            logger.debug(f"MLflow logging error (non-critical): {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def get_model_stats(self, model_name: str) -> Optional[ModelStats]:
        """Get statistics for a model"""
        with self._lock:
            return self.model_stats.get(model_name)
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all models"""
        with self._lock:
            return {
                model_name: stats.to_dict()
                for model_name, stats in self.model_stats.items()
            }
    
    def check_drift(self, model_name: str) -> Dict[str, Any]:
        """Check for model drift"""
        return self.drift_detector.detect_drift(model_name)
    
    def set_baseline(
        self,
        model_name: str,
        mean_prediction: float,
        std_prediction: float,
        mean_latency_ms: float,
        std_latency_ms: float
    ):
        """Set baseline statistics for drift detection"""
        self.drift_detector.set_baseline(
            model_name, mean_prediction, std_prediction, mean_latency_ms, std_latency_ms
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        with self._lock:
            health = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "models": {},
                "alerts": []
            }
            
            for model_name, stats in self.model_stats.items():
                model_health = {
                    "status": "healthy",
                    "total_requests": stats.total_requests,
                    "error_rate": stats.error_rate,
                    "avg_latency_ms": stats.avg_latency_ms,
                    "requests_per_minute": stats.requests_per_minute
                }
                
                # Check for issues
                if stats.error_rate > 5.0:
                    model_health["status"] = "degraded"
                    health["alerts"].append({
                        "model": model_name,
                        "severity": "high",
                        "message": f"High error rate: {stats.error_rate:.2f}%"
                    })
                
                if stats.avg_latency_ms > 1000:  # > 1 second
                    model_health["status"] = "degraded"
                    health["alerts"].append({
                        "model": model_name,
                        "severity": "medium",
                        "message": f"High latency: {stats.avg_latency_ms:.2f}ms"
                    })
                
                # Check drift
                drift_result = self.check_drift(model_name)
                if drift_result.get("drift_detected"):
                    model_health["status"] = "warning"
                    health["alerts"].append({
                        "model": model_name,
                        "severity": "medium",
                        "message": "Model drift detected",
                        "drift_details": drift_result
                    })
                
                health["models"][model_name] = model_health
            
            # Overall status
            if any(m["status"] == "degraded" for m in health["models"].values()):
                health["status"] = "degraded"
            elif any(m["status"] == "warning" for m in health["models"].values()):
                health["status"] = "warning"
            
            return health

