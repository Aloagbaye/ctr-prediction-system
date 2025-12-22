"""
A/B Testing Manager for CTR Prediction Models

Handles:
- Traffic splitting between model versions
- Performance tracking and comparison
- A/B test configuration and management
"""

import hashlib
import json
import logging
import time
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import threading

# MLflow integration
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not installed. A/B test logging to MLflow disabled.")

logger = logging.getLogger(__name__)


@dataclass
class ABTestConfig:
    """Configuration for an A/B test"""
    test_id: str
    model_a: str  # Control model (e.g., "xgboost")
    model_b: str  # Variant model (e.g., "lightgbm")
    traffic_split: float  # Percentage of traffic to model_b (0.0 to 1.0)
    start_time: datetime
    end_time: Optional[datetime] = None
    enabled: bool = True
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            **asdict(self),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None
        }


@dataclass
class ABTestMetrics:
    """Metrics for A/B test performance comparison"""
    test_id: str
    model_name: str
    total_requests: int = 0
    total_predictions: int = 0
    avg_prediction_time_ms: float = 0.0
    total_prediction_time_ms: float = 0.0
    min_prediction_time_ms: float = float('inf')
    max_prediction_time_ms: float = 0.0
    error_count: int = 0
    last_updated: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            **asdict(self),
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'min_prediction_time_ms': self.min_prediction_time_ms if self.min_prediction_time_ms != float('inf') else None
        }


class ABTestManager:
    """
    Manages A/B tests for model comparison
    
    Features:
    - Consistent traffic splitting using user_id hashing
    - Performance metrics tracking
    - Real-time comparison statistics
    """
    
    def __init__(
        self,
        storage_path: str = "data/ab_tests",
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: str = "ctr_prediction_ab_tests",
        enable_mlflow: bool = True
    ):
        """
        Initialize A/B Test Manager
        
        Args:
            storage_path: Path to store A/B test data
            mlflow_tracking_uri: MLflow tracking URI (defaults to local file store)
            mlflow_experiment_name: MLflow experiment name for A/B tests
            enable_mlflow: Whether to enable MLflow logging
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Active A/B tests: test_id -> ABTestConfig
        self.active_tests: Dict[str, ABTestConfig] = {}
        
        # Metrics: test_id -> model_name -> ABTestMetrics
        self.metrics: Dict[str, Dict[str, ABTestMetrics]] = defaultdict(dict)
        
        # Thread lock for thread-safe operations
        self._lock = threading.Lock()
        
        # MLflow integration
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        self.mlflow_experiment_name = mlflow_experiment_name
        self.mlflow_run_ids: Dict[str, str] = {}  # test_id -> run_id
        
        if self.enable_mlflow:
            self._setup_mlflow(mlflow_tracking_uri)
        
        # Load existing tests and metrics
        self._load_tests()
        self._load_metrics()
    
    def _setup_mlflow(self, mlflow_tracking_uri: Optional[str] = None):
        """Setup MLflow for A/B test logging"""
        try:
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
            else:
                # Use same tracking URI as monitoring (Windows-compatible)
                project_root = Path(__file__).parent.parent.parent
                mlruns_dir = project_root / "mlruns"
                mlruns_dir.mkdir(exist_ok=True)
                
                abs_path = mlruns_dir.absolute()
                if os.name == 'nt':  # Windows
                    tracking_uri = f"file:///{str(abs_path).replace(chr(92), '/')}"
                else:
                    tracking_uri = f"file://{abs_path.as_posix()}"
                
                mlflow.set_tracking_uri(tracking_uri)
            
            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(self.mlflow_experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.mlflow_experiment_name)
                logger.info(f"Created MLflow experiment for A/B tests: {self.mlflow_experiment_name}")
            else:
                logger.info(f"Using MLflow experiment for A/B tests: {self.mlflow_experiment_name}")
            
            mlflow.set_experiment(self.mlflow_experiment_name)
            logger.info(f"MLflow A/B test logging enabled: {mlflow.get_tracking_uri()}")
        except Exception as e:
            logger.warning(f"MLflow setup failed for A/B tests: {e}")
            self.enable_mlflow = False
    
    def create_test(
        self,
        test_id: str,
        model_a: str,
        model_b: str,
        traffic_split: float = 0.5,
        duration_hours: Optional[int] = None,
        description: str = ""
    ) -> ABTestConfig:
        """
        Create a new A/B test
        
        Args:
            test_id: Unique identifier for the test
            model_a: Control model name
            model_b: Variant model name
            traffic_split: Percentage of traffic to model_b (0.0 to 1.0)
            duration_hours: Test duration in hours (None for indefinite)
            description: Test description
        
        Returns:
            ABTestConfig object
        """
        if not 0.0 <= traffic_split <= 1.0:
            raise ValueError("traffic_split must be between 0.0 and 1.0")
        
        if test_id in self.active_tests:
            raise ValueError(f"Test {test_id} already exists")
        
        start_time = datetime.now()
        end_time = None
        if duration_hours:
            end_time = start_time + timedelta(hours=duration_hours)
        
        config = ABTestConfig(
            test_id=test_id,
            model_a=model_a,
            model_b=model_b,
            traffic_split=traffic_split,
            start_time=start_time,
            end_time=end_time,
            enabled=True,
            description=description
        )
        
        with self._lock:
            self.active_tests[test_id] = config
            # Initialize metrics for both models
            self.metrics[test_id][model_a] = ABTestMetrics(test_id=test_id, model_name=model_a)
            self.metrics[test_id][model_b] = ABTestMetrics(test_id=test_id, model_name=model_b)
            self._save_tests()
        
        # Create MLflow run for this A/B test
        if self.enable_mlflow:
            self._create_mlflow_run(test_id, config)
        
        logger.info(f"Created A/B test: {test_id} ({model_a} vs {model_b}, {traffic_split*100:.1f}% to {model_b})")
        return config
    
    def _create_mlflow_run(self, test_id: str, config: ABTestConfig):
        """Create an MLflow run for an A/B test"""
        try:
            run_name = f"ab_test_{test_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run = mlflow.start_run(run_name=run_name, nested=False)
            run_id = run.info.run_id
            self.mlflow_run_ids[test_id] = run_id
            
            # Log A/B test configuration as parameters
            mlflow.log_params({
                "test_id": test_id,
                "model_a": config.model_a,
                "model_b": config.model_b,
                "traffic_split": config.traffic_split,
                "description": config.description or ""
            })
            
            # Set tags
            mlflow.set_tag("test_type", "ab_test")
            mlflow.set_tag("test_id", test_id)
            mlflow.set_tag("model_a", config.model_a)
            mlflow.set_tag("model_b", config.model_b)
            mlflow.set_tag("start_time", config.start_time.isoformat())
            if config.end_time:
                mlflow.set_tag("end_time", config.end_time.isoformat())
            
            logger.info(f"Created MLflow run for A/B test {test_id}: {run_id[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to create MLflow run for A/B test {test_id}: {e}")
    
    def get_model_for_request(
        self,
        test_id: str,
        user_id: str
    ) -> Tuple[str, str]:
        """
        Determine which model to use for a request based on traffic splitting
        
        Uses consistent hashing on user_id to ensure same user always gets same model
        
        Args:
            test_id: A/B test identifier
            user_id: User identifier for consistent routing
        
        Returns:
            Tuple of (model_name, variant) where variant is 'A' or 'B'
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        config = self.active_tests[test_id]
        
        if not config.enabled:
            return config.model_a, 'A'
        
        # Check if test has expired
        if config.end_time and datetime.now() > config.end_time:
            logger.info(f"Test {test_id} has expired, disabling")
            config.enabled = False
            self._save_tests()
            return config.model_a, 'A'
        
        # Consistent hashing: hash user_id and map to [0, 1)
        hash_value = int(hashlib.md5(f"{test_id}:{user_id}".encode()).hexdigest(), 16)
        hash_ratio = (hash_value % 10000) / 10000.0
        
        # Route based on traffic split
        if hash_ratio < config.traffic_split:
            return config.model_b, 'B'
        else:
            return config.model_a, 'A'
    
    def record_prediction(
        self,
        test_id: str,
        model_name: str,
        prediction_time_ms: float,
        error: bool = False
    ):
        """
        Record a prediction for metrics tracking
        
        Args:
            test_id: A/B test identifier
            model_name: Model that made the prediction
            prediction_time_ms: Prediction latency in milliseconds
            error: Whether an error occurred
        """
        with self._lock:
            if test_id not in self.metrics:
                self.metrics[test_id] = {}
            
            if model_name not in self.metrics[test_id]:
                self.metrics[test_id][model_name] = ABTestMetrics(
                    test_id=test_id,
                    model_name=model_name
                )
            
            metrics = self.metrics[test_id][model_name]
            metrics.total_requests += 1
            
            if not error:
                metrics.total_predictions += 1
                metrics.total_prediction_time_ms += prediction_time_ms
                metrics.avg_prediction_time_ms = (
                    metrics.total_prediction_time_ms / metrics.total_predictions
                )
                metrics.min_prediction_time_ms = min(
                    metrics.min_prediction_time_ms,
                    prediction_time_ms
                )
                metrics.max_prediction_time_ms = max(
                    metrics.max_prediction_time_ms,
                    prediction_time_ms
                )
            else:
                metrics.error_count += 1
            
            metrics.last_updated = datetime.now()
            self._save_metrics()
            
            # Log to MLflow periodically (every 50 requests for performance)
            if self.enable_mlflow and metrics.total_requests % 50 == 0:
                self._log_metrics_to_mlflow(test_id, model_name, metrics)
    
    def get_test_stats(self, test_id: str) -> Dict[str, Any]:
        """
        Get statistics for an A/B test
        
        Args:
            test_id: A/B test identifier
        
        Returns:
            Dictionary with test statistics
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        config = self.active_tests[test_id]
        metrics_a = self.metrics[test_id].get(config.model_a)
        metrics_b = self.metrics[test_id].get(config.model_b)
        
        stats = {
            'test_id': test_id,
            'config': config.to_dict(),
            'model_a_stats': metrics_a.to_dict() if metrics_a else None,
            'model_b_stats': metrics_b.to_dict() if metrics_b else None,
        }
        
        # Calculate comparison metrics
        if metrics_a and metrics_b:
            stats['comparison'] = {
                'traffic_split': {
                    'model_a': f"{(1 - config.traffic_split) * 100:.1f}%",
                    'model_b': f"{config.traffic_split * 100:.1f}%"
                },
                'total_requests': {
                    'model_a': metrics_a.total_requests,
                    'model_b': metrics_b.total_requests,
                    'total': metrics_a.total_requests + metrics_b.total_requests
                },
                'avg_latency_ms': {
                    'model_a': round(metrics_a.avg_prediction_time_ms, 2),
                    'model_b': round(metrics_b.avg_prediction_time_ms, 2),
                    'difference_ms': round(
                        metrics_b.avg_prediction_time_ms - metrics_a.avg_prediction_time_ms,
                        2
                    ),
                    'improvement_pct': round(
                        ((metrics_a.avg_prediction_time_ms - metrics_b.avg_prediction_time_ms) /
                         metrics_a.avg_prediction_time_ms * 100) if metrics_a.avg_prediction_time_ms > 0 else 0,
                        2
                    )
                },
                'error_rate': {
                    'model_a': round(
                        (metrics_a.error_count / metrics_a.total_requests * 100)
                        if metrics_a.total_requests > 0 else 0,
                        2
                    ),
                    'model_b': round(
                        (metrics_b.error_count / metrics_b.total_requests * 100)
                        if metrics_b.total_requests > 0 else 0,
                        2
                    )
                }
            }
        
        # Log comparison metrics to MLflow
        if self.enable_mlflow and metrics_a and metrics_b:
            self._log_comparison_to_mlflow(test_id, stats['comparison'])
        
        return stats
    
    def _log_metrics_to_mlflow(self, test_id: str, model_name: str, metrics: ABTestMetrics):
        """Log individual model metrics to MLflow"""
        if test_id not in self.mlflow_run_ids:
            return
        
        try:
            run_id = self.mlflow_run_ids[test_id]
            with mlflow.start_run(run_id=run_id):
                # Log metrics with model prefix
                mlflow.log_metrics({
                    f"{model_name}_total_requests": metrics.total_requests,
                    f"{model_name}_total_predictions": metrics.total_predictions,
                    f"{model_name}_avg_latency_ms": metrics.avg_prediction_time_ms,
                    f"{model_name}_min_latency_ms": metrics.min_prediction_time_ms if metrics.min_prediction_time_ms != float('inf') else 0,
                    f"{model_name}_max_latency_ms": metrics.max_prediction_time_ms,
                    f"{model_name}_error_count": metrics.error_count,
                    f"{model_name}_error_rate": (metrics.error_count / metrics.total_requests * 100) if metrics.total_requests > 0 else 0
                })
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow for {test_id}/{model_name}: {e}")
    
    def _log_comparison_to_mlflow(self, test_id: str, comparison: Dict[str, Any]):
        """Log A/B test comparison metrics to MLflow"""
        if test_id not in self.mlflow_run_ids:
            return
        
        try:
            run_id = self.mlflow_run_ids[test_id]
            with mlflow.start_run(run_id=run_id):
                # Log comparison metrics
                mlflow.log_metrics({
                    "total_requests": comparison.get('total_requests', {}).get('total', 0),
                    "latency_diff_ms": comparison.get('avg_latency_ms', {}).get('difference_ms', 0),
                    "latency_improvement_pct": comparison.get('avg_latency_ms', {}).get('improvement_pct', 0),
                    "error_rate_a": comparison.get('error_rate', {}).get('model_a', 0),
                    "error_rate_b": comparison.get('error_rate', {}).get('model_b', 0),
                })
                
                # Log individual model metrics
                if 'avg_latency_ms' in comparison:
                    mlflow.log_metrics({
                        "model_a_avg_latency_ms": comparison['avg_latency_ms'].get('model_a', 0),
                        "model_b_avg_latency_ms": comparison['avg_latency_ms'].get('model_b', 0),
                    })
        except Exception as e:
            logger.warning(f"Failed to log comparison to MLflow for {test_id}: {e}")
    
    def list_tests(self) -> List[Dict[str, Any]]:
        """List all active A/B tests"""
        return [config.to_dict() for config in self.active_tests.values()]
    
    def stop_test(self, test_id: str):
        """Stop an A/B test"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        with self._lock:
            self.active_tests[test_id].enabled = False
            self._save_tests()
        
        # Log final stats to MLflow and end run
        if self.enable_mlflow and test_id in self.mlflow_run_ids:
            try:
                # Log final comparison
                stats = self.get_test_stats(test_id)
                if 'comparison' in stats:
                    self._log_comparison_to_mlflow(test_id, stats['comparison'])
                
                # End the MLflow run
                mlflow.end_run()
                logger.info(f"Ended MLflow run for A/B test: {test_id}")
            except Exception as e:
                logger.warning(f"Failed to end MLflow run for {test_id}: {e}")
        
        logger.info(f"Stopped A/B test: {test_id}")
    
    def delete_test(self, test_id: str):
        """Delete an A/B test and its metrics"""
        # End MLflow run if it exists
        if self.enable_mlflow and test_id in self.mlflow_run_ids:
            try:
                # Log final stats before deletion
                if test_id in self.active_tests:
                    stats = self.get_test_stats(test_id)
                    if 'comparison' in stats:
                        self._log_comparison_to_mlflow(test_id, stats['comparison'])
                
                # End the MLflow run
                mlflow.end_run()
                del self.mlflow_run_ids[test_id]
                logger.info(f"Ended and removed MLflow run for A/B test: {test_id}")
            except Exception as e:
                logger.warning(f"Failed to end MLflow run for {test_id}: {e}")
        
        with self._lock:
            if test_id in self.active_tests:
                del self.active_tests[test_id]
            if test_id in self.metrics:
                del self.metrics[test_id]
            self._save_tests()
            self._save_metrics()
        
        logger.info(f"Deleted A/B test: {test_id}")
    
    def _save_tests(self):
        """Save tests to disk"""
        tests_file = self.storage_path / "tests.json"
        tests_data = {
            test_id: config.to_dict()
            for test_id, config in self.active_tests.items()
        }
        with open(tests_file, 'w') as f:
            json.dump(tests_data, f, indent=2)
    
    def _load_tests(self):
        """Load tests from disk"""
        tests_file = self.storage_path / "tests.json"
        if not tests_file.exists():
            return
        
        try:
            with open(tests_file, 'r') as f:
                tests_data = json.load(f)
            
            for test_id, config_data in tests_data.items():
                config_data['start_time'] = datetime.fromisoformat(config_data['start_time'])
                if config_data.get('end_time'):
                    config_data['end_time'] = datetime.fromisoformat(config_data['end_time'])
                self.active_tests[test_id] = ABTestConfig(**config_data)
        except Exception as e:
            logger.error(f"Error loading tests: {e}")
    
    def _save_metrics(self):
        """Save metrics to disk"""
        metrics_file = self.storage_path / "metrics.json"
        metrics_data = {
            test_id: {
                model_name: metrics.to_dict()
                for model_name, metrics in model_metrics.items()
            }
            for test_id, model_metrics in self.metrics.items()
        }
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
    
    def _load_metrics(self):
        """Load metrics from disk"""
        metrics_file = self.storage_path / "metrics.json"
        if not metrics_file.exists():
            return
        
        try:
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            for test_id, model_metrics_data in metrics_data.items():
                for model_name, metrics_dict in model_metrics_data.items():
                    if metrics_dict.get('last_updated'):
                        metrics_dict['last_updated'] = datetime.fromisoformat(metrics_dict['last_updated'])
                    if metrics_dict.get('min_prediction_time_ms') is None:
                        metrics_dict['min_prediction_time_ms'] = float('inf')
                    self.metrics[test_id][model_name] = ABTestMetrics(**metrics_dict)
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")

