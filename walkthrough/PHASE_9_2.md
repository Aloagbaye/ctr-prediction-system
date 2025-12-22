# Phase 9.2: Real-time Monitoring with MLflow

## Overview

This phase implements comprehensive real-time monitoring for the CTR prediction system using MLflow for metrics tracking. The monitoring system provides:

- **Prediction Latency Tracking**: Real-time latency metrics (avg, p50, p95, p99)
- **Model Performance Drift Detection**: Statistical drift detection for predictions and latency
- **Error Rate Monitoring**: Track and alert on error rates
- **Request Volume Tracking**: Monitor request rates and volumes
- **MLflow Integration**: Automatic logging to MLflow for visualization and analysis

## Features

### 1. Real-time Metrics Collection
- Automatic collection on every prediction
- Aggregated statistics per model
- Thread-safe metric storage
- Configurable history window

### 2. MLflow Integration
- Automatic metrics logging to MLflow
- Experiment tracking for production monitoring
- Historical trend analysis
- Model comparison capabilities

### 3. Drift Detection
- Statistical drift detection (mean shift, variance change)
- Latency anomaly detection
- Configurable thresholds
- Baseline comparison

### 4. Health Monitoring
- Overall system health status
- Per-model health checks
- Automatic alerting
- Status indicators (healthy/warning/degraded)

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       │ POST /predict_ctr
       ▼
┌─────────────────────────────────┐
│      FastAPI Application        │
│  ┌───────────────────────────┐ │
│  │   MonitoringManager       │ │
│  │  - Metrics Collection     │ │
│  │  - MLflow Integration     │ │
│  │  - Health Checks          │ │
│  └───────────────────────────┘ │
│              │                 │
│              ▼                 │
│  ┌───────────────────────────┐ │
│  │   DriftDetector           │ │
│  │  - Statistical Analysis  │ │
│  │  - Baseline Comparison   │ │
│  └───────────────────────────┘ │
│              │                 │
│              ▼                 │
│  ┌───────────────────────────┐ │
│  │   MLflow Tracking         │ │
│  │  - Metrics Logging       │ │
│  │  - Experiment Tracking   │ │
│  └───────────────────────────┘ │
└─────────────────────────────────┘
```

## MLflow Setup

### Installation

MLflow is automatically installed with:
```bash
pip install -r requirements.txt
```

### Local MLflow Tracking

By default, MLflow uses a local file store:
- **Tracking URI**: `file://./mlruns`
- **Experiment**: `ctr_prediction_production`

### Viewing MLflow UI

Start the MLflow UI to view metrics:
```bash
mlflow ui
```

Then open: `http://localhost:5000`

### Custom MLflow Tracking URI

Set environment variable for remote tracking:
```bash
export MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"
```

Or use a database backend:
```bash
export MLFLOW_TRACKING_URI="postgresql://user:pass@host:5432/mlflow"
```

## API Endpoints

### Get System Health
```http
GET /monitoring/health
```

Returns overall health status with alerts and model statistics.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "models": {
    "xgboost": {
      "status": "healthy",
      "total_requests": 1000,
      "error_rate": 0.5,
      "avg_latency_ms": 45.2,
      "requests_per_minute": 120.5
    }
  },
  "alerts": []
}
```

### Get Monitoring Statistics
```http
GET /monitoring/stats?model_name=xgboost
```

Get detailed statistics for a model or all models.

**Response:**
```json
{
  "model_name": "xgboost",
  "total_requests": 1000,
  "successful_requests": 995,
  "error_count": 5,
  "avg_latency_ms": 45.2,
  "p50_latency_ms": 42.1,
  "p95_latency_ms": 78.5,
  "p99_latency_ms": 125.3,
  "min_latency_ms": 12.5,
  "max_latency_ms": 234.7,
  "requests_per_minute": 120.5,
  "error_rate": 0.5,
  "last_updated": "2024-01-15T10:30:00"
}
```

### Check Model Drift
```http
GET /monitoring/drift/{model_name}
```

Check if model has drifted from baseline.

**Response:**
```json
{
  "drift_detected": true,
  "model_name": "xgboost",
  "current_stats": {
    "mean_prediction": 0.028,
    "std_prediction": 0.018,
    "mean_latency_ms": 52.3,
    "std_latency_ms": 15.2
  },
  "drifts": [
    {
      "type": "mean_shift",
      "severity": "medium",
      "z_score": 2.3,
      "baseline_mean": 0.023,
      "current_mean": 0.028
    }
  ]
}
```

### Set Baseline
```http
POST /monitoring/baseline
Content-Type: application/json

{
  "model_name": "xgboost",
  "mean_prediction": 0.0234,
  "std_prediction": 0.0123,
  "mean_latency_ms": 45.6,
  "std_latency_ms": 12.3
}
```

Set baseline statistics for drift detection.

## Local Testing

### Prerequisites

1. **Start the API**:
   ```bash
   python scripts/run_api.py
   ```

2. **Ensure models are trained**:
   ```bash
   python scripts/train_models.py
   ```

### Run Test Script

```bash
python scripts/test_monitoring.py
```

This script will:
1. Check API health
2. Make 50 predictions to generate monitoring data
3. View monitoring statistics
4. Check system health status
5. Check for model drift
6. Set baseline for drift detection

### Manual Testing

#### 1. Make Predictions

```bash
curl -X POST "http://localhost:8000/predict_ctr" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "ad_id": "ad_456",
    "device": "mobile",
    "placement": "header",
    "hour": 14,
    "day_of_week": 0
  }'
```

#### 2. View Health Status

```bash
curl "http://localhost:8000/monitoring/health"
```

#### 3. View Statistics

```bash
curl "http://localhost:8000/monitoring/stats"
```

Or for a specific model:
```bash
curl "http://localhost:8000/monitoring/stats?model_name=xgboost"
```

#### 4. Check Drift

```bash
curl "http://localhost:8000/monitoring/drift/xgboost"
```

#### 5. Set Baseline

```bash
curl -X POST "http://localhost:8000/monitoring/baseline" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "xgboost",
    "mean_prediction": 0.025,
    "std_prediction": 0.015,
    "mean_latency_ms": 50.0,
    "std_latency_ms": 10.0
  }'
```

## Metrics Tracked

### Per-Model Metrics

- **Total Requests**: Total number of prediction requests
- **Successful Requests**: Number of successful predictions
- **Error Count**: Number of failed predictions
- **Error Rate**: Percentage of errors
- **Average Latency**: Mean prediction time (ms)
- **P50 Latency**: Median prediction time (ms)
- **P95 Latency**: 95th percentile latency (ms)
- **P99 Latency**: 99th percentile latency (ms)
- **Min/Max Latency**: Minimum and maximum latency (ms)
- **Requests per Minute**: Current request rate

### MLflow Metrics

The following metrics are automatically logged to MLflow:
- `{model_name}_latency_ms`: Individual prediction latencies
- `{model_name}_error`: Error indicator (0 or 1)
- `{model_name}_prediction`: Prediction values
- `{model_name}_avg_latency_ms`: Aggregated average latency
- `{model_name}_p95_latency_ms`: Aggregated P95 latency
- `{model_name}_error_rate`: Aggregated error rate
- `{model_name}_requests_per_minute`: Request rate
- `{model_name}_total_requests`: Total request count

## Drift Detection

### Types of Drift Detected

1. **Mean Shift**: Significant change in prediction mean
   - Uses Z-score test (default threshold: 2.0)
   - Severity: High (>3.0) or Medium (2.0-3.0)

2. **Variance Change**: Significant change in prediction variance
   - Detects variance ratio >1.5 or <0.67
   - Severity: High (>2.0 or <0.5) or Medium

3. **Latency Anomaly**: Significant change in prediction latency
   - Uses Z-score test on latency mean
   - Helps detect performance degradation

### Setting Baselines

Baselines should be set based on:
- **Training data statistics**: Mean and std of predictions
- **Validation performance**: Expected latency from validation
- **Initial production period**: First few hours/days of production

Example baseline calculation:
```python
# From training/validation
mean_prediction = 0.0234  # Average predicted CTR
std_prediction = 0.0123   # Standard deviation
mean_latency_ms = 45.6    # Average latency
std_latency_ms = 12.3     # Latency standard deviation
```

## Health Status

### Status Levels

- **healthy**: All systems normal
- **warning**: Minor issues detected (drift, elevated latency)
- **degraded**: Significant issues (high error rate, high latency)

### Alert Conditions

- **High Error Rate**: >5% error rate → degraded
- **High Latency**: >1000ms average latency → degraded
- **Model Drift**: Detected drift → warning

## MLflow UI Usage

### Viewing Metrics

1. Start MLflow UI:
   ```bash
   mlflow ui
   ```

2. Open browser: `http://localhost:5000`

3. Select experiment: `ctr_prediction_production`

4. View metrics:
   - Click on a run
   - View "Metrics" tab
   - See real-time latency, error rates, etc.

### Comparing Models

1. Select multiple runs
2. Compare metrics side-by-side
3. View trends over time
4. Export data for analysis

### Custom Queries

Use MLflow's search API:
```python
import mlflow

client = mlflow.tracking.MlflowClient()
runs = client.search_runs(
    experiment_ids=["0"],
    filter_string="metrics.xgboost_error_rate > 1.0"
)
```

## Best Practices

### 1. Baseline Setting
- Set baselines after initial production period (1-7 days)
- Use validation set statistics as initial baseline
- Update baselines when models are retrained

### 2. Monitoring Frequency
- Check health status regularly (every 5-15 minutes)
- Review MLflow metrics daily
- Investigate alerts immediately

### 3. Drift Thresholds
- Start with conservative thresholds (2.0-3.0)
- Adjust based on business requirements
- Consider false positive rates

### 4. MLflow Tracking
- Use remote tracking server for production
- Set up automated backups
- Archive old experiments periodically

### 5. Alerting
- Set up alerts for degraded status
- Monitor error rates closely
- Track latency trends

## Integration with A/B Testing

The monitoring system integrates seamlessly with A/B testing (Phase 9.1):

- Metrics are tracked per model automatically
- A/B test results can be compared in MLflow
- Drift detection works for all models
- Health status includes A/B test models

## Troubleshooting

### MLflow Not Logging
- Check MLflow is installed: `pip install mlflow`
- Verify tracking URI is accessible
- Check API logs for MLflow errors

### No Metrics Collected
- Ensure predictions are being made
- Check monitoring manager is initialized
- Verify models are loaded

### Drift Detection Not Working
- Set baseline first: `POST /monitoring/baseline`
- Ensure sufficient data (>100 predictions)
- Check baseline statistics are reasonable

### High Memory Usage
- Reduce history window size in `DriftDetector`
- Clear old metrics periodically
- Use MLflow for long-term storage

## Next Steps

After Phase 9.2, you can:
- Set up MLflow server for production
- Create custom dashboards
- Implement alerting (email, Slack, etc.)
- Add more sophisticated drift detection
- Integrate with model retraining pipeline

## API Documentation

Full API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

All monitoring endpoints are tagged under "Monitoring".

