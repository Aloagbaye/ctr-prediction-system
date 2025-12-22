# Phase 9.1: A/B Testing Setup

## Overview

This phase implements A/B testing capabilities for comparing different model versions in production. The system supports:

- **Traffic Splitting**: Consistent routing of users to different model variants
- **Performance Tracking**: Real-time metrics collection (latency, error rates)
- **Statistical Comparison**: Automated comparison between model variants
- **Test Management**: Create, monitor, and manage A/B tests via API

## Features

### 1. Consistent Traffic Splitting
- Uses consistent hashing on `user_id` to ensure the same user always gets the same model
- Configurable traffic split (e.g., 50/50, 80/20, 90/10)
- Automatic expiration support for time-limited tests

### 2. Performance Metrics
- **Latency**: Average, min, max prediction times
- **Error Rates**: Track failures per model
- **Request Volume**: Total requests per variant
- **Real-time Updates**: Metrics updated after each prediction

### 3. Test Management
- Create tests with custom configurations
- Monitor test statistics in real-time
- Stop tests without deleting data
- Delete tests and clean up metrics

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       │ POST /predict_ctr?ab_test_id=test_123
       ▼
┌─────────────────────────────────┐
│      FastAPI Application        │
│  ┌───────────────────────────┐ │
│  │   ABTestManager           │ │
│  │  - Traffic Routing        │ │
│  │  - Metrics Collection     │ │
│  │  - Test Management        │ │
│  └───────────────────────────┘ │
│              │                 │
│              ▼                 │
│  ┌───────────────────────────┐ │
│  │   ModelPredictor          │ │
│  │  - Model Loading          │ │
│  │  - Feature Engineering    │ │
│  │  - Prediction            │ │
│  └───────────────────────────┘ │
└─────────────────────────────────┘
```

## API Endpoints

### Create A/B Test
```http
POST /ab_test/create
Content-Type: application/json

{
  "test_id": "xgboost_vs_lightgbm",
  "model_a": "xgboost",
  "model_b": "lightgbm",
  "traffic_split": 0.5,
  "duration_hours": 24,
  "description": "Compare XGBoost vs LightGBM performance"
}
```

### List All Tests
```http
GET /ab_test/list
```

### Get Test Statistics
```http
GET /ab_test/stats/{test_id}
```

### Stop Test
```http
POST /ab_test/stop/{test_id}
```

### Delete Test
```http
DELETE /ab_test/{test_id}
```

### Make Prediction with A/B Test
```http
POST /predict_ctr?ab_test_id={test_id}
Content-Type: application/json

{
  "user_id": "user_123",
  "ad_id": "ad_456",
  "device": "mobile",
  "placement": "header",
  "hour": 14,
  "day_of_week": 0
}
```

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

The provided test script demonstrates all A/B testing features:

```bash
python scripts/test_ab_testing.py
```

This script will:
1. Check API health
2. Verify available models
3. Create an A/B test
4. Make 100 predictions with traffic splitting
5. Display performance statistics
6. Show comparison metrics

### Manual Testing

#### 1. Create an A/B Test

```bash
curl -X POST "http://localhost:8000/ab_test/create" \
  -H "Content-Type: application/json" \
  -d '{
    "test_id": "test_xgboost_vs_lightgbm",
    "model_a": "xgboost",
    "model_b": "lightgbm",
    "traffic_split": 0.5,
    "description": "Compare XGBoost vs LightGBM"
  }'
```

#### 2. Make Predictions with A/B Test

```bash
curl -X POST "http://localhost:8000/predict_ctr?ab_test_id=test_xgboost_vs_lightgbm" \
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

**Note**: The same `user_id` will always get routed to the same model variant.

#### 3. View Test Statistics

```bash
curl "http://localhost:8000/ab_test/stats/test_xgboost_vs_lightgbm"
```

Response includes:
- Total requests per model
- Average latency comparison
- Error rates
- Performance improvement percentages

#### 4. List All Tests

```bash
curl "http://localhost:8000/ab_test/list"
```

#### 5. Stop a Test

```bash
curl -X POST "http://localhost:8000/ab_test/stop/test_xgboost_vs_lightgbm"
```

#### 6. Delete a Test

```bash
curl -X DELETE "http://localhost:8000/ab_test/test_xgboost_vs_lightgbm"
```

## Traffic Splitting Algorithm

The system uses **consistent hashing** to ensure:
- Same user → Same model (consistency)
- Traffic split matches configured ratio (fairness)
- Deterministic routing (reproducibility)

```python
# Hash user_id + test_id
hash_value = md5(f"{test_id}:{user_id}")

# Map to [0, 1)
hash_ratio = (hash_value % 10000) / 10000.0

# Route based on traffic_split
if hash_ratio < traffic_split:
    return model_b  # Variant
else:
    return model_a  # Control
```

## Metrics Collected

### Per-Model Metrics
- `total_requests`: Total number of requests
- `total_predictions`: Successful predictions
- `avg_prediction_time_ms`: Average latency
- `min_prediction_time_ms`: Minimum latency
- `max_prediction_time_ms`: Maximum latency
- `error_count`: Number of errors
- `last_updated`: Last update timestamp

### Comparison Metrics
- Traffic distribution percentages
- Latency difference and improvement percentage
- Error rate comparison
- Total request counts

## Data Storage

A/B test data is stored locally in:
- `data/ab_tests/tests.json`: Test configurations
- `data/ab_tests/metrics.json`: Performance metrics

Both files are automatically created and updated by the `ABTestManager`.

## Best Practices

### 1. Test Duration
- Start with short tests (1-24 hours) to validate setup
- Extend duration based on traffic volume
- Monitor metrics regularly

### 2. Traffic Split
- **50/50**: Equal comparison, fastest statistical significance
- **80/20**: Test new model with lower risk
- **90/10**: Very conservative testing
- **10/90**: Gradual rollout

### 3. Model Selection
- Compare models with similar feature sets
- Ensure both models are properly trained
- Test on representative traffic

### 4. Monitoring
- Check statistics regularly during test
- Look for:
  - Latency differences
  - Error rate spikes
  - Traffic distribution accuracy
  - Statistical significance

### 5. Decision Making
- Consider multiple metrics:
  - Latency (lower is better)
  - Error rate (lower is better)
  - Prediction quality (if ground truth available)
- Run tests long enough for statistical significance
- Document decisions and rationale

## Example Workflow

1. **Train Models**: Ensure both models are trained and loaded
2. **Create Test**: Create A/B test with desired configuration
3. **Monitor**: Make predictions and monitor statistics
4. **Analyze**: Compare performance metrics
5. **Decide**: Choose winning model or continue testing
6. **Stop/Delete**: Stop test and clean up if needed

## Troubleshooting

### Test Not Routing Traffic
- Verify test is enabled: `GET /ab_test/list`
- Check test hasn't expired
- Ensure both models are loaded

### Inconsistent Routing
- Same `user_id` should always get same model
- If not, check hash function consistency
- Verify `test_id` is consistent

### Missing Metrics
- Metrics are recorded after each prediction
- Check that predictions are being made with `ab_test_id` parameter
- Verify API logs for errors

### Models Not Found
- Ensure models are trained: `python scripts/train_models.py`
- Check model names match exactly (case-sensitive)
- Verify models are loaded: `GET /model_info`

## Next Steps

After Phase 9.1, you can proceed to:
- **Phase 9.2**: Real-time Monitoring Dashboard
- Enhanced metrics visualization
- Alerting and notifications
- Historical trend analysis

## API Documentation

Full API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

All A/B testing endpoints are tagged under "A/B Testing" in the documentation.

