# Phase 9.1: A/B Testing Setup - Implementation Summary

## ✅ What Was Implemented

### Core Components

1. **`src/api/ab_test_manager.py`** - A/B Testing Manager
   - Traffic splitting with consistent hashing
   - Performance metrics tracking
   - Test configuration management
   - Persistent storage (JSON files)

2. **Updated `src/api/main.py`**
   - New A/B testing endpoints
   - Integrated A/B test routing in prediction endpoint
   - Metrics recording

3. **Updated `src/api/models.py`**
   - New Pydantic models for A/B testing
   - Request/response validation

4. **`scripts/test_ab_testing.py`** - Local Testing Script
   - Comprehensive test suite
   - Demonstrates all features
   - Easy to run and understand

5. **`src/app/ab_test_dashboard.py`** - Streamlit Dashboard
   - Visual A/B test comparison
   - Real-time metrics display
   - Interactive charts and tables
   - Statistical summaries

6. **`walkthrough/PHASE_9_1.md`** - Documentation
   - Complete guide
   - API reference
   - Best practices

## 🚀 Quick Start

### 1. Start the API
```bash
python scripts/run_api.py
```

### 2. Run the Test Script
```bash
python scripts/test_ab_testing.py
```

This will:
- ✅ Check API health
- ✅ Verify models are loaded
- ✅ Create an A/B test
- ✅ Make 100 predictions with traffic splitting
- ✅ Display performance statistics
- ✅ Show comparison metrics

### 3. View Dashboard (Optional)
```bash
python scripts/run_ab_test_dashboard.py
```

Then open http://localhost:8502 in your browser to see:
- ✅ Visual comparison tables
- ✅ Interactive charts (latency, error rates, traffic distribution)
- ✅ Statistical summaries
- ✅ Real-time test status

## 📊 Features

### Traffic Splitting
- Consistent routing (same user → same model)
- Configurable split ratios (0.0 to 1.0)
- Automatic expiration support

### Performance Tracking
- Latency metrics (avg, min, max)
- Error rate tracking
- Request volume counting
- Real-time updates

### Test Management
- Create/stop/delete tests via API
- View statistics in real-time
- Persistent storage

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ab_test/create` | POST | Create new A/B test |
| `/ab_test/list` | GET | List all tests |
| `/ab_test/stats/{test_id}` | GET | Get test statistics |
| `/ab_test/stop/{test_id}` | POST | Stop a test |
| `/ab_test/{test_id}` | DELETE | Delete a test |
| `/predict_ctr?ab_test_id=...` | POST | Make prediction with A/B test |

## 📝 Example Usage

### Create Test
```bash
curl -X POST "http://localhost:8000/ab_test/create" \
  -H "Content-Type: application/json" \
  -d '{
    "test_id": "test_1",
    "model_a": "xgboost",
    "model_b": "lightgbm",
    "traffic_split": 0.5
  }'
```

### Make Prediction
```bash
curl -X POST "http://localhost:8000/predict_ctr?ab_test_id=test_1" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "ad_id": "ad_456",
    "device": "mobile",
    "placement": "header"
  }'
```

### View Statistics
```bash
curl "http://localhost:8000/ab_test/stats/test_1"
```

## 📁 Files Created/Modified

### New Files
- `src/api/ab_test_manager.py` - Core A/B testing logic
- `scripts/test_ab_testing.py` - Test script
- `walkthrough/PHASE_9_1.md` - Documentation

### Modified Files
- `src/api/main.py` - Added A/B testing endpoints
- `src/api/models.py` - Added A/B testing models

## 🧪 Testing Locally

1. **Ensure API is running**:
   ```bash
   python scripts/run_api.py
   ```

2. **Ensure models are trained**:
   ```bash
   python scripts/train_models.py
   ```

3. **Run test script**:
   ```bash
   python scripts/test_ab_testing.py
   ```

4. **Or use API directly**:
   - Visit `http://localhost:8000/docs` for interactive API docs
   - All A/B testing endpoints are under "A/B Testing" tag

## 📊 What Gets Tracked

- **Per Model**:
  - Total requests
  - Successful predictions
  - Average latency (ms)
  - Min/Max latency (ms)
  - Error count
  - Last update time

- **Comparison**:
  - Traffic distribution
  - Latency difference
  - Performance improvement %
  - Error rate comparison

## 🎨 Streamlit Dashboard

### Features

The A/B Test Dashboard provides a visual interface for viewing test results:

- **Test Overview**: Status, duration, total requests
- **Comparison Table**: Side-by-side metrics for Model A and Model B
- **Visualizations**:
  - Latency comparison bar chart
  - Error rate comparison
  - Traffic distribution pie chart
- **Statistical Summary**: Key metrics and improvements
- **Real-time Updates**: Refresh to see latest data

### Usage

```bash
# Start the dashboard
python scripts/run_ab_test_dashboard.py

# Or with custom port
python scripts/run_ab_test_dashboard.py --port 8503

# Or with custom API URL
python scripts/run_ab_test_dashboard.py --api-url http://localhost:8000
```

Then open http://localhost:8502 in your browser.

### Dashboard Sections

1. **Header**: Test status, duration, total requests
2. **Test Information**: Test ID, models, traffic split, timestamps
3. **Model Comparison**: Detailed metrics table
4. **Visualizations**: Interactive charts
5. **Statistical Summary**: Key improvements and differences
6. **Raw Data**: JSON view of all test data

## 💾 Data Storage

Data is stored in:
- `data/ab_tests/tests.json` - Test configurations
- `data/ab_tests/metrics.json` - Performance metrics

Both are automatically created and updated.

## 📊 MLflow Integration

A/B test experiments are automatically logged to MLflow for tracking and analysis:

### Features:
- **Automatic Experiment Tracking**: Each A/B test creates a dedicated MLflow run
- **Configuration Logging**: Test parameters (models, traffic split, description) logged as MLflow parameters
- **Real-time Metrics**: Performance metrics logged periodically (every 50 requests)
- **Comparison Metrics**: Latency differences, error rates, and improvement percentages logged
- **Model-specific Metrics**: Separate metrics for Model A and Model B

### MLflow Experiment:
- **Experiment Name**: `ctr_prediction_ab_tests`
- **Tracking URI**: Same as monitoring system (defaults to local `mlruns/` directory)

### Viewing Results:
```bash
# Start MLflow UI
mlflow ui

# Open in browser
http://localhost:5000

# Select experiment: "ctr_prediction_ab_tests"
```

### Logged Metrics:
- `{model_name}_total_requests` - Total requests per model
- `{model_name}_avg_latency_ms` - Average prediction latency
- `{model_name}_error_rate` - Error rate percentage
- `latency_diff_ms` - Difference in latency between models
- `latency_improvement_pct` - Performance improvement percentage
- `total_requests` - Total requests across both models
- `error_rate_a` / `error_rate_b` - Error rates for each model

### Tags:
- `test_type`: "ab_test"
- `test_id`: Unique test identifier
- `model_a`: Control model name
- `model_b`: Variant model name
- `start_time`: Test start timestamp

## 🎯 Next Steps

After testing Phase 9.1 locally, you can:
1. Deploy to production (GCP Cloud Run)
2. Integrate with monitoring dashboard
3. Add alerting for significant differences
4. Implement Phase 9.2 (Real-time Monitoring)

## 📚 Documentation

Full documentation: `walkthrough/PHASE_9_1.md`

API docs: `http://localhost:8000/docs`

