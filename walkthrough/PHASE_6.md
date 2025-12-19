# Phase 6: API Development

## 📋 Overview

In Phase 6, we'll create a production-ready FastAPI service that exposes our trained models as RESTful endpoints. This enables real-time CTR predictions for ad impressions.

**Goals:**
- Create FastAPI service with prediction endpoints
- Implement input validation and error handling
- Optimize for low latency (<100ms per prediction)
- Add health checks and monitoring endpoints

**Deliverable:** Working FastAPI service running locally

---

## 🎯 Learning Objectives

By the end of this phase, you'll understand:
- How to expose ML models via REST APIs
- FastAPI framework and async request handling
- Input validation with Pydantic
- Model loading and caching strategies
- API error handling and logging

---

## 📦 Prerequisites

Before starting Phase 6, ensure you have:
- ✅ Completed Phase 3 (trained models exist in `models/` directory)
- ✅ FastAPI and uvicorn installed (`pip install fastapi uvicorn`)
- ✅ Models saved: `xgboost_model.pkl`, `logistic_model.pkl`, `lightgbm_model.pkl`

---

## 🚀 Step-by-Step Guide

### Step 1: Understand the API Structure

Our API will have the following endpoints:

```
GET  /              - API information
GET  /health        - Health check
GET  /model_info    - Model metadata
POST /predict_ctr   - Single prediction
POST /batch_predict  - Batch predictions
```

### Step 2: Start the API Server

```bash
# Start the API server
python scripts/run_api.py

# Or with custom options
python scripts/run_api.py --host 0.0.0.0 --port 8000 --reload
```

The API will start on `http://localhost:8000`

**Available URLs:**
- API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Step 3: Test the Health Endpoint

```bash
# Using curl
curl http://localhost:8000/health

# Using Python
import requests
response = requests.get("http://localhost:8000/health")
print(response.json())
```

**Expected Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-15T14:30:00",
    "version": "1.0.0"
}
```

### Step 4: Check Model Information

```bash
curl http://localhost:8000/model_info?model_name=xgboost
```

**Expected Response:**
```json
{
    "model_name": "xgboost",
    "model_type": "xgboost",
    "features": ["feature1", "feature2", ...],
    "num_features": 50,
    "loaded": true,
    "load_time": 0.45
}
```

### Step 5: Make a Single Prediction

**Using curl:**
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

**Using Python:**
```python
import requests

url = "http://localhost:8000/predict_ctr"
data = {
    "user_id": "user_123",
    "ad_id": "ad_456",
    "device": "mobile",
    "placement": "header",
    "hour": 14,
    "day_of_week": 0
}

response = requests.post(url, json=data)
print(response.json())
```

**Expected Response:**
```json
{
    "predicted_ctr": 0.0234,
    "probability": 0.0234,
    "model_name": "xgboost"
}
```

### Step 6: Make Batch Predictions

**Using Python:**
```python
import requests

url = "http://localhost:8000/batch_predict"
data = {
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

response = requests.post(url, json=data)
result = response.json()

print(f"Processed {result['total']} predictions in {result['processing_time_ms']:.2f}ms")
for pred in result['predictions']:
    print(f"CTR: {pred['predicted_ctr']:.4f}")
```

**Expected Response:**
```json
{
    "predictions": [
        {
            "predicted_ctr": 0.0234,
            "probability": 0.0234,
            "model_name": "xgboost"
        },
        {
            "predicted_ctr": 0.0156,
            "probability": 0.0156,
            "model_name": "xgboost"
        }
    ],
    "total": 2,
    "processing_time_ms": 45.2
}
```

### Step 7: Use Swagger UI for Testing

1. Open `http://localhost:8000/docs` in your browser
2. Click on an endpoint (e.g., `/predict_ctr`)
3. Click "Try it out"
4. Fill in the request body
5. Click "Execute"
6. View the response

This is great for interactive testing!

---

## 🔍 Understanding the Code

### API Structure

```
src/api/
├── __init__.py      # Package initialization
├── main.py          # FastAPI application and endpoints
├── models.py        # Pydantic request/response models
└── predictor.py     # Model loading and prediction logic
```

### Key Components

#### 1. **Pydantic Models** (`models.py`)

Define request/response schemas with validation:

```python
class PredictionRequest(BaseModel):
    user_id: str
    ad_id: str
    device: str
    placement: str
    hour: Optional[int] = Field(None, ge=0, le=23)
    # ... more fields
```

**Benefits:**
- Automatic validation
- Type checking
- API documentation generation
- Clear error messages

#### 2. **Model Predictor** (`predictor.py`)

Handles model loading and predictions:

```python
class ModelPredictor:
    def __init__(self, models_dir="models"):
        # Load models on initialization
        self.models = {}
        self.load_all_models()
    
    def predict(self, request_data, model_name=None):
        # Prepare features
        # Make prediction
        # Return results
```

**Features:**
- Lazy loading (loads models once)
- Feature engineering integration
- Error handling
- Fallback mechanisms

#### 3. **FastAPI Application** (`main.py`)

Defines endpoints and request handling:

```python
app = FastAPI(title="CTR Prediction API")

@app.post("/predict_ctr")
async def predict_ctr(request: PredictionRequest):
    # Validate input
    # Make prediction
    # Return response
```

**Features:**
- Async request handling
- Automatic OpenAPI docs
- CORS support
- Error handling

---

## ⚡ Performance Optimization

### 1. Model Loading Strategy

**Load Once, Reuse Many:**
- Models are loaded once at startup
- Stored in memory for fast access
- No reloading per request

```python
@app.on_event("startup")
async def startup_event():
    predictor.load_all_models()  # Load once
```

### 2. Feature Engineering

**Efficient Processing:**
- Use simplified feature engineering for speed
- Cache common transformations
- Batch processing for multiple requests

### 3. Async Handling

**Non-blocking Requests:**
- FastAPI uses async/await
- Multiple requests processed concurrently
- Better throughput

### 4. Response Time Targets

- **Single prediction**: <100ms
- **Batch (10 items)**: <200ms
- **Health check**: <10ms

---

## 🛡️ Error Handling

### Input Validation

Pydantic automatically validates inputs:

```python
# Invalid hour (must be 0-23)
{
    "hour": 25  # ❌ Validation error
}

# Missing required field
{
    "user_id": "user_123"
    # ❌ Missing 'ad_id'
}
```

### Error Responses

```json
{
    "detail": "Model xgboost not loaded. Available: ['logistic']"
}
```

**HTTP Status Codes:**
- `200`: Success
- `400`: Bad request (validation error)
- `404`: Not found
- `500`: Internal server error
- `503`: Service unavailable (no models loaded)

---

## 📊 Monitoring & Logging

### Logging

The API logs:
- Model loading status
- Prediction requests
- Errors and exceptions
- Performance metrics

**Log Format:**
```
2024-01-15 14:30:00 - INFO - Loaded xgboost model in 0.45s
2024-01-15 14:30:05 - INFO - Prediction completed in 45.2ms
```

### Health Checks

Monitor API health:
```bash
curl http://localhost:8000/health
```

**Status Values:**
- `healthy`: Models loaded, API working
- `degraded`: Some models missing, API partially working

---

## 🧪 Testing the API

### Manual Testing

1. **Start the server:**
   ```bash
   python scripts/run_api.py
   ```

2. **Test health:**
   ```bash
   curl http://localhost:8000/health
   ```

3. **Test prediction:**
   ```bash
   curl -X POST http://localhost:8000/predict_ctr \
     -H "Content-Type: application/json" \
     -d @test_request.json
   ```

### Using Python Requests

```python
import requests
import time

# Test single prediction
start = time.time()
response = requests.post(
    "http://localhost:8000/predict_ctr",
    json={
        "user_id": "user_123",
        "ad_id": "ad_456",
        "device": "mobile",
        "placement": "header"
    }
)
latency = (time.time() - start) * 1000
print(f"Latency: {latency:.2f}ms")
print(f"Response: {response.json()}")
```

### Load Testing

```python
import requests
import concurrent.futures

def make_prediction():
    response = requests.post(
        "http://localhost:8000/predict_ctr",
        json={
            "user_id": "user_123",
            "ad_id": "ad_456",
            "device": "mobile",
            "placement": "header"
        }
    )
    return response.json()

# Test with 100 concurrent requests
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(make_prediction) for _ in range(100)]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]

print(f"Completed {len(results)} predictions")
```

---

## 🔍 Common Issues & Solutions

### Issue 1: Models Not Found

**Problem:** `503 Service Unavailable - No models loaded`

**Solution:**
```bash
# Train models first
python scripts/train_models.py \
  --input data/processed/features.csv \
  --output-dir models/
```

### Issue 2: Port Already in Use

**Problem:** `Address already in use`

**Solution:**
```bash
# Use a different port
python scripts/run_api.py --port 8001

# Or find and kill the process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8000 | xargs kill
```

### Issue 3: Feature Mismatch

**Problem:** Prediction fails with feature mismatch error

**Solution:**
- Ensure feature engineering pipeline matches training
- Check that `feature_names.pkl` exists
- Verify feature order matches model expectations

### Issue 4: Slow Predictions

**Problem:** Predictions take >100ms

**Solutions:**
- Use simpler model (logistic regression)
- Reduce feature engineering complexity
- Enable model caching
- Use batch predictions for multiple requests

---

## 📊 Expected Results

After completing Phase 6, you should have:

1. **Working API Server:**
   - Running on `http://localhost:8000`
   - All endpoints functional
   - Models loaded and ready

2. **Performance Metrics:**
   - Single prediction: <100ms
   - Health check: <10ms
   - Batch prediction: <200ms for 10 items

3. **API Documentation:**
   - Swagger UI at `/docs`
   - ReDoc at `/redoc`
   - OpenAPI schema available

---

## ✅ Phase 6 Checklist

- [ ] API server starts successfully
- [ ] Health endpoint returns `healthy` status
- [ ] Model info endpoint shows loaded models
- [ ] Single prediction endpoint works
- [ ] Batch prediction endpoint works
- [ ] Input validation catches errors
- [ ] Error handling returns proper status codes
- [ ] Swagger UI accessible and functional
- [ ] Predictions complete in <100ms
- [ ] Ready for Phase 7 (Containerization)

---

## 🚀 Next Steps

Once you've completed Phase 6, you're ready for:

**Phase 7: Containerization**
- Create Dockerfile
- Build Docker image
- Test containerized API
- Prepare for cloud deployment

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [REST API Best Practices](https://restfulapi.net/)
- [API Design Guide](https://cloud.google.com/apis/design)

---

## 💡 Tips for Success

1. **Test Incrementally**: Test each endpoint before moving to the next
2. **Use Swagger UI**: Great for interactive testing and documentation
3. **Monitor Logs**: Watch for errors and performance issues
4. **Validate Inputs**: Always validate user inputs
5. **Handle Errors**: Provide clear error messages
6. **Optimize Performance**: Profile and optimize slow endpoints

---

**Congratulations on completing Phase 6!** 🎉

You now have a production-ready API service that can serve CTR predictions in real-time.

