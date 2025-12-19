# Phase 8.2: Vertex AI & BigQuery Integration

## 📋 Overview

In Phase 8.2, we'll extend the GCP deployment to use Vertex AI for model management and BigQuery for data storage and analytics. This provides a complete MLOps pipeline with model versioning, training pipelines, and data analytics.

**Goals:**
- Upload models to Vertex AI Model Registry
- Set up BigQuery for prediction storage and analytics
- Create Vertex AI training pipeline (optional)
- Deploy models to Vertex AI Endpoints (alternative to Cloud Run)
- Create analytics dashboards with BigQuery

**Deliverable:** Complete MLOps pipeline with Vertex AI and BigQuery

---

## 🎯 Learning Objectives

By the end of this phase, you'll understand:
- How to use Vertex AI Model Registry for model versioning
- How to store predictions and data in BigQuery
- How to create Vertex AI training pipelines
- How to query and analyze data in BigQuery
- How to integrate Vertex AI with Cloud Run services

---

## 📦 Prerequisites

Before starting Phase 8.2, ensure you have:
- ✅ Completed Phase 8 (Cloud Run deployment)
- ✅ GCP project with billing enabled
- ✅ Vertex AI API enabled
- ✅ BigQuery API enabled
- ✅ Trained models in `models/` directory

**Enable Required APIs:**
```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable bigquery.googleapis.com
```

---

## 🚀 Step-by-Step Guide

### Step 1: Set Up Vertex AI

**Create Vertex AI Dataset:**
```bash
# Set variables
export PROJECT_ID=your-project-id
export REGION=us-central1
export DATASET_NAME=ctr_predictions

# Create dataset in Vertex AI (for training data)
gcloud ai datasets create \
    --display-name="CTR Prediction Dataset" \
    --metadata-schema-uri=gs://google-cloud-aiplatform/schema/dataset/metadata/tabular_1.0.0.yaml \
    --region=$REGION \
    --project=$PROJECT_ID
```

### Step 2: Upload Models to Vertex AI Model Registry

**Using the Upload Script:**
```bash
# Linux/Mac
chmod +x scripts/vertex_ai_upload.sh
./scripts/vertex_ai_upload.sh

# Windows PowerShell
.\scripts\vertex_ai_upload.ps1
```

**Manual Upload:**
```bash
# Upload a model
gcloud ai models upload \
    --region=us-central1 \
    --display-name="CTR Prediction XGBoost" \
    --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest \
    --artifact-uri=gs://your-bucket/models/xgboost_model.pkl
```

### Step 3: Set Up BigQuery

**Create BigQuery Dataset:**
```bash
# Create dataset for predictions
bq mk --dataset \
    --location=US \
    --description="CTR Prediction System Data" \
    $PROJECT_ID:ctr_predictions

# Create table for predictions
bq mk --table \
    --description="Historical CTR Predictions" \
    $PROJECT_ID:ctr_predictions.predictions \
    user_id:STRING,ad_id:STRING,device:STRING,placement:STRING, \
    predicted_ctr:FLOAT,actual_ctr:FLOAT,timestamp:TIMESTAMP, \
    model_name:STRING,request_id:STRING
```

**Using the Setup Script:**
```bash
# Linux/Mac
chmod +x scripts/bigquery_setup.sh
./scripts/bigquery_setup.sh

# Windows PowerShell
.\scripts\bigquery_setup.ps1
```

### Step 4: Update API to Store Predictions in BigQuery

The API will automatically log predictions to BigQuery when configured.

**Configure Environment Variables:**
```bash
# Update Cloud Run service
gcloud run services update ctr-prediction-api \
    --region=us-central1 \
    --set-env-vars="BIGQUERY_DATASET=ctr_predictions,BIGQUERY_TABLE=predictions"
```

### Step 5: Create Vertex AI Training Pipeline (Optional)

**Using Training Pipeline Script:**
```bash
# Linux/Mac
chmod +x scripts/vertex_ai_training.sh
./scripts/vertex_ai_training.sh

# Windows PowerShell
.\scripts\vertex_ai_training.ps1
```

---

## 🔍 Understanding the Components

### Vertex AI Model Registry

**Benefits:**
- Model versioning and management
- Model metadata and lineage
- Easy model deployment
- Model monitoring

**Use Cases:**
- Store multiple model versions
- Track model performance over time
- Deploy models to Vertex AI Endpoints
- A/B testing different model versions

### BigQuery

**Benefits:**
- Scalable data warehouse
- Fast SQL queries
- Integration with Vertex AI
- Analytics and reporting

**Use Cases:**
- Store historical predictions
- Store training data
- Analytics queries
- Create dashboards

### Integration Flow

```
Training Data → BigQuery → Vertex AI Training → Model Registry
                                                      ↓
Predictions ← Cloud Run API ← Vertex AI Endpoint ← Model Version
     ↓
BigQuery (Storage & Analytics)
```

---

## 📊 BigQuery Analytics

### Query Historical Predictions

```sql
-- Average CTR by device
SELECT 
    device,
    AVG(predicted_ctr) as avg_predicted_ctr,
    AVG(actual_ctr) as avg_actual_ctr,
    COUNT(*) as prediction_count
FROM `ctr_predictions.predictions`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
GROUP BY device
ORDER BY avg_predicted_ctr DESC;
```

### Model Performance Comparison

```sql
-- Compare model performance
SELECT 
    model_name,
    AVG(predicted_ctr) as avg_prediction,
    AVG(ABS(predicted_ctr - actual_ctr)) as mae,
    COUNT(*) as total_predictions
FROM `ctr_predictions.predictions`
WHERE actual_ctr IS NOT NULL
GROUP BY model_name
ORDER BY mae ASC;
```

### Time Series Analysis

```sql
-- Predictions over time
SELECT 
    TIMESTAMP_TRUNC(timestamp, HOUR) as hour,
    AVG(predicted_ctr) as avg_ctr,
    COUNT(*) as request_count
FROM `ctr_predictions.predictions`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
GROUP BY hour
ORDER BY hour;
```

---

## 🔧 Configuration

### Vertex AI Configuration

**Model Registry Settings:**
- Region: `us-central1` (or your preferred region)
- Model naming: `ctr-prediction-{model-type}-v{version}`
- Container images: Use pre-built Vertex AI containers

**Training Pipeline Settings:**
- Machine type: `n1-standard-4`
- Training job timeout: 24 hours
- Hyperparameter tuning: Optional

### BigQuery Configuration

**Dataset Settings:**
- Location: `US` (or your preferred location)
- Default table expiration: 90 days (optional)
- Default partition expiration: 30 days (optional)

**Table Schema:**
- Partition by `timestamp` for better query performance
- Cluster by `model_name` and `device` for common queries

---

## 🧪 Testing

### Test Model Upload

```bash
# List uploaded models
gcloud ai models list --region=us-central1

# Get model details
gcloud ai models describe MODEL_ID --region=us-central1
```

### Test BigQuery Integration

```bash
# Query predictions
bq query --use_legacy_sql=false \
    "SELECT COUNT(*) FROM \`$PROJECT_ID.ctr_predictions.predictions\`"

# Check recent predictions
bq query --use_legacy_sql=false \
    "SELECT * FROM \`$PROJECT_ID.ctr_predictions.predictions\` 
     ORDER BY timestamp DESC LIMIT 10"
```

### Test Prediction Logging

```bash
# Make a prediction (should be logged to BigQuery)
curl -X POST "https://your-api-url/predict_ctr?model_name=xgboost" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "ad_id": "test_ad",
    "device": "mobile",
    "placement": "header",
    "hour": 14,
    "day_of_week": 0
  }'

# Verify in BigQuery
bq query --use_legacy_sql=false \
    "SELECT * FROM \`$PROJECT_ID.ctr_predictions.predictions\` 
     WHERE user_id = 'test_user'"
```

---

## 💰 Cost Considerations

### Vertex AI Costs

- **Model Registry**: Free (storage costs apply)
- **Training**: ~$0.50-2.00 per hour (depending on machine type)
- **Prediction Endpoints**: ~$0.10-0.50 per 1000 predictions

### BigQuery Costs

- **Storage**: $0.02 per GB/month
- **Queries**: $5 per TB processed (first 1 TB free per month)
- **Streaming Inserts**: $0.01 per 200 MB

### Cost Optimization

1. **Use Partitioning**: Reduces query costs
2. **Set Table Expiration**: Auto-delete old data
3. **Use Clustering**: Improves query performance
4. **Monitor Usage**: Set up billing alerts

---

## 🔍 Common Issues & Solutions

### Issue 1: Vertex AI API Not Enabled

**Problem**: `aiplatform.googleapis.com` not enabled

**Solution:**
```bash
gcloud services enable aiplatform.googleapis.com
```

### Issue 2: BigQuery Permission Denied

**Problem**: Service account lacks BigQuery permissions

**Solution:**
```bash
# Get service account
SERVICE_ACCOUNT=$(gcloud run services describe ctr-prediction-api \
    --region=us-central1 --format="value(spec.template.spec.serviceAccountName)")

# Grant BigQuery permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/bigquery.dataEditor"
```

### Issue 3: Model Upload Fails

**Problem**: Container image not found

**Solution:**
- Use pre-built Vertex AI container images
- Or build custom container and push to Artifact Registry

### Issue 4: BigQuery Streaming Inserts Fail

**Problem**: Rate limits or quota exceeded

**Solution:**
- Use batch inserts instead of streaming
- Implement retry logic with exponential backoff
- Consider using BigQuery Storage Write API

---

## 📊 Expected Results

After completing Phase 8.2, you should have:

1. **Vertex AI Setup:**
   - Models uploaded to Model Registry
   - Model versions tracked
   - Training pipeline configured (optional)

2. **BigQuery Setup:**
   - Dataset created
   - Prediction table created
   - Historical data stored

3. **Integration:**
   - API logging predictions to BigQuery
   - Analytics queries working
   - Model performance tracking

---

## ✅ Phase 8.2 Checklist

- [ ] Vertex AI API enabled
- [ ] BigQuery API enabled
- [ ] Models uploaded to Vertex AI Model Registry
- [ ] BigQuery dataset and tables created
- [ ] API updated to log predictions
- [ ] Test predictions logged to BigQuery
- [ ] Analytics queries tested
- [ ] Model versioning working
- [ ] Cost monitoring set up

---

## 🚀 Next Steps

Once you've completed Phase 8.2, you can:

1. **Create Dashboards**: Use BigQuery data in Data Studio
2. **Set Up Alerts**: Monitor model performance
3. **A/B Testing**: Compare model versions
4. **Retraining Pipeline**: Automate model retraining

---

## 📚 Additional Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [BigQuery Documentation](https://cloud.google.com/bigquery/docs)
- [Vertex AI Model Registry](https://cloud.google.com/vertex-ai/docs/model-registry)
- [BigQuery Best Practices](https://cloud.google.com/bigquery/docs/best-practices)
- [MLOps on GCP](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

---

## 💡 Tips for Success

1. **Start Small**: Upload one model first, then expand
2. **Use Partitioning**: Partition BigQuery tables by timestamp
3. **Monitor Costs**: Set up billing alerts
4. **Version Models**: Always version your models in Vertex AI
5. **Test Queries**: Test BigQuery queries before production
6. **Use Clustering**: Cluster tables for better performance
7. **Set Expiration**: Auto-delete old data to save costs

---

## 🎓 Key Concepts Learned

### 1. Model Management

- **Vertex AI Model Registry**: Centralized model storage
- **Model Versioning**: Track model iterations
- **Model Metadata**: Store model information

### 2. Data Warehousing

- **BigQuery**: Scalable data warehouse
- **Partitioning**: Improve query performance
- **Clustering**: Optimize common queries

### 3. MLOps Integration

- **End-to-End Pipeline**: Training → Registry → Deployment → Monitoring
- **Data Lineage**: Track data flow
- **Model Monitoring**: Track model performance

---

**Congratulations on completing Phase 8.2!** 🎉

You now have a complete MLOps pipeline with Vertex AI and BigQuery!

