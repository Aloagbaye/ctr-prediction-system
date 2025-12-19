# Phase 8: Cloud Deployment (GCP)

## 📋 Overview

In Phase 8, we'll deploy the CTR Prediction System to Google Cloud Platform (GCP) using Cloud Run. This enables serverless, scalable deployment with automatic scaling and pay-per-use pricing.

**Goals:**
- Deploy API service to Cloud Run
- Deploy Streamlit UI to Cloud Run
- Store models in Cloud Storage
- Set up resource labeling for easy cleanup
- Test the deployed services

**Deliverable:** Production-ready API deployed on GCP

---

## 🎯 Learning Objectives

By the end of this phase, you'll understand:
- How to deploy containerized applications to Cloud Run
- How to use Cloud Storage for model artifacts
- How to configure environment variables in Cloud Run
- How to manage GCP resources with labels
- How to clean up resources to control costs

---

## 📦 Prerequisites

Before starting Phase 8, ensure you have:
- ✅ Completed Phase 7 (Docker containers working)
- ✅ GCP account with billing enabled
- ✅ `gcloud` CLI installed and configured
- ✅ Docker installed and running
- ✅ Trained models in `models/` directory

**Install gcloud CLI:**
- [Installation Guide](https://cloud.google.com/sdk/docs/install)

**Authenticate:**
```bash
gcloud auth login
gcloud auth configure-docker
```

---

## 🚀 Step-by-Step Guide

### Step 1: Set Up GCP Project

**Create or Select Project:**
```bash
# List existing projects
gcloud projects list

# Create new project (optional)
gcloud projects create YOUR_PROJECT_ID --name="CTR Prediction System"

# Set the project
export GCP_PROJECT_ID=YOUR_PROJECT_ID
gcloud config set project $GCP_PROJECT_ID

# Or on Windows PowerShell:
$env:GCP_PROJECT_ID = "YOUR_PROJECT_ID"
gcloud config set project $env:GCP_PROJECT_ID
```

**Enable Billing:**
- Go to [GCP Console](https://console.cloud.google.com/)
- Navigate to Billing
- Link a billing account to your project

### Step 2: Prepare for Deployment

**Check Prerequisites:**
```bash
# Verify gcloud is installed
gcloud --version

# Verify Docker is running
docker ps

# Verify models exist
ls models/*.pkl
```

**Set Environment Variables:**
```bash
# Linux/Mac
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-central1  # Optional, defaults to us-central1

# Windows PowerShell
$env:GCP_PROJECT_ID = "your-project-id"
$env:GCP_REGION = "us-central1"
```

### Step 3: Deploy to GCP

**Option A: Using Deployment Script (Recommended)**

```bash
# Linux/Mac
chmod +x scripts/gcp_deploy.sh
./scripts/gcp_deploy.sh

# Windows PowerShell
.\scripts\gcp_deploy.ps1
```

**What the Script Does:**
1. ✅ Checks prerequisites (gcloud, Docker)
2. ✅ Enables required GCP APIs
3. ✅ Creates Cloud Storage bucket for models
4. ✅ Uploads models to Cloud Storage
5. ✅ Creates Artifact Registry repository
6. ✅ Builds and pushes Docker images
7. ✅ Deploys API service to Cloud Run
8. ✅ Deploys Streamlit UI to Cloud Run
9. ✅ Configures environment variables
10. ✅ Labels all resources for easy cleanup

**Option B: Manual Deployment**

See detailed manual steps in the script comments or GCP documentation.

### Step 4: Verify Deployment

**Check Service Status:**
```bash
# List Cloud Run services
gcloud run services list --region=us-central1

# Get service URLs
gcloud run services describe ctr-prediction-api --region=us-central1 --format="value(status.url)"
gcloud run services describe ctr-prediction-ui --region=us-central1 --format="value(status.url)"
```

**Test API:**
```bash
# Get API URL
API_URL=$(gcloud run services describe ctr-prediction-api \
    --region=us-central1 \
    --format="value(status.url)")

# Health check
curl $API_URL/health

# Test prediction
curl -X POST "$API_URL/predict_ctr?model_name=xgboost" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "ad_id": "test_ad",
    "device": "mobile",
    "placement": "header",
    "hour": 14,
    "day_of_week": 0,
    "timestamp": "2024-01-01T14:00:00"
  }'
```

**Access Streamlit UI:**
- Open the Streamlit URL in your browser
- The UI should automatically connect to the API
- Test predictions through the interface

### Step 5: Monitor and Logs

**View Logs:**
```bash
# API logs
gcloud run services logs read ctr-prediction-api --region=us-central1 --limit=50

# Streamlit logs
gcloud run services logs read ctr-prediction-ui --region=us-central1 --limit=50

# Follow logs in real-time
gcloud run services logs tail ctr-prediction-api --region=us-central1
```

**View in Console:**
- Go to [Cloud Run Console](https://console.cloud.google.com/run)
- Click on service name to view metrics, logs, and configuration

---

## 🔍 Understanding the Deployment

### Cloud Run Services

**API Service:**
- **Memory**: 2GB
- **CPU**: 2 vCPU
- **Timeout**: 300 seconds
- **Scaling**: 0-10 instances (auto-scales based on traffic)
- **Environment Variables**:
  - `MODELS_DIR=/tmp/models`
  - `GCP_PROJECT_ID=<your-project>`
  - `GCS_BUCKET=<bucket-name>`

**Streamlit UI:**
- **Memory**: 1GB
- **CPU**: 1 vCPU
- **Timeout**: 300 seconds
- **Scaling**: 0-5 instances
- **Environment Variables**:
  - `API_URL=<api-service-url>`

### Cloud Storage Bucket

- **Purpose**: Store model artifacts (.pkl files)
- **Location**: Same region as Cloud Run services
- **Access**: Private (only Cloud Run services can access)
- **Naming**: `ctr-prediction-models-<project-id>`

### Artifact Registry

- **Purpose**: Store Docker images
- **Format**: Docker
- **Location**: Same region as services
- **Naming**: `ctr-prediction-repo`

### Resource Labels

All resources are labeled with:
- `managed-by=ctr-prediction-deployment`
- `deployment-date=YYYYMMDD`

This allows easy identification and cleanup.

---

## 💰 Cost Management

### Cloud Run Pricing

- **Free Tier**: 2 million requests/month, 360,000 GB-seconds, 180,000 vCPU-seconds
- **Pricing**: Pay only for what you use
- **Scaling**: Automatically scales to zero when not in use

### Cost Optimization Tips

1. **Set Min Instances to 0**: Services scale to zero when idle
2. **Use Appropriate Resources**: Don't over-provision memory/CPU
3. **Clean Up After Testing**: Use cleanup script to delete resources
4. **Monitor Usage**: Check Cloud Console for actual usage

### Estimated Costs (for testing)

- **Cloud Run**: ~$0.10-0.50/day (depending on usage)
- **Cloud Storage**: ~$0.01-0.05/day (for model storage)
- **Artifact Registry**: ~$0.01-0.02/day (for image storage)
- **Total**: ~$0.12-0.57/day for light testing

---

## 🧹 Cleanup

### Delete All Resources

**Using Cleanup Script:**
```bash
# Linux/Mac
chmod +x scripts/gcp_cleanup.sh
./scripts/gcp_cleanup.sh

# Windows PowerShell
.\scripts\gcp_cleanup.ps1
```

**What Gets Deleted:**
- ✅ Cloud Run services (API and Streamlit)
- ✅ Artifact Registry repository
- ✅ Cloud Storage bucket (and all models)

**Manual Cleanup:**
```bash
# Delete Cloud Run services
gcloud run services delete ctr-prediction-api --region=us-central1 --quiet
gcloud run services delete ctr-prediction-ui --region=us-central1 --quiet

# Delete Artifact Registry
gcloud artifacts repositories delete ctr-prediction-repo \
    --location=us-central1 --quiet

# Delete Storage bucket
gsutil rm -r gs://ctr-prediction-models-<project-id>
```

---

## 🔧 Configuration Options

### Customize Deployment

**Change Region:**
```bash
export GCP_REGION=us-east1
./scripts/gcp_deploy.sh
```

**Change Resource Prefix:**
Edit the script and change:
```bash
RESOURCE_PREFIX="your-prefix"
```

**Adjust Cloud Run Resources:**
Edit deployment script to change:
- Memory allocation
- CPU allocation
- Min/max instances
- Timeout values

### Environment Variables

**API Service:**
```bash
gcloud run services update ctr-prediction-api \
    --region=us-central1 \
    --set-env-vars="NEW_VAR=value"
```

**Streamlit Service:**
```bash
gcloud run services update ctr-prediction-ui \
    --region=us-central1 \
    --set-env-vars="API_URL=new-url"
```

---

## 🧪 Testing the Deployment

### Test API Endpoints

**Health Check:**
```bash
curl https://ctr-prediction-api-<hash>-uc.a.run.app/health
```

**Model Info:**
```bash
curl https://ctr-prediction-api-<hash>-uc.a.run.app/model_info
```

**Prediction:**
```bash
curl -X POST "https://ctr-prediction-api-<hash>-uc.a.run.app/predict_ctr?model_name=xgboost" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "ad_id": "ad_456",
    "device": "mobile",
    "placement": "header",
    "hour": 14,
    "day_of_week": 0,
    "timestamp": "2024-01-01T14:00:00"
  }'
```

### Test Streamlit UI

1. Open Streamlit URL in browser
2. Check API status in sidebar
3. Make a test prediction
4. Verify results and explanations

---

## 🔍 Common Issues & Solutions

### Issue 1: Authentication Error

**Problem**: `gcloud` not authenticated

**Solution:**
```bash
gcloud auth login
gcloud auth application-default login
```

### Issue 2: Permission Denied

**Problem**: Insufficient permissions

**Solution:**
```bash
# Grant necessary roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="user:YOUR_EMAIL" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="user:YOUR_EMAIL" \
    --role="roles/storage.admin"
```

### Issue 3: Models Not Found

**Problem**: API can't find models

**Solution:**
```bash
# Verify models uploaded to GCS
gsutil ls gs://ctr-prediction-models-<project-id>/

# Re-upload if needed
gsutil cp models/*.pkl gs://ctr-prediction-models-<project-id>/
```

### Issue 4: Image Build Fails

**Problem**: Docker build fails

**Solution:**
```bash
# Test build locally first
docker build -f Dockerfile.api -t test-api .
docker build -f Dockerfile.streamlit -t test-streamlit .

# Check Docker is running
docker ps
```

### Issue 5: Service Timeout

**Problem**: Requests timing out

**Solution:**
```bash
# Increase timeout
gcloud run services update ctr-prediction-api \
    --region=us-central1 \
    --timeout=600
```

---

## 📊 Expected Results

After completing Phase 8, you should have:

1. **Deployed Services:**
   - API service running on Cloud Run
   - Streamlit UI running on Cloud Run
   - Both services accessible via HTTPS URLs

2. **Storage:**
   - Models stored in Cloud Storage
   - Docker images in Artifact Registry

3. **Monitoring:**
   - Logs accessible in Cloud Console
   - Metrics available for monitoring

4. **Cleanup Ready:**
   - All resources labeled for easy deletion

---

## ✅ Phase 8 Checklist

- [ ] GCP account created and billing enabled
- [ ] `gcloud` CLI installed and authenticated
- [ ] GCP project created/selected
- [ ] Models trained and available locally
- [ ] Deployment script executed successfully
- [ ] API service deployed and accessible
- [ ] Streamlit UI deployed and accessible
- [ ] Services tested and working
- [ ] Logs reviewed
- [ ] Cleanup script tested (optional)

---

## 🚀 Next Steps

Once you've completed Phase 8, you're ready for:

**Phase 9: Online Evaluation & Monitoring**
- Set up monitoring dashboards
- Configure alerts
- Track prediction performance
- Monitor costs

---

## 📚 Additional Resources

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud Storage Documentation](https://cloud.google.com/storage/docs)
- [Artifact Registry Documentation](https://cloud.google.com/artifact-registry/docs)
- [gcloud CLI Reference](https://cloud.google.com/sdk/gcloud/reference)
- [Cloud Run Pricing](https://cloud.google.com/run/pricing)
- [GCP Free Tier](https://cloud.google.com/free)

---

## 💡 Tips for Success

1. **Start Small**: Test with minimal resources first
2. **Monitor Costs**: Check billing dashboard regularly
3. **Use Labels**: Always label resources for easy management
4. **Test Locally**: Verify Docker images work before deploying
5. **Clean Up**: Delete resources when not in use
6. **Read Logs**: Check logs if services don't work as expected
7. **Use Scripts**: Leverage deployment scripts for consistency

---

## 🎓 Key Concepts Learned

### 1. Serverless Computing

- **Cloud Run**: Fully managed serverless platform
- **Auto-scaling**: Automatically scales based on traffic
- **Pay-per-use**: Only pay for actual usage
- **Zero-config**: No infrastructure management needed

### 2. Container Deployment

- **Artifact Registry**: Store and manage Docker images
- **Cloud Run**: Run containers without managing servers
- **Image Building**: Build and push images to registry

### 3. Cloud Storage

- **Model Storage**: Store large model files in Cloud Storage
- **Download on Startup**: Models downloaded when service starts
- **Cost-effective**: Cheaper than keeping models in container

### 4. Resource Management

- **Labels**: Tag resources for organization
- **Cleanup**: Easy deletion of related resources
- **Cost Control**: Monitor and manage spending

---

**Congratulations on completing Phase 8!** 🎉

You now have a production-ready ML system deployed on GCP!

