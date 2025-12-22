# GCP Cloud Run Deployment Steps

## Artifacts Prepared

### Images Pushed to Artifact Registry:
- **API Image**: `us-central1-docker.pkg.dev/ctr-prediction-system/ctr-prediction-repo/api:latest`
- **Streamlit Image**: `us-central1-docker.pkg.dev/ctr-prediction-system/ctr-prediction-repo/streamlit:latest`

### Models Uploaded to Cloud Storage:
- **Bucket**: `gs://ctr-prediction-models-ctr-prediction-system/`

---

## Deployment via Google Cloud Console

### Step 1: Navigate to Cloud Run
Go to: **https://console.cloud.google.com/run?project=ctr-prediction-system**

### Step 2: Deploy API Service

1. Click **"CREATE SERVICE"**

2. **Basic Settings:**
   - **Service name**: `ctr-prediction-api`
   - **Region**: `us-central1`
   - **Allow unauthenticated invocations**: ✅ **Yes**

3. **Container Settings:**
   - **Container image URL**: `us-central1-docker.pkg.dev/ctr-prediction-system/ctr-prediction-repo/api:latest`
   - Click **"SELECT"** to verify the image

4. **Container, Networking, Security:**
   - Expand **"Container, Networking, Security"** section
   - **Memory**: `2 GiB`
   - **CPU**: `2`
   - **Timeout**: `300 seconds`
   - **Max instances**: `10`
   - **Min instances**: `0` (optional, for cost savings)

5. **Environment Variables:**
   Click **"ADD VARIABLE"** for each:
   - **Name**: `MODELS_DIR` → **Value**: `/tmp/models`
   - **Name**: `GCP_PROJECT_ID` → **Value**: `ctr-prediction-system`
   - **Name**: `GCS_BUCKET` → **Value**: `ctr-prediction-models-ctr-prediction-system`

6. Click **"CREATE"** and wait for deployment (2-5 minutes)

7. **After deployment**, copy the **Service URL** (e.g., `https://ctr-prediction-api-xxxxx-uc.a.run.app`)

---

### Step 3: Deploy Streamlit UI Service

1. Click **"CREATE SERVICE"** again

2. **Basic Settings:**
   - **Service name**: `ctr-prediction-ui`
   - **Region**: `us-central1`
   - **Allow unauthenticated invocations**: ✅ **Yes**

3. **Container Settings:**
   - **Container image URL**: `us-central1-docker.pkg.dev/ctr-prediction-system/ctr-prediction-repo/streamlit:latest`
   - Click **"SELECT"** to verify the image

4. **Container, Networking, Security:**
   - Expand **"Container, Networking, Security"** section
   - **Memory**: `1 GiB`
   - **CPU**: `1`
   - **Timeout**: `300 seconds`
   - **Max instances**: `5`
   - **Min instances**: `0` (optional, for cost savings)

5. **Environment Variables:**
   Click **"ADD VARIABLE"**:
   - **Name**: `API_URL` → **Value**: `<API_SERVICE_URL>` 
     - Replace `<API_SERVICE_URL>` with the URL you copied from Step 2 (e.g., `https://ctr-prediction-api-xxxxx-uc.a.run.app`)

6. Click **"CREATE"** and wait for deployment (2-5 minutes)

---

## Summary of Values Used

| Variable | Value |
|----------|-------|
| **PROJECT_ID** | `ctr-prediction-system` |
| **REGION** | `us-central1` |
| **RESOURCE_PREFIX** | `ctr-prediction` |
| **BUCKET_NAME** | `ctr-prediction-models-ctr-prediction-system` |
| **ARTIFACT_REGISTRY** | `us-central1-docker.pkg.dev/ctr-prediction-system/ctr-prediction-repo` |
| **API_SERVICE_NAME** | `ctr-prediction-api` |
| **UI_SERVICE_NAME** | `ctr-prediction-ui` |
| **API_IMAGE** | `us-central1-docker.pkg.dev/ctr-prediction-system/ctr-prediction-repo/api:latest` |
| **STREAMLIT_IMAGE** | `us-central1-docker.pkg.dev/ctr-prediction-system/ctr-prediction-repo/streamlit:latest` |

---

## Environment Variables Summary

### API Service:
- `MODELS_DIR` = `/tmp/models`
- `GCP_PROJECT_ID` = `ctr-prediction-system`
- `GCS_BUCKET` = `ctr-prediction-models-ctr-prediction-system`

### Streamlit UI Service:
- `API_URL` = `<API_SERVICE_URL>` (use the URL from API service after deployment)

---

## Verification

After both services are deployed, you can verify them:

1. **List services:**
   ```powershell
   gcloud run services list --region=us-central1 --project=ctr-prediction-system
   ```

2. **Get API URL:**
   ```powershell
   gcloud run services describe ctr-prediction-api --region=us-central1 --project=ctr-prediction-system --format="value(status.url)"
   ```

3. **Get UI URL:**
   ```powershell
   gcloud run services describe ctr-prediction-ui --region=us-central1 --project=ctr-prediction-system --format="value(status.url)"
   ```

4. **Test API:**
   - Visit: `<API_URL>/docs` for Swagger UI
   - Visit: `<API_URL>/health` for health check

5. **Test Streamlit UI:**
   - Visit: `<UI_URL>` to access the Streamlit interface

