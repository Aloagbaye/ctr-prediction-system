# GCP gcloud Commands Tutorial

## 📋 Overview

This tutorial covers the essential `gcloud` and `gsutil` commands used in the CTR Prediction System deployment. These commands are fundamental for working with Google Cloud Platform (GCP) services.

**Prerequisites:**
- GCP account with billing enabled
- `gcloud` CLI installed ([Installation Guide](https://cloud.google.com/sdk/docs/install))
- Basic familiarity with command-line interfaces

---

## 🔐 Authentication & Configuration

### 1. Login to GCP

```bash
# Authenticate with your Google account
gcloud auth login

# For service accounts or CI/CD
gcloud auth activate-service-account --key-file=path/to/key.json
```

**What it does:** Authenticates you with Google Cloud and stores credentials locally.

**When to use:** First time setup, or when credentials expire.

---

### 2. Set Default Project

```bash
# Set the default project for all commands
gcloud config set project PROJECT_ID

# View current project
gcloud config get-value project

# List all projects
gcloud projects list
```

**What it does:** Sets the default project ID so you don't need to specify `--project` in every command.

**Example:**
```bash
gcloud config set project my-ctr-prediction-project
gcloud config get-value project
# Output: my-ctr-prediction-project
```

**In the script:** Used to ensure all commands use the correct project.

---

### 3. Configure Docker Authentication

```bash
# Configure Docker to use gcloud credentials for Artifact Registry
gcloud auth configure-docker REGION-docker.pkg.dev

# Example for us-central1
gcloud auth configure-docker us-central1-docker.pkg.dev
```

**What it does:** Configures Docker to authenticate with GCP Artifact Registry using your gcloud credentials.

**When to use:** Before pushing Docker images to Artifact Registry.

**In the script:** Used before building and pushing Docker images.

---

## 🗄️ Cloud Storage (gsutil)

### 4. List Buckets

```bash
# List all buckets in the project
gsutil ls

# Check if a specific bucket exists
gsutil ls -b gs://BUCKET_NAME

# List contents of a bucket
gsutil ls gs://BUCKET_NAME/
```

**What it does:** Lists buckets or checks if a bucket exists.

**Example:**
```bash
gsutil ls -b gs://ctr-prediction-models-my-project
# Output: gs://ctr-prediction-models-my-project (if exists)
# Or: CommandException: 404 gs://... bucket does not exist (if not)
```

**In the script:** Used to check if the models bucket exists before creating it.

---

### 5. Create Storage Bucket

```bash
# Create a new bucket
gsutil mb -p PROJECT_ID -c STORAGE_CLASS -l LOCATION gs://BUCKET_NAME

# Example
gsutil mb -p my-project -c STANDARD -l us-central1 gs://my-bucket
```

**Parameters:**
- `-p PROJECT_ID`: Project ID
- `-c STORAGE_CLASS`: Storage class (STANDARD, NEARLINE, COLDLINE, ARCHIVE)
- `-l LOCATION`: Bucket location (us-central1, us-east1, etc.)

**What it does:** Creates a new Cloud Storage bucket.

**Example:**
```bash
gsutil mb -p my-project -c STANDARD -l us-central1 gs://ctr-prediction-models-my-project
# Output: Creating gs://ctr-prediction-models-my-project/...
```

**In the script:** Creates the bucket for storing model files.

---

### 6. Set Bucket Labels

```bash
# Add or update a label on a bucket
gsutil label ch -l KEY:VALUE gs://BUCKET_NAME

# Multiple labels (separate commands)
gsutil label ch -l key1:value1 gs://BUCKET_NAME
gsutil label ch -l key2:value2 gs://BUCKET_NAME
```

**Important:** `gsutil` uses `KEY:VALUE` format (colon), not `KEY=VALUE` (equals).

**What it does:** Adds metadata labels to a bucket for organization and filtering.

**Example:**
```bash
gsutil label ch -l managed-by:ctr-prediction-deployment gs://my-bucket
gsutil label ch -l deployment-date:20251220 gs://my-bucket
```

**In the script:** Labels buckets for easy identification and cleanup.

---

### 7. Copy Files to Cloud Storage

```bash
# Copy a single file
gsutil cp LOCAL_FILE gs://BUCKET_NAME/

# Copy multiple files (wildcard)
gsutil cp *.pkl gs://BUCKET_NAME/

# Parallel copy (faster for multiple files)
gsutil -m cp *.pkl gs://BUCKET_NAME/

# Quiet mode (suppress progress)
gsutil -m -q cp *.pkl gs://BUCKET_NAME/
```

**Parameters:**
- `-m`: Parallel/threaded copy (faster)
- `-q`: Quiet mode (suppress output)
- `-r`: Recursive (for directories)

**What it does:** Uploads files to Cloud Storage.

**Example:**
```bash
gsutil -m -q cp models/*.pkl gs://ctr-prediction-models-my-project/
# Uploads all .pkl files from models/ directory
```

**In the script:** Uploads trained model files to the bucket.

---

## 🏗️ Artifact Registry

### 8. Create Artifact Registry Repository

```bash
# Create a Docker repository
gcloud artifacts repositories create REPOSITORY_NAME \
    --repository-format=docker \
    --location=REGION \
    --project=PROJECT_ID \
    --labels=KEY=VALUE

# Example
gcloud artifacts repositories create ctr-prediction-repo \
    --repository-format=docker \
    --location=us-central1 \
    --project=my-project \
    --labels=managed-by=ctr-prediction-deployment
```

**Parameters:**
- `--repository-format`: Format (docker, maven, npm, etc.)
- `--location`: Region (us-central1, us-east1, etc.)
- `--labels`: Key-value pairs for organization

**What it does:** Creates a repository for storing Docker images (or other artifacts).

**Example:**
```bash
gcloud artifacts repositories create my-repo \
    --repository-format=docker \
    --location=us-central1 \
    --project=my-project
```

**In the script:** Creates repository for storing Docker images.

---

### 9. Check if Repository Exists

```bash
# Describe a repository (returns error if not found)
gcloud artifacts repositories describe REPOSITORY_NAME \
    --location=REGION \
    --project=PROJECT_ID
```

**What it does:** Returns repository details if it exists, or error if not.

**Example:**
```bash
gcloud artifacts repositories describe my-repo \
    --location=us-central1 \
    --project=my-project
# Output: Repository details if exists
# Or: ERROR: (gcloud.artifacts.repositories.describe) NOT_FOUND
```

**In the script:** Used to check if repository exists before creating it.

---

## 🚀 Cloud Run

### 10. Deploy Service to Cloud Run

```bash
# Deploy a service
gcloud run deploy SERVICE_NAME \
    --image=IMAGE_URL \
    --platform=managed \
    --region=REGION \
    --project=PROJECT_ID \
    --allow-unauthenticated \
    --memory=MEMORY \
    --cpu=CPU \
    --timeout=TIMEOUT \
    --max-instances=MAX \
    --min-instances=MIN \
    --set-env-vars="KEY1=VALUE1,KEY2=VALUE2" \
    --labels="KEY1=VALUE1,KEY2=VALUE2"

# Example
gcloud run deploy ctr-prediction-api \
    --image=us-central1-docker.pkg.dev/my-project/my-repo/api:latest \
    --platform=managed \
    --region=us-central1 \
    --project=my-project \
    --allow-unauthenticated \
    --memory=2Gi \
    --cpu=2 \
    --timeout=300 \
    --max-instances=10 \
    --min-instances=0 \
    --set-env-vars="MODELS_DIR=/tmp/models,GCP_PROJECT_ID=my-project" \
    --labels="managed-by=ctr-prediction-deployment"
```

**Key Parameters:**
- `--image`: Docker image URL (from Artifact Registry)
- `--platform`: `managed` for fully managed Cloud Run
- `--allow-unauthenticated`: Allow public access (or use `--no-allow-unauthenticated`)
- `--memory`: Memory allocation (512Mi, 1Gi, 2Gi, etc.)
- `--cpu`: CPU allocation (1, 2, 4, etc.)
- `--timeout`: Request timeout in seconds (max 3600)
- `--max-instances`: Maximum concurrent instances
- `--min-instances`: Minimum instances to keep warm
- `--set-env-vars`: Environment variables (comma-separated)
- `--labels`: Resource labels

**What it does:** Deploys a containerized application to Cloud Run.

**Example:**
```bash
gcloud run deploy my-api \
    --image=us-central1-docker.pkg.dev/my-project/repo/api:latest \
    --region=us-central1 \
    --allow-unauthenticated \
    --memory=2Gi
```

**In the script:** Deploys both API and Streamlit services.

---

### 11. Get Service URL

```bash
# Get the service URL
gcloud run services describe SERVICE_NAME \
    --region=REGION \
    --project=PROJECT_ID \
    --format="value(status.url)"

# Example
gcloud run services describe ctr-prediction-api \
    --region=us-central1 \
    --project=my-project \
    --format="value(status.url)"
# Output: https://ctr-prediction-api-xxxxx-uc.a.run.app
```

**Parameters:**
- `--format`: Output format (`value()` for single value, `json`, `yaml`, etc.)

**What it does:** Retrieves the public URL of a deployed Cloud Run service.

**Example:**
```bash
gcloud run services describe my-api \
    --region=us-central1 \
    --format="value(status.url)"
# Output: https://my-api-xxxxx-uc.a.run.app
```

**In the script:** Gets the API URL to pass to the Streamlit service.

---

## 🔧 API Management

### 12. Enable GCP APIs

```bash
# Enable a single API
gcloud services enable API_NAME --project=PROJECT_ID

# Enable multiple APIs
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    storage-component.googleapis.com \
    artifactregistry.googleapis.com \
    --project=PROJECT_ID
```

**Common APIs:**
- `run.googleapis.com`: Cloud Run
- `cloudbuild.googleapis.com`: Cloud Build
- `storage-component.googleapis.com`: Cloud Storage
- `artifactregistry.googleapis.com`: Artifact Registry
- `aiplatform.googleapis.com`: Vertex AI
- `bigquery.googleapis.com`: BigQuery

**What it does:** Enables required APIs for your project.

**Example:**
```bash
gcloud services enable run.googleapis.com --project=my-project
```

**In the script:** Enables all required APIs before deployment.

---

## 📊 Useful Query Commands

### 13. List All Resources with Labels

```bash
# List Cloud Run services with specific label
gcloud run services list \
    --filter="labels.managed-by=ctr-prediction-deployment" \
    --project=PROJECT_ID

# List Artifact Registry repositories
gcloud artifacts repositories list \
    --location=REGION \
    --project=PROJECT_ID

# List Cloud Storage buckets
gsutil ls -L | grep -A 10 "gs://BUCKET_NAME"
```

**What it does:** Lists resources filtered by labels or other criteria.

**Example:**
```bash
gcloud run services list \
    --filter="labels.managed-by=ctr-prediction-deployment" \
    --project=my-project
```

**Use case:** Finding all resources created by the deployment script for cleanup.

---

### 14. Get Service Details

```bash
# Get full service details (JSON)
gcloud run services describe SERVICE_NAME \
    --region=REGION \
    --project=PROJECT_ID \
    --format=json

# Get specific field
gcloud run services describe SERVICE_NAME \
    --region=REGION \
    --format="value(status.url)"
```

**What it does:** Retrieves detailed information about a Cloud Run service.

**Example:**
```bash
gcloud run services describe my-api \
    --region=us-central1 \
    --format=json
```

---

## 🧹 Cleanup Commands

### 15. Delete Cloud Run Service

```bash
# Delete a service
gcloud run services delete SERVICE_NAME \
    --region=REGION \
    --project=PROJECT_ID \
    --quiet
```

**What it does:** Deletes a Cloud Run service.

**Example:**
```bash
gcloud run services delete ctr-prediction-api \
    --region=us-central1 \
    --quiet
```

---

### 16. Delete Artifact Registry Repository

```bash
# Delete a repository
gcloud artifacts repositories delete REPOSITORY_NAME \
    --location=REGION \
    --project=PROJECT_ID \
    --quiet
```

**What it does:** Deletes an Artifact Registry repository.

**Example:**
```bash
gcloud artifacts repositories delete ctr-prediction-repo \
    --location=us-central1 \
    --quiet
```

---

### 17. Delete Cloud Storage Bucket

```bash
# Delete a bucket (must be empty first)
gsutil rm -r gs://BUCKET_NAME

# Delete all objects first, then bucket
gsutil rm -r gs://BUCKET_NAME/**
gsutil rb gs://BUCKET_NAME
```

**What it does:** Deletes a Cloud Storage bucket and its contents.

**Example:**
```bash
gsutil rm -r gs://ctr-prediction-models-my-project
```

---

## 💡 Tips & Best Practices

### 1. Use `--quiet` Flag

```bash
# Suppress prompts (useful in scripts)
gcloud run services delete SERVICE_NAME --region=REGION --quiet
```

**When to use:** In automation scripts to avoid interactive prompts.

---

### 2. Use `--format` for Parsing

```bash
# Get just the URL (useful in scripts)
gcloud run services describe SERVICE_NAME \
    --region=REGION \
    --format="value(status.url)"
```

**When to use:** When you need specific values in scripts.

---

### 3. Use Labels for Organization

```bash
# Add labels to all resources
--labels="managed-by=my-deployment,environment=production"
```

**Best practice:** Label all resources for easy filtering and cleanup.

---

### 4. Check Exit Codes

```bash
# In PowerShell, check $LASTEXITCODE
gsutil ls -b gs://BUCKET_NAME 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    # Bucket exists
}

# In Bash, check $?
gsutil ls -b gs://BUCKET_NAME 2>&1 > /dev/null
if [ $? -eq 0 ]; then
    # Bucket exists
fi
```

**Best practice:** Always check exit codes when checking resource existence.

---

### 5. Use Parallel Operations

```bash
# Faster for multiple files
gsutil -m cp *.pkl gs://BUCKET_NAME/
```

**Best practice:** Use `-m` flag for multiple file operations.

---

## 🔍 Common Error Messages & Solutions

### Error: "Project not found"

```bash
# Solution: Set the correct project
gcloud config set project PROJECT_ID
```

---

### Error: "API not enabled"

```bash
# Solution: Enable the required API
gcloud services enable API_NAME --project=PROJECT_ID
```

---

### Error: "Permission denied"

```bash
# Solution: Check IAM permissions
gcloud projects get-iam-policy PROJECT_ID

# Or authenticate again
gcloud auth login
```

---

### Error: "Bucket already exists"

```bash
# Solution: Check if bucket exists first
gsutil ls -b gs://BUCKET_NAME
```

---

## 📚 Additional Resources

- [gcloud CLI Reference](https://cloud.google.com/sdk/gcloud/reference)
- [gsutil Documentation](https://cloud.google.com/storage/docs/gsutil)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Artifact Registry Documentation](https://cloud.google.com/artifact-registry/docs)
- [Cloud Storage Documentation](https://cloud.google.com/storage/docs)

---

## 🎯 Quick Reference

### Most Common Commands

```bash
# Authentication
gcloud auth login
gcloud config set project PROJECT_ID

# Storage
gsutil mb -p PROJECT_ID -l REGION gs://BUCKET_NAME
gsutil cp FILE gs://BUCKET_NAME/
gsutil ls gs://BUCKET_NAME/

# Artifact Registry
gcloud artifacts repositories create REPO --repository-format=docker --location=REGION
gcloud auth configure-docker REGION-docker.pkg.dev

# Cloud Run
gcloud run deploy SERVICE --image=IMAGE --region=REGION --allow-unauthenticated
gcloud run services describe SERVICE --region=REGION --format="value(status.url)"

# APIs
gcloud services enable API_NAME --project=PROJECT_ID
```

---

**Happy deploying!** 🚀

