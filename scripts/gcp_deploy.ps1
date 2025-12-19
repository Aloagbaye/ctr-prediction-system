# PowerShell script for GCP Deployment
# Creates all resources with labels for easy cleanup

$ErrorActionPreference = "Stop"

# Configuration
$PROJECT_ID = if ($env:GCP_PROJECT_ID) { $env:GCP_PROJECT_ID } else { "" }
$REGION = if ($env:GCP_REGION) { $env:GCP_REGION } else { "us-central1" }
$RESOURCE_PREFIX = "ctr-prediction"
$RESOURCE_LABEL = "managed-by=ctr-prediction-deployment"
$RESOURCE_LABEL_DATE = "deployment-date=$(Get-Date -Format 'yyyyMMdd')"

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Check-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    # Check if gcloud is installed
    if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
        Write-Error "gcloud CLI is not installed. Please install it from https://cloud.google.com/sdk/docs/install"
        exit 1
    }
    
    # Check if Docker is installed
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Docker is not installed. Please install it from https://docs.docker.com/get-docker/"
        exit 1
    }
    
    # Check if PROJECT_ID is set
    if ([string]::IsNullOrEmpty($PROJECT_ID)) {
        $PROJECT_ID = gcloud config get-value project 2>$null
        if ([string]::IsNullOrEmpty($PROJECT_ID)) {
            Write-Error "GCP_PROJECT_ID environment variable is not set."
            Write-Info "Please set it: `$env:GCP_PROJECT_ID='your-project-id'"
            Write-Info "Or run: gcloud config set project YOUR_PROJECT_ID"
            exit 1
        }
        Write-Info "Using project from gcloud config: $PROJECT_ID"
    }
    
    # Set the project
    gcloud config set project $PROJECT_ID | Out-Null
    
    # Enable required APIs
    Write-Info "Enabling required GCP APIs..."
    gcloud services enable `
        run.googleapis.com `
        cloudbuild.googleapis.com `
        storage-component.googleapis.com `
        artifactregistry.googleapis.com `
        --project=$PROJECT_ID | Out-Null
    
    Write-Info "Prerequisites check complete!"
}

function Create-StorageBucket {
    Write-Info "Creating Cloud Storage bucket for models..."
    
    $BUCKET_NAME = "${RESOURCE_PREFIX}-models-${PROJECT_ID}"
    
    # Check if bucket exists
    $bucketExists = gsutil ls -b "gs://${BUCKET_NAME}" 2>$null
    if ($bucketExists) {
        Write-Warn "Bucket ${BUCKET_NAME} already exists. Skipping creation."
    } else {
        gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION "gs://${BUCKET_NAME}"
        gsutil label ch -l "${RESOURCE_LABEL}" "gs://${BUCKET_NAME}"
        gsutil label ch -l "${RESOURCE_LABEL_DATE}" "gs://${BUCKET_NAME}"
        Write-Info "Bucket created: gs://${BUCKET_NAME}"
    }
    
    # Upload models if they exist locally
    if ((Test-Path "models") -and (Get-ChildItem "models\*.pkl" -ErrorAction SilentlyContinue)) {
        Write-Info "Uploading models to Cloud Storage..."
        gsutil -m cp models\*.pkl "gs://${BUCKET_NAME}/" 2>$null
        Write-Info "Models uploaded to gs://${BUCKET_NAME}/"
    } else {
        Write-Warn "No local models found. Please upload models manually to gs://${BUCKET_NAME}/"
    }
    
    return $BUCKET_NAME
}

function Build-AndPushImages {
    Write-Info "Building and pushing Docker images to Artifact Registry..."
    
    $ARTIFACT_REGISTRY = "${REGION}-docker.pkg.dev/${PROJECT_ID}/${RESOURCE_PREFIX}-repo"
    
    # Create Artifact Registry repository if it doesn't exist
    $repoExists = gcloud artifacts repositories describe "${RESOURCE_PREFIX}-repo" `
        --location=$REGION --project=$PROJECT_ID 2>$null
    if (-not $repoExists) {
        gcloud artifacts repositories create "${RESOURCE_PREFIX}-repo" `
            --repository-format=docker `
            --location=$REGION `
            --project=$PROJECT_ID `
            --labels="${RESOURCE_LABEL},${RESOURCE_LABEL_DATE}"
    }
    
    # Configure Docker to use gcloud as credential helper
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet | Out-Null
    
    # Build and push API image
    Write-Info "Building API image..."
    docker build -f Dockerfile.api -t "${ARTIFACT_REGISTRY}/api:latest" .
    docker push "${ARTIFACT_REGISTRY}/api:latest"
    
    # Build and push Streamlit image
    Write-Info "Building Streamlit image..."
    docker build -f Dockerfile.streamlit -t "${ARTIFACT_REGISTRY}/streamlit:latest" .
    docker push "${ARTIFACT_REGISTRY}/streamlit:latest"
    
    Write-Info "Images pushed to Artifact Registry"
    return $ARTIFACT_REGISTRY
}

function Deploy-ApiService {
    param(
        [string]$ArtifactRegistry,
        [string]$BucketName
    )
    
    Write-Info "Deploying API service to Cloud Run..."
    
    $SERVICE_NAME = "${RESOURCE_PREFIX}-api"
    $IMAGE = "${ArtifactRegistry}/api:latest"
    
    gcloud run deploy $SERVICE_NAME `
        --image=$IMAGE `
        --platform=managed `
        --region=$REGION `
        --project=$PROJECT_ID `
        --allow-unauthenticated `
        --memory=2Gi `
        --cpu=2 `
        --timeout=300 `
        --max-instances=10 `
        --min-instances=0 `
        --set-env-vars="MODELS_DIR=/tmp/models,GCP_PROJECT_ID=${PROJECT_ID},GCS_BUCKET=${BucketName}" `
        --labels="${RESOURCE_LABEL},${RESOURCE_LABEL_DATE}" `
        --quiet | Out-Null
    
    # Get the service URL
    $API_URL = gcloud run services describe $SERVICE_NAME `
        --region=$REGION `
        --project=$PROJECT_ID `
        --format="value(status.url)"
    
    Write-Info "API service deployed: $API_URL"
    return $API_URL
}

function Deploy-StreamlitService {
    param(
        [string]$ArtifactRegistry,
        [string]$ApiUrl
    )
    
    Write-Info "Deploying Streamlit UI to Cloud Run..."
    
    $SERVICE_NAME = "${RESOURCE_PREFIX}-ui"
    $IMAGE = "${ArtifactRegistry}/streamlit:latest"
    
    gcloud run deploy $SERVICE_NAME `
        --image=$IMAGE `
        --platform=managed `
        --region=$REGION `
        --project=$PROJECT_ID `
        --allow-unauthenticated `
        --memory=1Gi `
        --cpu=1 `
        --timeout=300 `
        --max-instances=5 `
        --min-instances=0 `
        --set-env-vars="API_URL=${ApiUrl}" `
        --labels="${RESOURCE_LABEL},${RESOURCE_LABEL_DATE}" `
        --quiet | Out-Null
    
    # Get the service URL
    $STREAMLIT_URL = gcloud run services describe $SERVICE_NAME `
        --region=$REGION `
        --project=$PROJECT_ID `
        --format="value(status.url)"
    
    Write-Info "Streamlit UI deployed: $STREAMLIT_URL"
    return $STREAMLIT_URL
}

# Main deployment flow
Write-Info "=========================================="
Write-Info "GCP Deployment for CTR Prediction System"
Write-Info "=========================================="
Write-Info ""
Write-Info "Project ID: $PROJECT_ID"
Write-Info "Region: $REGION"
Write-Info "Resource Prefix: $RESOURCE_PREFIX"
Write-Info ""

Check-Prerequisites

# Create storage bucket
$BUCKET_NAME = Create-StorageBucket

# Build and push images
$ARTIFACT_REGISTRY = Build-AndPushImages

# Deploy API service
$API_URL = Deploy-ApiService $ARTIFACT_REGISTRY $BUCKET_NAME

# Deploy Streamlit service
$STREAMLIT_URL = Deploy-StreamlitService $ARTIFACT_REGISTRY $API_URL

Write-Info ""
Write-Info "=========================================="
Write-Info "Deployment Complete!"
Write-Info "=========================================="
Write-Info ""
Write-Info "API Service: $API_URL"
Write-Info "API Docs: $API_URL/docs"
Write-Info "Streamlit UI: $STREAMLIT_URL"
Write-Info ""
Write-Info "To clean up all resources, run:"
Write-Info "  .\scripts\gcp_cleanup.ps1"
Write-Info ""

