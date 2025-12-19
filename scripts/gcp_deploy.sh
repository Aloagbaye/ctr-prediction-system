#!/bin/bash
# GCP Deployment Script for CTR Prediction System
# Creates all resources with labels for easy cleanup

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-us-central1}"
RESOURCE_PREFIX="ctr-prediction"
RESOURCE_LABEL="managed-by=ctr-prediction-deployment"
RESOURCE_LABEL_DATE="deployment-date=$(date +%Y%m%d)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI is not installed. Please install it from https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install it from https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check if PROJECT_ID is set
    if [ -z "$PROJECT_ID" ]; then
        print_error "GCP_PROJECT_ID environment variable is not set."
        print_info "Please set it: export GCP_PROJECT_ID=your-project-id"
        print_info "Or run: gcloud config set project YOUR_PROJECT_ID"
        PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
        if [ -z "$PROJECT_ID" ]; then
            exit 1
        fi
        print_info "Using project from gcloud config: $PROJECT_ID"
    fi
    
    # Set the project
    gcloud config set project "$PROJECT_ID" > /dev/null 2>&1
    
    # Enable required APIs
    print_info "Enabling required GCP APIs..."
    gcloud services enable \
        run.googleapis.com \
        cloudbuild.googleapis.com \
        storage-component.googleapis.com \
        artifactregistry.googleapis.com \
        --project="$PROJECT_ID" > /dev/null 2>&1 || true
    
    print_info "Prerequisites check complete!"
}

create_storage_bucket() {
    print_info "Creating Cloud Storage bucket for models..."
    
    BUCKET_NAME="${RESOURCE_PREFIX}-models-${PROJECT_ID}"
    
    # Check if bucket exists
    if gsutil ls -b "gs://${BUCKET_NAME}" &> /dev/null; then
        print_warn "Bucket ${BUCKET_NAME} already exists. Skipping creation."
    else
        gsutil mb -p "$PROJECT_ID" -c STANDARD -l "$REGION" "gs://${BUCKET_NAME}"
        gsutil label ch -l "${RESOURCE_LABEL}" "gs://${BUCKET_NAME}"
        gsutil label ch -l "${RESOURCE_LABEL_DATE}" "gs://${BUCKET_NAME}"
        print_info "Bucket created: gs://${BUCKET_NAME}"
    fi
    
    # Upload models if they exist locally
    if [ -d "models" ] && [ "$(ls -A models/*.pkl 2>/dev/null)" ]; then
        print_info "Uploading models to Cloud Storage..."
        gsutil -m cp models/*.pkl "gs://${BUCKET_NAME}/" || print_warn "Failed to upload some model files"
        print_info "Models uploaded to gs://${BUCKET_NAME}/"
    else
        print_warn "No local models found. Please upload models manually to gs://${BUCKET_NAME}/"
    fi
    
    echo "$BUCKET_NAME"
}

build_and_push_images() {
    print_info "Building and pushing Docker images to Artifact Registry..."
    
    ARTIFACT_REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${RESOURCE_PREFIX}-repo"
    
    # Create Artifact Registry repository if it doesn't exist
    if ! gcloud artifacts repositories describe "${RESOURCE_PREFIX}-repo" \
        --location="$REGION" --project="$PROJECT_ID" &> /dev/null; then
        gcloud artifacts repositories create "${RESOURCE_PREFIX}-repo" \
            --repository-format=docker \
            --location="$REGION" \
            --project="$PROJECT_ID" \
            --labels="${RESOURCE_LABEL},${RESOURCE_LABEL_DATE}"
    fi
    
    # Configure Docker to use gcloud as credential helper
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
    
    # Build and push API image
    print_info "Building API image..."
    docker build -f Dockerfile.api -t "${ARTIFACT_REGISTRY}/api:latest" .
    docker push "${ARTIFACT_REGISTRY}/api:latest"
    
    # Build and push Streamlit image
    print_info "Building Streamlit image..."
    docker build -f Dockerfile.streamlit -t "${ARTIFACT_REGISTRY}/streamlit:latest" .
    docker push "${ARTIFACT_REGISTRY}/streamlit:latest"
    
    print_info "Images pushed to Artifact Registry"
    echo "$ARTIFACT_REGISTRY"
}

deploy_api_service() {
    print_info "Deploying API service to Cloud Run..."
    
    ARTIFACT_REGISTRY="$1"
    BUCKET_NAME="$2"
    
    SERVICE_NAME="${RESOURCE_PREFIX}-api"
    IMAGE="${ARTIFACT_REGISTRY}/api:latest"
    
    gcloud run deploy "$SERVICE_NAME" \
        --image="$IMAGE" \
        --platform=managed \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --allow-unauthenticated \
        --memory=2Gi \
        --cpu=2 \
        --timeout=300 \
        --max-instances=10 \
        --min-instances=0 \
        --set-env-vars="MODELS_DIR=/tmp/models" \
        --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID}" \
        --set-env-vars="GCS_BUCKET=${BUCKET_NAME}" \
        --labels="${RESOURCE_LABEL},${RESOURCE_LABEL_DATE}" \
        --quiet
    
    # Get the service URL
    API_URL=$(gcloud run services describe "$SERVICE_NAME" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format="value(status.url)")
    
    print_info "API service deployed: $API_URL"
    echo "$API_URL"
}

deploy_streamlit_service() {
    print_info "Deploying Streamlit UI to Cloud Run..."
    
    ARTIFACT_REGISTRY="$1"
    API_URL="$2"
    
    SERVICE_NAME="${RESOURCE_PREFIX}-ui"
    IMAGE="${ARTIFACT_REGISTRY}/streamlit:latest"
    
    gcloud run deploy "$SERVICE_NAME" \
        --image="$IMAGE" \
        --platform=managed \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --allow-unauthenticated \
        --memory=1Gi \
        --cpu=1 \
        --timeout=300 \
        --max-instances=5 \
        --min-instances=0 \
        --set-env-vars="API_URL=${API_URL}" \
        --labels="${RESOURCE_LABEL},${RESOURCE_LABEL_DATE}" \
        --quiet
    
    # Get the service URL
    STREAMLIT_URL=$(gcloud run services describe "$SERVICE_NAME" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format="value(status.url)")
    
    print_info "Streamlit UI deployed: $STREAMLIT_URL"
    echo "$STREAMLIT_URL"
}

# Main deployment flow
main() {
    print_info "=========================================="
    print_info "GCP Deployment for CTR Prediction System"
    print_info "=========================================="
    print_info ""
    print_info "Project ID: $PROJECT_ID"
    print_info "Region: $REGION"
    print_info "Resource Prefix: $RESOURCE_PREFIX"
    print_info ""
    
    check_prerequisites
    
    # Create storage bucket
    BUCKET_NAME=$(create_storage_bucket)
    
    # Build and push images
    ARTIFACT_REGISTRY=$(build_and_push_images)
    
    # Deploy API service
    API_URL=$(deploy_api_service "$ARTIFACT_REGISTRY" "$BUCKET_NAME")
    
    # Deploy Streamlit service
    STREAMLIT_URL=$(deploy_streamlit_service "$ARTIFACT_REGISTRY" "$API_URL")
    
    print_info ""
    print_info "=========================================="
    print_info "Deployment Complete!"
    print_info "=========================================="
    print_info ""
    print_info "API Service: $API_URL"
    print_info "API Docs: $API_URL/docs"
    print_info "Streamlit UI: $STREAMLIT_URL"
    print_info ""
    print_info "To clean up all resources, run:"
    print_info "  ./scripts/gcp_cleanup.sh"
    print_info ""
}

# Run main function
main

