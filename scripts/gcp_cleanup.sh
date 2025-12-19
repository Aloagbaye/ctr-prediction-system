#!/bin/bash
# GCP Cleanup Script for CTR Prediction System
# Deletes all resources created by the deployment script

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-us-central1}"
RESOURCE_PREFIX="ctr-prediction"
RESOURCE_LABEL="managed-by=ctr-prediction-deployment"

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

confirm_deletion() {
    echo ""
    print_warn "This will delete ALL resources with label: ${RESOURCE_LABEL}"
    print_warn "This includes:"
    print_warn "  - Cloud Run services (${RESOURCE_PREFIX}-api, ${RESOURCE_PREFIX}-ui)"
    print_warn "  - Artifact Registry repository (${RESOURCE_PREFIX}-repo)"
    print_warn "  - Cloud Storage bucket (${RESOURCE_PREFIX}-models-*)"
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        print_info "Cleanup cancelled."
        exit 0
    fi
}

delete_cloud_run_services() {
    print_info "Deleting Cloud Run services..."
    
    SERVICES=("${RESOURCE_PREFIX}-api" "${RESOURCE_PREFIX}-ui")
    
    for service in "${SERVICES[@]}"; do
        if gcloud run services describe "$service" --region="$REGION" --project="$PROJECT_ID" &> /dev/null; then
            print_info "Deleting Cloud Run service: $service"
            gcloud run services delete "$service" \
                --region="$REGION" \
                --project="$PROJECT_ID" \
                --quiet || print_warn "Failed to delete $service"
        else
            print_warn "Service $service does not exist. Skipping."
        fi
    done
}

delete_artifact_registry() {
    print_info "Deleting Artifact Registry repository..."
    
    REPO_NAME="${RESOURCE_PREFIX}-repo"
    
    if gcloud artifacts repositories describe "$REPO_NAME" \
        --location="$REGION" \
        --project="$PROJECT_ID" &> /dev/null; then
        print_info "Deleting Artifact Registry repository: $REPO_NAME"
        gcloud artifacts repositories delete "$REPO_NAME" \
            --location="$REGION" \
            --project="$PROJECT_ID" \
            --quiet || print_warn "Failed to delete repository"
    else
        print_warn "Repository $REPO_NAME does not exist. Skipping."
    fi
}

delete_storage_buckets() {
    print_info "Deleting Cloud Storage buckets..."
    
    # Find all buckets with the label
    BUCKETS=$(gsutil ls -L 2>/dev/null | grep -E "gs://${RESOURCE_PREFIX}-" || true)
    
    if [ -n "$BUCKETS" ]; then
        for bucket in $(gsutil ls | grep -E "gs://${RESOURCE_PREFIX}-"); do
            print_info "Deleting bucket: $bucket"
            gsutil -m rm -r "$bucket" || print_warn "Failed to delete $bucket"
        done
    else
        print_warn "No buckets found with prefix ${RESOURCE_PREFIX}-"
    fi
}

# Main cleanup flow
main() {
    print_info "=========================================="
    print_info "GCP Cleanup for CTR Prediction System"
    print_info "=========================================="
    print_info ""
    
    # Check if PROJECT_ID is set
    if [ -z "$PROJECT_ID" ]; then
        PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
        if [ -z "$PROJECT_ID" ]; then
            print_error "GCP_PROJECT_ID environment variable is not set."
            exit 1
        fi
        print_info "Using project from gcloud config: $PROJECT_ID"
    fi
    
    gcloud config set project "$PROJECT_ID" > /dev/null 2>&1
    
    confirm_deletion
    
    delete_cloud_run_services
    delete_artifact_registry
    delete_storage_buckets
    
    print_info ""
    print_info "=========================================="
    print_info "Cleanup Complete!"
    print_info "=========================================="
    print_info ""
    print_info "All resources with label '${RESOURCE_LABEL}' have been deleted."
    print_info ""
}

# Run main function
main

