#!/bin/bash
# Upload models to Vertex AI Model Registry

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-us-central1}"
MODELS_DIR="${MODELS_DIR:-models}"
GCS_BUCKET="${GCS_BUCKET:-}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

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
    if [ -z "$PROJECT_ID" ]; then
        PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
        if [ -z "$PROJECT_ID" ]; then
            print_error "GCP_PROJECT_ID not set"
            exit 1
        fi
    fi
    
    if [ -z "$GCS_BUCKET" ]; then
        GCS_BUCKET="${RESOURCE_PREFIX:-ctr-prediction}-models-${PROJECT_ID}"
        print_info "Using bucket: $GCS_BUCKET"
    fi
    
    gcloud config set project "$PROJECT_ID" > /dev/null 2>&1
}

upload_model_to_gcs() {
    local model_file=$1
    local gcs_path="gs://${GCS_BUCKET}/vertex-ai/${model_file}"
    
    if [ ! -f "${MODELS_DIR}/${model_file}" ]; then
        print_warn "Model file not found: ${MODELS_DIR}/${model_file}"
        return 1
    fi
    
    print_info "Uploading ${model_file} to GCS..."
    gsutil cp "${MODELS_DIR}/${model_file}" "$gcs_path"
    echo "$gcs_path"
}

upload_to_vertex_ai() {
    local model_name=$1
    local model_file=$2
    local version=${3:-v1}
    
    print_info "Uploading ${model_name} to Vertex AI Model Registry..."
    
    # Upload model to Vertex AI
    gcloud ai models upload \
        --region="$REGION" \
        --display-name="CTR Prediction ${model_name}" \
        --description="CTR Prediction Model - ${model_name}" \
        --container-image-uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest" \
        --artifact-uri="gs://${GCS_BUCKET}/vertex-ai/${model_file}" \
        --labels="managed-by=ctr-prediction-deployment,model-type=${model_name}" \
        --project="$PROJECT_ID" || {
        print_error "Failed to upload ${model_name}"
        return 1
    }
    
    print_info "Successfully uploaded ${model_name}"
}

main() {
    print_info "=========================================="
    print_info "Uploading Models to Vertex AI"
    print_info "=========================================="
    print_info ""
    
    check_prerequisites
    
    # Upload models
    MODELS=("xgboost_model.pkl" "lightgbm_model.pkl" "logistic_model.pkl")
    
    for model_file in "${MODELS[@]}"; do
        model_name=$(basename "$model_file" .pkl | sed 's/_model//')
        
        # Upload to GCS first
        gcs_path=$(upload_model_to_gcs "$model_file") || continue
        
        # Upload to Vertex AI
        upload_to_vertex_ai "$model_name" "$model_file" "v1" || continue
    done
    
    print_info ""
    print_info "=========================================="
    print_info "Upload Complete!"
    print_info "=========================================="
    print_info ""
    print_info "List models:"
    print_info "  gcloud ai models list --region=$REGION"
    print_info ""
}

main

