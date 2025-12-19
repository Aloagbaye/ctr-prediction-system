#!/bin/bash
# Set up BigQuery dataset and tables for CTR predictions

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
DATASET_NAME="${BIGQUERY_DATASET:-ctr_predictions}"
LOCATION="${BIGQUERY_LOCATION:-US}"

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
    
    print_info "Using project: $PROJECT_ID"
    print_info "Dataset: $DATASET_NAME"
}

create_dataset() {
    print_info "Creating BigQuery dataset..."
    
    if bq ls -d "$PROJECT_ID:$DATASET_NAME" &> /dev/null; then
        print_warn "Dataset $DATASET_NAME already exists. Skipping creation."
    else
        bq mk --dataset \
            --location="$LOCATION" \
            --description="CTR Prediction System Data" \
            --label=managed-by:ctr-prediction-deployment \
            "$PROJECT_ID:$DATASET_NAME"
        print_info "Dataset created: $DATASET_NAME"
    fi
}

create_predictions_table() {
    print_info "Creating predictions table..."
    
    TABLE_NAME="predictions"
    FULL_TABLE="$PROJECT_ID:$DATASET_NAME.$TABLE_NAME"
    
    if bq ls -t "$FULL_TABLE" &> /dev/null; then
        print_warn "Table $TABLE_NAME already exists. Skipping creation."
    else
        bq mk --table \
            --description="Historical CTR Predictions" \
            --time_partitioning_field=timestamp \
            --time_partitioning_type=DAY \
            --clustering_fields=model_name,device \
            --label=managed-by:ctr-prediction-deployment \
            "$FULL_TABLE" \
            user_id:STRING,ad_id:STRING,device:STRING,placement:STRING, \
            predicted_ctr:FLOAT,actual_ctr:FLOAT,timestamp:TIMESTAMP, \
            model_name:STRING,request_id:STRING,features:JSON
        print_info "Table created: $TABLE_NAME"
    fi
}

create_training_data_table() {
    print_info "Creating training data table..."
    
    TABLE_NAME="training_data"
    FULL_TABLE="$PROJECT_ID:$DATASET_NAME.$TABLE_NAME"
    
    if bq ls -t "$FULL_TABLE" &> /dev/null; then
        print_warn "Table $TABLE_NAME already exists. Skipping creation."
    else
        bq mk --table \
            --description="Training Data for CTR Models" \
            --time_partitioning_field=timestamp \
            --time_partitioning_type=DAY \
            --label=managed-by:ctr-prediction-deployment \
            "$FULL_TABLE" \
            user_id:STRING,ad_id:STRING,device:STRING,placement:STRING, \
            clicked:BOOLEAN,timestamp:TIMESTAMP,features:JSON
        print_info "Table created: $TABLE_NAME"
    fi
}

main() {
    print_info "=========================================="
    print_info "Setting up BigQuery for CTR Predictions"
    print_info "=========================================="
    print_info ""
    
    check_prerequisites
    create_dataset
    create_predictions_table
    create_training_data_table
    
    print_info ""
    print_info "=========================================="
    print_info "Setup Complete!"
    print_info "=========================================="
    print_info ""
    print_info "Dataset: $PROJECT_ID.$DATASET_NAME"
    print_info "Tables:"
    print_info "  - predictions"
    print_info "  - training_data"
    print_info ""
    print_info "Query example:"
    print_info "  bq query --use_legacy_sql=false \"SELECT COUNT(*) FROM \`$PROJECT_ID.$DATASET_NAME.predictions\`\""
    print_info ""
}

main

