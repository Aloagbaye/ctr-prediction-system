# PowerShell script to set up BigQuery dataset and tables

$ErrorActionPreference = "Stop"

# Configuration
$PROJECT_ID = if ($env:GCP_PROJECT_ID) { $env:GCP_PROJECT_ID } else { "" }
$DATASET_NAME = if ($env:BIGQUERY_DATASET) { $env:BIGQUERY_DATASET } else { "ctr_predictions" }
$LOCATION = if ($env:BIGQUERY_LOCATION) { $env:BIGQUERY_LOCATION } else { "US" }

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
    if ([string]::IsNullOrEmpty($PROJECT_ID)) {
        $PROJECT_ID = gcloud config get-value project 2>$null
        if ([string]::IsNullOrEmpty($PROJECT_ID)) {
            Write-Error "GCP_PROJECT_ID not set"
            exit 1
        }
    }
    
    Write-Info "Using project: $PROJECT_ID"
    Write-Info "Dataset: $DATASET_NAME"
}

function Create-Dataset {
    Write-Info "Creating BigQuery dataset..."
    
    $exists = bq ls -d "$PROJECT_ID:$DATASET_NAME" 2>$null
    if ($exists) {
        Write-Warn "Dataset $DATASET_NAME already exists. Skipping creation."
    } else {
        bq mk --dataset `
            --location=$LOCATION `
            --description="CTR Prediction System Data" `
            --label=managed-by:ctr-prediction-deployment `
            "$PROJECT_ID:$DATASET_NAME"
        Write-Info "Dataset created: $DATASET_NAME"
    }
}

function Create-PredictionsTable {
    Write-Info "Creating predictions table..."
    
    $TABLE_NAME = "predictions"
    $FULL_TABLE = "$PROJECT_ID:$DATASET_NAME.$TABLE_NAME"
    
    $exists = bq ls -t $FULL_TABLE 2>$null
    if ($exists) {
        Write-Warn "Table $TABLE_NAME already exists. Skipping creation."
    } else {
        bq mk --table `
            --description="Historical CTR Predictions" `
            --time_partitioning_field=timestamp `
            --time_partitioning_type=DAY `
            --clustering_fields=model_name,device `
            --label=managed-by:ctr-prediction-deployment `
            $FULL_TABLE `
            "user_id:STRING,ad_id:STRING,device:STRING,placement:STRING,predicted_ctr:FLOAT,actual_ctr:FLOAT,timestamp:TIMESTAMP,model_name:STRING,request_id:STRING,features:JSON"
        Write-Info "Table created: $TABLE_NAME"
    }
}

function Create-TrainingDataTable {
    Write-Info "Creating training data table..."
    
    $TABLE_NAME = "training_data"
    $FULL_TABLE = "$PROJECT_ID:$DATASET_NAME.$TABLE_NAME"
    
    $exists = bq ls -t $FULL_TABLE 2>$null
    if ($exists) {
        Write-Warn "Table $TABLE_NAME already exists. Skipping creation."
    } else {
        bq mk --table `
            --description="Training Data for CTR Models" `
            --time_partitioning_field=timestamp `
            --time_partitioning_type=DAY `
            --label=managed-by:ctr-prediction-deployment `
            $FULL_TABLE `
            "user_id:STRING,ad_id:STRING,device:STRING,placement:STRING,clicked:BOOLEAN,timestamp:TIMESTAMP,features:JSON"
        Write-Info "Table created: $TABLE_NAME"
    }
}

# Main
Write-Info "=========================================="
Write-Info "Setting up BigQuery for CTR Predictions"
Write-Info "=========================================="
Write-Info ""

Check-Prerequisites
Create-Dataset
Create-PredictionsTable
Create-TrainingDataTable

Write-Info ""
Write-Info "=========================================="
Write-Info "Setup Complete!"
Write-Info "=========================================="
Write-Info ""
Write-Info "Dataset: $PROJECT_ID.$DATASET_NAME"
Write-Info "Tables:"
Write-Info "  - predictions"
Write-Info "  - training_data"
Write-Info ""
Write-Info "Query example:"
Write-Info "  bq query --use_legacy_sql=false \"SELECT COUNT(*) FROM \`$PROJECT_ID.$DATASET_NAME.predictions\`\""
Write-Info ""

