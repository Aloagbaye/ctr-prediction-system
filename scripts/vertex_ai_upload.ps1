# PowerShell script to upload models to Vertex AI Model Registry

$ErrorActionPreference = "Stop"

# Configuration
$PROJECT_ID = if ($env:GCP_PROJECT_ID) { $env:GCP_PROJECT_ID } else { "" }
$REGION = if ($env:GCP_REGION) { $env:GCP_REGION } else { "us-central1" }
$MODELS_DIR = if ($env:MODELS_DIR) { $env:MODELS_DIR } else { "models" }
$GCS_BUCKET = if ($env:GCS_BUCKET) { $env:GCS_BUCKET } else { "" }

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
    
    if ([string]::IsNullOrEmpty($GCS_BUCKET)) {
        $GCS_BUCKET = "ctr-prediction-models-${PROJECT_ID}"
        Write-Info "Using bucket: $GCS_BUCKET"
    }
    
    gcloud config set project $PROJECT_ID | Out-Null
}

function Upload-ModelToGCS {
    param(
        [string]$ModelFile
    )
    
    $localPath = Join-Path $MODELS_DIR $ModelFile
    if (-not (Test-Path $localPath)) {
        Write-Warn "Model file not found: $localPath"
        return $null
    }
    
    Write-Info "Uploading $ModelFile to GCS..."
    $gcsPath = "gs://${GCS_BUCKET}/vertex-ai/${ModelFile}"
    gsutil cp $localPath $gcsPath
    return $gcsPath
}

function Upload-ToVertexAI {
    param(
        [string]$ModelName,
        [string]$ModelFile,
        [string]$Version = "v1"
    )
    
    Write-Info "Uploading ${ModelName} to Vertex AI Model Registry..."
    
    gcloud ai models upload `
        --region=$REGION `
        --display-name="CTR Prediction ${ModelName}" `
        --description="CTR Prediction Model - ${ModelName}" `
        --container-image-uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest" `
        --artifact-uri="gs://${GCS_BUCKET}/vertex-ai/${ModelFile}" `
        --labels="managed-by=ctr-prediction-deployment,model-type=${ModelName}" `
        --project=$PROJECT_ID
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to upload ${ModelName}"
        return $false
    }
    
    Write-Info "Successfully uploaded ${ModelName}"
    return $true
}

# Main
Write-Info "=========================================="
Write-Info "Uploading Models to Vertex AI"
Write-Info "=========================================="
Write-Info ""

Check-Prerequisites

$MODELS = @("xgboost_model.pkl", "lightgbm_model.pkl", "logistic_model.pkl")

foreach ($modelFile in $MODELS) {
    $modelName = [System.IO.Path]::GetFileNameWithoutExtension($modelFile) -replace '_model', ''
    
    $gcsPath = Upload-ModelToGCS $modelFile
    if ($gcsPath) {
        Upload-ToVertexAI $modelName $modelFile "v1" | Out-Null
    }
}

Write-Info ""
Write-Info "=========================================="
Write-Info "Upload Complete!"
Write-Info "=========================================="
Write-Info ""
Write-Info "List models:"
Write-Info "  gcloud ai models list --region=$REGION"
Write-Info ""

