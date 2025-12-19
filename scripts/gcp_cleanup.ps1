# PowerShell script for GCP Cleanup
# Deletes all resources created by the deployment script

$ErrorActionPreference = "Stop"

# Configuration
$PROJECT_ID = if ($env:GCP_PROJECT_ID) { $env:GCP_PROJECT_ID } else { "" }
$REGION = if ($env:GCP_REGION) { $env:GCP_REGION } else { "us-central1" }
$RESOURCE_PREFIX = "ctr-prediction"
$RESOURCE_LABEL = "managed-by=ctr-prediction-deployment"

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

function Confirm-Deletion {
    Write-Host ""
    Write-Warn "This will delete ALL resources with label: ${RESOURCE_LABEL}"
    Write-Warn "This includes:"
    Write-Warn "  - Cloud Run services (${RESOURCE_PREFIX}-api, ${RESOURCE_PREFIX}-ui)"
    Write-Warn "  - Artifact Registry repository (${RESOURCE_PREFIX}-repo)"
    Write-Warn "  - Cloud Storage bucket (${RESOURCE_PREFIX}-models-*)"
    Write-Host ""
    $confirm = Read-Host "Are you sure you want to continue? (yes/no)"
    if ($confirm -ne "yes") {
        Write-Info "Cleanup cancelled."
        exit 0
    }
}

function Delete-CloudRunServices {
    Write-Info "Deleting Cloud Run services..."
    
    $SERVICES = @("${RESOURCE_PREFIX}-api", "${RESOURCE_PREFIX}-ui")
    
    foreach ($service in $SERVICES) {
        $exists = gcloud run services describe $service --region=$REGION --project=$PROJECT_ID 2>$null
        if ($exists) {
            Write-Info "Deleting Cloud Run service: $service"
            gcloud run services delete $service `
                --region=$REGION `
                --project=$PROJECT_ID `
                --quiet 2>$null
        } else {
            Write-Warn "Service $service does not exist. Skipping."
        }
    }
}

function Delete-ArtifactRegistry {
    Write-Info "Deleting Artifact Registry repository..."
    
    $REPO_NAME = "${RESOURCE_PREFIX}-repo"
    
    $exists = gcloud artifacts repositories describe $REPO_NAME `
        --location=$REGION `
        --project=$PROJECT_ID 2>$null
    if ($exists) {
        Write-Info "Deleting Artifact Registry repository: $REPO_NAME"
        gcloud artifacts repositories delete $REPO_NAME `
            --location=$REGION `
            --project=$PROJECT_ID `
            --quiet 2>$null
    } else {
        Write-Warn "Repository $REPO_NAME does not exist. Skipping."
    }
}

function Delete-StorageBuckets {
    Write-Info "Deleting Cloud Storage buckets..."
    
    try {
        $buckets = gsutil ls | Where-Object { $_ -match "gs://${RESOURCE_PREFIX}-" }
        if ($buckets) {
            foreach ($bucket in $buckets) {
                Write-Info "Deleting bucket: $bucket"
                gsutil -m rm -r $bucket 2>$null
            }
        } else {
            Write-Warn "No buckets found with prefix ${RESOURCE_PREFIX}-"
        }
    } catch {
        Write-Warn "Error listing buckets: $_"
    }
}

# Main cleanup flow
Write-Info "=========================================="
Write-Info "GCP Cleanup for CTR Prediction System"
Write-Info "=========================================="
Write-Info ""

# Check if PROJECT_ID is set
if ([string]::IsNullOrEmpty($PROJECT_ID)) {
    $PROJECT_ID = gcloud config get-value project 2>$null
    if ([string]::IsNullOrEmpty($PROJECT_ID)) {
        Write-Error "GCP_PROJECT_ID environment variable is not set."
        exit 1
    }
    Write-Info "Using project from gcloud config: $PROJECT_ID"
}

gcloud config set project $PROJECT_ID | Out-Null

Confirm-Deletion

Delete-CloudRunServices
Delete-ArtifactRegistry
Delete-StorageBuckets

Write-Info ""
Write-Info "=========================================="
Write-Info "Cleanup Complete!"
Write-Info "=========================================="
Write-Info ""
Write-Info "All resources with label '${RESOURCE_LABEL}' have been deleted."
Write-Info ""

