# PowerShell script for GCP Deployment - SPLATTING VERSION
$ErrorActionPreference = "Continue"
$ProgressPreference = 'SilentlyContinue'

# Configuration
$PROJECT_ID = if ($env:GCP_PROJECT_ID) { $env:GCP_PROJECT_ID } else { "" }
$REGION = if ($env:GCP_REGION) { $env:GCP_REGION } else { "us-central1" }
$RESOURCE_PREFIX = "ctr-prediction"

function Log { 
    param([string]$msg) 
    Write-Host "[INFO] $msg" 
}

function Log-Error { 
    param([string]$msg) 
    Write-Host "[ERROR] $msg" -ForegroundColor Red
}

function Get-ShortPath {
    param([Parameter(Mandatory=$true)][string]$Path)
    if (-not (Test-Path $Path)) { return $Path }
    try {
        Add-Type -Namespace Win32 -Name PathUtil -MemberDefinition @"
using System;
using System.Runtime.InteropServices;
public static class PathUtil {
  [DllImport("kernel32.dll", CharSet = CharSet.Auto)]
  public static extern int GetShortPathName(string longPath, System.Text.StringBuilder shortPath, int shortPathLength);
}
"@ -ErrorAction SilentlyContinue | Out-Null

        $sb = New-Object System.Text.StringBuilder 1024
        $len = [Win32.PathUtil]::GetShortPathName((Resolve-Path $Path).Path, $sb, $sb.Capacity)
        if ($len -gt 0) { return $sb.ToString() }
    } catch { }
    return $Path
}


function Log-Warn { 
    param([string]$msg) 
    Write-Host "[WARN] $msg" -ForegroundColor Yellow
}

function Check-Prerequisites {
    Log "Checking prerequisites..."
    
    if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
        Log-Error "gcloud CLI not installed"
        exit 1
    }
    
    if ([string]::IsNullOrEmpty($PROJECT_ID)) {
        $script:PROJECT_ID = (gcloud config get-value project 2>&1).Trim()
        if ([string]::IsNullOrEmpty($script:PROJECT_ID)) {
            Log-Error "GCP_PROJECT_ID not set"
            exit 1
        }
    }
    
    gcloud config set project $PROJECT_ID | Out-Null
    gcloud services enable run.googleapis.com cloudbuild.googleapis.com storage-component.googleapis.com artifactregistry.googleapis.com --project=$PROJECT_ID | Out-Null
    
    Log "Prerequisites OK"
}

function Create-StorageBucket {
    Log "Creating storage bucket..."
    $BUCKET_NAME = "${RESOURCE_PREFIX}-models-${PROJECT_ID}"
    
    try {
        gsutil ls -b "gs://${BUCKET_NAME}" 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Log "Bucket exists: $BUCKET_NAME"
            return $BUCKET_NAME
        }
    } catch { }
    
    gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION "gs://${BUCKET_NAME}"
    Log "Bucket created: $BUCKET_NAME"
    
    return $BUCKET_NAME
}

function Build-AndPushImages {
    Log "Building images..."
    $ARTIFACT_REGISTRY = "${REGION}-docker.pkg.dev/${PROJECT_ID}/${RESOURCE_PREFIX}-repo"
    
    try {
        gcloud artifacts repositories describe "${RESOURCE_PREFIX}-repo" --location=$REGION --project=$PROJECT_ID 2>&1 | Out-Null
    } catch {
        gcloud artifacts repositories create "${RESOURCE_PREFIX}-repo" --repository-format=docker --location=$REGION --project=$PROJECT_ID
    }
    
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet | Out-Null
    
    Log "Building API image..."
    docker build -f Dockerfile.api -t "${ARTIFACT_REGISTRY}/api:latest" .
    if ($LASTEXITCODE -ne 0) { 
        Log-Error "Docker build failed for API image"
        exit 1 
    }
    Log "Pushing API image..."
    docker push "${ARTIFACT_REGISTRY}/api:latest"
    if ($LASTEXITCODE -ne 0) { 
        Log-Error "Docker push failed for API image"
        exit 1 
    }
    
    Log "Building Streamlit image..."
    docker build -f Dockerfile.streamlit -t "${ARTIFACT_REGISTRY}/streamlit:latest" .
    if ($LASTEXITCODE -ne 0) { 
        Log-Error "Docker build failed for Streamlit image"
        exit 1 
    }
    Log "Pushing Streamlit image..."
    docker push "${ARTIFACT_REGISTRY}/streamlit:latest"
    if ($LASTEXITCODE -ne 0) { 
        Log-Error "Docker push failed for Streamlit image"
        exit 1 
    }
    
    return $ARTIFACT_REGISTRY
}

function Deploy-ApiService {
    param([string]$ArtifactRegistry, [string]$BucketName)
    
    Log "Deploying API service..."
    
    $SERVICE_NAME = "${RESOURCE_PREFIX}-api"
    $IMAGE = "${ArtifactRegistry}/api:latest"
    
    Log "Service: $SERVICE_NAME"
    Log "Image: $IMAGE"
    
    # Create env vars file in YAML format
    $tmpDir = "C:\Temp"
    if (-not (Test-Path $tmpDir)) { New-Item -ItemType Directory -Path $tmpDir | Out-Null }
    $envFile = Join-Path $tmpDir ([System.IO.Path]::GetRandomFileName() + ".yaml")
    $envContent = @"
MODELS_DIR: /tmp/models
GCP_PROJECT_ID: $PROJECT_ID
GCS_BUCKET: $BucketName
"@
    [System.IO.File]::WriteAllText($envFile, $envContent, [System.Text.Encoding]::UTF8)
    
    Log "Deploying service (this may take 2-5 minutes)..."
    
    # Use argument array instead of splatting (more compatible with external commands)
    $gcloudArgs = @(
        "run", "deploy", $SERVICE_NAME,
        "--image=$IMAGE",
        "--platform=managed",
        "--region=$REGION",
        "--project=$PROJECT_ID",
        "--allow-unauthenticated",
        "--memory=2Gi",
        "--cpu=2",
        "--timeout=300",
        "--max-instances=10",
        "--min-instances=0",
        "--env-vars-file=$(Get-ShortPath $envFile)"
    )
    
    & gcloud @gcloudArgs
    
    $deployExitCode = $LASTEXITCODE
    Remove-Item $envFile -ErrorAction SilentlyContinue
    
    if ($deployExitCode -ne 0) {
        Log-Error "API deployment failed with exit code: $deployExitCode"
        exit 1
    }
    
    Log "Deployment completed successfully!"
    
    Log "Retrieving API service URL..."
    $API_URL = gcloud run services describe $SERVICE_NAME --region=$REGION --project=$PROJECT_ID --format="value(status.url)" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Log-Warn "Could not retrieve API URL"
        $API_URL = ""
    } else {
        $API_URL = $API_URL.Trim()
        if ([string]::IsNullOrEmpty($API_URL)) {
            Log-Warn "API URL is empty"
        } else {
            Log "API deployed: $API_URL"
        }
    }
    return $API_URL
}

function Deploy-StreamlitService {
    param([string]$ArtifactRegistry, [string]$ApiUrl)
    
    Log "Deploying Streamlit UI..."
    
    $SERVICE_NAME = "${RESOURCE_PREFIX}-ui"
    $IMAGE = "${ArtifactRegistry}/streamlit:latest"
    
    Log "Service: $SERVICE_NAME"
    Log "Image: $IMAGE"
    
    # Create env vars file in YAML format
    $tmpDir = "C:\Temp"
    if (-not (Test-Path $tmpDir)) { New-Item -ItemType Directory -Path $tmpDir | Out-Null }
    $envFile = Join-Path $tmpDir ([System.IO.Path]::GetRandomFileName() + ".yaml")
    $envContent = "API_URL: $ApiUrl"
    [System.IO.File]::WriteAllText($envFile, $envContent, [System.Text.Encoding]::UTF8)
    
    Log "Deploying service (this may take 2-5 minutes)..."
    
    # Use argument array
    $gcloudArgs = @(
        "run", "deploy", $SERVICE_NAME,
        "--image=$IMAGE",
        "--platform=managed",
        "--region=$REGION",
        "--project=$PROJECT_ID",
        "--allow-unauthenticated",
        "--memory=1Gi",
        "--cpu=1",
        "--timeout=300",
        "--max-instances=5",
        "--min-instances=0",
        "--env-vars-file=$(Get-ShortPath $envFile)"
    )
    
    & gcloud @gcloudArgs
    
    $deployExitCode = $LASTEXITCODE
    Remove-Item $envFile -ErrorAction SilentlyContinue
    
    if ($deployExitCode -ne 0) {
        Log-Error "UI deployment failed with exit code: $deployExitCode"
        exit 1
    }
    
    Log "Deployment completed successfully!"
    
    Log "Retrieving UI service URL..."
    $UI_URL = gcloud run services describe $SERVICE_NAME --region=$REGION --project=$PROJECT_ID --format="value(status.url)" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Log-Warn "Could not retrieve UI URL"
        $UI_URL = ""
    } else {
        $UI_URL = $UI_URL.Trim()
        if ([string]::IsNullOrEmpty($UI_URL)) {
            Log-Warn "UI URL is empty"
        } else {
            Log "UI deployed: $UI_URL"
        }
    }
    return $UI_URL
}

function Get-ServiceUrl {
    param([string]$ServiceName)
    
    $url = gcloud run services describe $ServiceName --region=$REGION --project=$PROJECT_ID --format="value(status.url)" 2>&1
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($url)) {
        return $url.Trim()
    }
    
    return ""
}

# Main
Log "=========================================="
Log "GCP Deployment for CTR Prediction System"
Log "=========================================="

Check-Prerequisites
$BUCKET_NAME = Create-StorageBucket
$ARTIFACT_REGISTRY = Build-AndPushImages
$API_URL = Deploy-ApiService $ARTIFACT_REGISTRY $BUCKET_NAME
$UI_URL = Deploy-StreamlitService $ARTIFACT_REGISTRY $API_URL

# Try to get URLs again if they're empty
if ([string]::IsNullOrEmpty($API_URL)) {
    Log "Retrieving API URL..."
    $API_URL = Get-ServiceUrl "${RESOURCE_PREFIX}-api"
}
if ([string]::IsNullOrEmpty($UI_URL)) {
    Log "Retrieving UI URL..."
    $UI_URL = Get-ServiceUrl "${RESOURCE_PREFIX}-ui"
}

Log ""
Log "=========================================="
Log "Deployment Complete!"
Log "=========================================="
if (-not [string]::IsNullOrEmpty($API_URL)) {
    Log "API Service: $API_URL"
    Log "API Docs: ${API_URL}/docs"
} else {
    Log "API Service: (URL not available)"
    Log "  Run: gcloud run services describe ${RESOURCE_PREFIX}-api --region=$REGION --project=$PROJECT_ID --format='value(status.url)'"
}
if (-not [string]::IsNullOrEmpty($UI_URL)) {
    Log "Streamlit UI: $UI_URL"
} else {
    Log "Streamlit UI: (URL not available)"
    Log "  Run: gcloud run services describe ${RESOURCE_PREFIX}-ui --region=$REGION --project=$PROJECT_ID --format='value(status.url)'"
}
Log ""
Log "To list all services:"
Log "  gcloud run services list --region=$REGION --project=$PROJECT_ID"
Log "=========================================="