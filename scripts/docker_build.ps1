# PowerShell script to build Docker images

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Building CTR Prediction Docker Images" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Build API image
Write-Host "Building API image..." -ForegroundColor Yellow
docker build -f Dockerfile.api -t ctr-prediction-api:latest .

Write-Host ""
Write-Host "Building Streamlit image..." -ForegroundColor Yellow
docker build -f Dockerfile.streamlit -t ctr-prediction-streamlit:latest .

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "[OK] Build complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Images created:" -ForegroundColor Cyan
Write-Host "  - ctr-prediction-api:latest"
Write-Host "  - ctr-prediction-streamlit:latest"
Write-Host ""
Write-Host "To run with docker-compose:" -ForegroundColor Yellow
Write-Host "  docker-compose up"
Write-Host ""


