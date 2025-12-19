# PowerShell script to run Docker containers

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Running CTR Prediction System" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if models directory exists
if (-not (Test-Path "models") -or (Get-ChildItem "models\*.pkl" -ErrorAction SilentlyContinue).Count -eq 0) {
    Write-Host "[WARNING] No models found in models/ directory" -ForegroundColor Yellow
    Write-Host "   Please train models first:"
    Write-Host "   python scripts/train_models.py --input data/processed/features.csv --output-dir models/"
    Write-Host ""
    $response = Read-Host "Continue anyway? (y/n)"
    if ($response -ne "y" -and $response -ne "Y") {
        exit 1
    }
}

# Run with docker-compose
Write-Host "Starting containers..." -ForegroundColor Yellow
docker-compose up -d

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "[OK] Containers started!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Services available at:" -ForegroundColor Cyan
Write-Host "  - API: http://localhost:8000"
Write-Host "  - API Docs: http://localhost:8000/docs"
Write-Host "  - Streamlit UI: http://localhost:8501"
Write-Host ""
Write-Host "To view logs:" -ForegroundColor Yellow
Write-Host "  docker-compose logs -f"
Write-Host ""
Write-Host "To stop:" -ForegroundColor Yellow
Write-Host "  docker-compose down"
Write-Host ""


