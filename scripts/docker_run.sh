#!/bin/bash
# Script to run Docker containers

set -e

echo "=========================================="
echo "Running CTR Prediction System"
echo "=========================================="
echo ""

# Check if models directory exists
if [ ! -d "models" ] || [ -z "$(ls -A models/*.pkl 2>/dev/null)" ]; then
    echo "⚠️  Warning: No models found in models/ directory"
    echo "   Please train models first:"
    echo "   python scripts/train_models.py --input data/processed/features.csv --output-dir models/"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run with docker-compose
echo "Starting containers..."
docker-compose up -d

echo ""
echo "=========================================="
echo "✓ Containers started!"
echo "=========================================="
echo ""
echo "Services available at:"
echo "  - API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Streamlit UI: http://localhost:8501"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f"
echo ""
echo "To stop:"
echo "  docker-compose down"
echo ""


