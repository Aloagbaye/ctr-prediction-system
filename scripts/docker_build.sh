#!/bin/bash
# Script to build Docker images

set -e

echo "=========================================="
echo "Building CTR Prediction Docker Images"
echo "=========================================="
echo ""

# Build API image
echo "Building API image..."
docker build -f Dockerfile.api -t ctr-prediction-api:latest .

echo ""
echo "Building Streamlit image..."
docker build -f Dockerfile.streamlit -t ctr-prediction-streamlit:latest .

echo ""
echo "=========================================="
echo "✓ Build complete!"
echo "=========================================="
echo ""
echo "Images created:"
echo "  - ctr-prediction-api:latest"
echo "  - ctr-prediction-streamlit:latest"
echo ""
echo "To run with docker-compose:"
echo "  docker-compose up"
echo ""


