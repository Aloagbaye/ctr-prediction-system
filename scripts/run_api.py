#!/usr/bin/env python3
"""
Script to run the CTR Prediction API

Usage:
    python scripts/run_api.py [--host HOST] [--port PORT] [--reload]
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description='Run CTR Prediction API')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind to (default: 8000)')
    parser.add_argument('--reload', action='store_true',
                       help='Enable auto-reload for development')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes (default: 1)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CTR PREDICTION API")
    print("=" * 60)
    print(f"Starting API server...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    print(f"Reload: {args.reload}")
    print("=" * 60)
    print("\nAPI will be available at:")
    print(f"  http://{args.host}:{args.port}")
    print(f"  http://{args.host}:{args.port}/docs (Swagger UI)")
    print(f"  http://{args.host}:{args.port}/redoc (ReDoc)")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    # Run the API
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1  # Reload doesn't work with multiple workers
    )


if __name__ == "__main__":
    main()

