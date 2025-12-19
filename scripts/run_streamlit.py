#!/usr/bin/env python3
"""
Script to run the Streamlit application

Usage:
    python scripts/run_streamlit.py [--port PORT]
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description='Run Streamlit CTR Prediction App')
    parser.add_argument('--port', type=int, default=8501,
                       help='Port to run Streamlit on (default: 8501)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host to bind to (default: localhost)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("STREAMLIT CTR PREDICTION APP")
    print("=" * 60)
    print(f"Starting Streamlit app...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print("=" * 60)
    print("\nApp will be available at:")
    print(f"  http://{args.host}:{args.port}")
    print("\nMake sure the API server is running:")
    print("  python scripts/run_api.py")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    # Run Streamlit
    streamlit_path = Path(__file__).parent.parent / "src" / "app" / "streamlit_app.py"
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(streamlit_path),
        "--server.port", str(args.port),
        "--server.address", args.host
    ])


if __name__ == "__main__":
    main()

