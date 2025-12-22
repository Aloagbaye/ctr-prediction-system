#!/usr/bin/env python3
"""
Script to run the A/B Test Dashboard Streamlit app

Usage:
    python scripts/run_ab_test_dashboard.py [--port PORT]
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess
import os


def main():
    parser = argparse.ArgumentParser(description='Run A/B Test Dashboard')
    parser.add_argument('--port', type=int, default=8502,
                       help='Port to run Streamlit on (default: 8502)')
    parser.add_argument('--api-url', type=str, default=None,
                       help='API URL (default: http://localhost:8000 or API_URL env var)')
    
    args = parser.parse_args()
    
    # Set API URL if provided
    if args.api_url:
        os.environ['API_URL'] = args.api_url
    
    # Get the dashboard script path
    dashboard_path = Path(__file__).parent.parent / "src" / "app" / "ab_test_dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard not found: {dashboard_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("A/B Test Dashboard")
    print("=" * 60)
    print(f"Dashboard: {dashboard_path}")
    print(f"Port: {args.port}")
    print(f"API URL: {os.getenv('API_URL', 'http://localhost:8000')}")
    print("=" * 60)
    print("\nStarting Streamlit...")
    print(f"Open http://localhost:{args.port} in your browser")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", str(args.port),
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n\nStopping dashboard...")


if __name__ == "__main__":
    main()

