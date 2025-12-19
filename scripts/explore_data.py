#!/usr/bin/env python3
"""
Script to perform exploratory data analysis

Usage:
    python scripts/explore_data.py [--input PATH] [--output-dir DIR]
    
Supports:
    - CSV files (simulated data)
    - Parquet files (large datasets like Avazu)
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.explore import explore_dataset, CTRExplorer


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV or Parquet file"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    print(f"Loading data from {filepath}...")
    
    if filepath.suffix == '.parquet':
        df = pd.read_parquet(filepath)
    elif filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
    else:
        # Try CSV first, then parquet
        try:
            df = pd.read_csv(filepath)
        except:
            df = pd.read_parquet(filepath)
    
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def main():
    parser = argparse.ArgumentParser(description='Perform exploratory data analysis')
    parser.add_argument('--input', type=str, default='data/raw/impressions.csv', 
                       help='Input file path (CSV or Parquet)')
    parser.add_argument('--output-dir', type=str, default='data/eda', 
                       help='Output directory for plots and reports')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CTR PREDICTION SYSTEM - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    print()
    
    # Load data
    df = load_data(args.input)
    
    # Convert timestamp if present
    if 'timestamp' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Run EDA
    explorer = CTRExplorer(df, args.output_dir)
    explorer.run_full_eda()
    
    print("\n✓ EDA complete!")
    print(f"✓ Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

