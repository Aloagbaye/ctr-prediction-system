#!/usr/bin/env python3
"""
Script to create features from raw data

Usage:
    python scripts/create_features.py [--input PATH] [--output PATH] [OPTIONS]
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.pipeline import create_features


def main():
    parser = argparse.ArgumentParser(description='Create features from raw data')
    parser.add_argument('--input', type=str, default='data/raw/impressions.csv',
                       help='Input CSV/Parquet file path')
    parser.add_argument('--output', type=str, default='data/processed/features.csv',
                       help='Output file path')
    parser.add_argument('--target-col', type=str, default='clicked',
                       help='Name of target column')
    
    # Feature creation options
    parser.add_argument('--no-basic', action='store_true',
                       help='Skip basic feature creation')
    parser.add_argument('--no-advanced', action='store_true',
                       help='Skip advanced feature creation')
    parser.add_argument('--no-selection', action='store_true',
                       help='Skip feature selection')
    
    # Feature selection options
    parser.add_argument('--select-by-importance', action='store_true',
                       help='Select features by importance')
    parser.add_argument('--n-features', type=int, default=None,
                       help='Number of features to keep (if selecting by importance)')
    parser.add_argument('--no-remove-variance', action='store_true',
                       help='Skip removing low-variance features')
    parser.add_argument('--no-remove-correlated', action='store_true',
                       help='Skip removing correlated features')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("=" * 60)
    print()
    
    # Create features
    df = create_features(
        input_path=args.input,
        output_path=args.output,
        target_col=args.target_col,
        create_basic=not args.no_basic,
        create_advanced=not args.no_advanced,
        apply_selection=not args.no_selection,
        remove_low_variance=not args.no_remove_variance,
        remove_correlated=not args.no_remove_correlated,
        select_by_importance=args.select_by_importance,
        n_features=args.n_features
    )
    
    print("\n✓ Feature engineering complete!")
    print(f"✓ Features saved to: {args.output}")
    print(f"\nFeature summary:")
    print(f"  Total features: {len(df.columns)}")
    print(f"  Rows: {len(df):,}")
    
    # Show feature list
    feature_cols = [col for col in df.columns if col != args.target_col]
    print(f"\nFeature columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols[:20], 1):
        print(f"  {i}. {col}")
    if len(feature_cols) > 20:
        print(f"  ... and {len(feature_cols) - 20} more")


if __name__ == "__main__":
    main()

