#!/usr/bin/env python3
"""
Script to download and prepare Avazu dataset from Kaggle

Usage:
    python scripts/download_avazu.py [--download] [--load] [--n-rows N] [--sample-frac F]
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.kaggle_loader import AvazuDataLoader


def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare Avazu CTR dataset from Kaggle'
    )
    parser.add_argument('--download', action='store_true', 
                       help='Download dataset from Kaggle')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if files exist')
    parser.add_argument('--load', action='store_true',
                       help='Load and process the dataset')
    parser.add_argument('--n-rows', type=int, default=None,
                       help='Number of rows to load (for testing)')
    parser.add_argument('--sample-frac', type=float, default=None,
                       help='Fraction of data to sample (0.0-1.0, e.g., 0.1 for 10%%)')
    parser.add_argument('--output', type=str, default='data/raw/avazu_processed.csv',
                       help='Output path for processed data')
    parser.add_argument('--data-dir', type=str, default='data/raw/avazu',
                       help='Directory to store raw Avazu data')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AVAZU DATASET DOWNLOADER & LOADER")
    print("=" * 60)
    
    loader = AvazuDataLoader(args.data_dir)
    
    # Download if requested
    if args.download:
        print("\n📥 Downloading Avazu dataset from Kaggle...")
        success = loader.download_dataset(force=args.force)
        if not success:
            print("\n❌ Download failed. Please check Kaggle credentials.")
            sys.exit(1)
        print("✓ Download complete!")
    
    # Load and process if requested
    if args.load:
        print("\n📊 Loading and processing Avazu data...")
        
        if args.n_rows:
            print(f"Loading {args.n_rows:,} rows...")
        elif args.sample_frac:
            print(f"Sampling {args.sample_frac*100:.1f}% of dataset...")
        else:
            print("⚠️  Loading full dataset (this may take a while)...")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                print("Cancelled.")
                sys.exit(0)
        
        try:
            df = loader.load_and_prepare(
                n_rows=args.n_rows,
                sample_frac=args.sample_frac
            )
            
            print(f"\n✓ Loaded {len(df):,} rows")
            print(f"✓ Columns: {len(df.columns)}")
            print(f"✓ Overall CTR: {df['clicked'].mean():.4f}")
            
            # Save processed data
            if args.output:
                loader.save_processed(df, args.output)
                print(f"\n✓ Processed data saved to: {args.output}")
            
            print("\nDataset preview:")
            print(df.head())
            print(f"\nDataset info:")
            print(df.info())
            
        except FileNotFoundError as e:
            print(f"\n❌ Error: {e}")
            print("\nPlease download the dataset first:")
            print("  python scripts/download_avazu.py --download")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    if not args.download and not args.load:
        print("\nNo action specified. Use --download to download or --load to load data.")
        print("\nExample usage:")
        print("  # Download dataset")
        print("  python scripts/download_avazu.py --download")
        print("\n  # Load sample (10%%)")
        print("  python scripts/download_avazu.py --load --sample-frac 0.1")
        print("\n  # Load specific number of rows")
        print("  python scripts/download_avazu.py --load --n-rows 100000")


if __name__ == "__main__":
    main()

