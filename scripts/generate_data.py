#!/usr/bin/env python3
"""
Script to generate simulated ad impression data

Usage:
    python scripts/generate_data.py [--n-impressions N] [--output PATH]
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generator import generate_dataset


def main():
    parser = argparse.ArgumentParser(description='Generate simulated ad impression data')
    parser.add_argument('--n-users', type=int, default=10000, help='Number of unique users')
    parser.add_argument('--n-ads', type=int, default=1000, help='Number of unique ads')
    parser.add_argument('--n-impressions', type=int, default=100000, help='Total number of impressions')
    parser.add_argument('--output', type=str, default='data/raw/impressions.csv', help='Output file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CTR PREDICTION SYSTEM - DATA GENERATION")
    print("=" * 60)
    print(f"Users: {args.n_users:,}")
    print(f"Ads: {args.n_ads:,}")
    print(f"Impressions: {args.n_impressions:,}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    print()
    
    df = generate_dataset(
        output_path=args.output,
        n_users=args.n_users,
        n_ads=args.n_ads,
        n_impressions=args.n_impressions,
        seed=args.seed
    )
    
    print("\n✓ Data generation complete!")
    print(f"✓ Dataset saved to: {args.output}")
    print(f"✓ Shape: {df.shape}")
    print(f"✓ Overall CTR: {df['clicked'].mean():.4f}")


if __name__ == "__main__":
    main()

