"""
Kaggle Dataset Loader for Avazu CTR Prediction

Handles downloading and processing of Kaggle Avazu dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import os
from typing import Optional, Tuple
import subprocess
import sys


class AvazuDataLoader:
    """Load and process Avazu CTR prediction dataset from Kaggle"""
    
    def __init__(self, data_dir: str = "data/raw/avazu"):
        """
        Initialize Avazu data loader
        
        Args:
            data_dir: Directory to store Avazu data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def check_kaggle_credentials(self) -> bool:
        """Check if Kaggle API credentials are configured"""
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_json = kaggle_dir / "kaggle.json"
        
        if not kaggle_json.exists():
            print("⚠️  Kaggle API credentials not found!")
            print("\nTo download Kaggle datasets, you need to:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Scroll to 'API' section and click 'Create New API Token'")
            print("3. This downloads kaggle.json")
            print("4. Place it in ~/.kaggle/kaggle.json (or C:\\Users\\<username>\\.kaggle\\kaggle.json on Windows)")
            print("5. Set permissions: chmod 600 ~/.kaggle/kaggle.json (Linux/Mac)")
            return False
        
        return True
    
    def download_dataset(self, force: bool = False) -> bool:
        """
        Download Avazu dataset from Kaggle
        
        Args:
            force: Force re-download even if files exist
            
        Returns:
            True if successful, False otherwise
        """
        if not self.check_kaggle_credentials():
            return False
        
        train_file = self.data_dir / "train.gz"
        test_file = self.data_dir / "test.gz"
        
        # Check if already downloaded
        if not force and train_file.exists() and test_file.exists():
            print(f"✓ Dataset already exists at {self.data_dir}")
            print("  Use --force to re-download")
            return True
        
        print("Downloading Avazu CTR dataset from Kaggle...")
        print("This may take several minutes (dataset is ~1.5GB)...")
        
        try:
            # Use kaggle API to download
            cmd = [
                "kaggle", "competitions", "download", 
                "-c", "avazu-ctr-prediction",
                "-p", str(self.data_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Error downloading dataset:")
                print(result.stderr)
                print("\nTrying alternative method...")
                return self._download_alternative()
            
            # Extract zip files if needed
            zip_files = list(self.data_dir.glob("*.zip"))
            for zip_file in zip_files:
                print(f"Extracting {zip_file.name}...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                # Remove zip file after extraction
                zip_file.unlink()
            
            print(f"✓ Dataset downloaded to {self.data_dir}")
            return True
            
        except FileNotFoundError:
            print("❌ Kaggle CLI not found!")
            print("Install it with: pip install kaggle")
            return False
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return False
    
    def _download_alternative(self) -> bool:
        """Alternative download method using manual instructions"""
        print("\n" + "=" * 60)
        print("MANUAL DOWNLOAD INSTRUCTIONS")
        print("=" * 60)
        print("1. Go to: https://www.kaggle.com/c/avazu-ctr-prediction/data")
        print("2. Download the following files:")
        print("   - train.gz")
        print("   - test.gz")
        print(f"3. Place them in: {self.data_dir}")
        print("4. Then run the script again")
        print("=" * 60)
        return False
    
    def load_train_data(
        self, 
        n_rows: Optional[int] = None,
        sample_frac: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Load Avazu training data
        
        Args:
            n_rows: Number of rows to load (for testing)
            sample_frac: Fraction of data to sample (0.0 to 1.0)
            
        Returns:
            DataFrame with Avazu data
        """
        train_file = self.data_dir / "train.gz"
        
        if not train_file.exists():
            raise FileNotFoundError(
                f"Training file not found: {train_file}\n"
                "Run download_dataset() first or download manually from Kaggle."
            )
        
        print(f"Loading Avazu training data from {train_file}...")
        
        # Avazu dataset is large, so we can sample or limit rows
        if n_rows:
            df = pd.read_csv(train_file, nrows=n_rows, compression='gzip')
            print(f"Loaded {len(df):,} rows (limited to {n_rows:,})")
        elif sample_frac:
            # Read a sample
            total_rows = self._count_lines(train_file)
            n_rows = int(total_rows * sample_frac)
            df = pd.read_csv(train_file, nrows=n_rows, compression='gzip')
            print(f"Loaded {len(df):,} rows (sampled {sample_frac*100:.1f}% of dataset)")
        else:
            # Load full dataset (warning: very large!)
            print("⚠️  Loading full dataset - this may take a while and use significant memory...")
            df = pd.read_csv(train_file, compression='gzip')
            print(f"Loaded {len(df):,} rows")
        
        return df
    
    def _count_lines(self, filepath: Path) -> int:
        """Count lines in a gzipped file (approximate)"""
        try:
            import gzip
            with gzip.open(filepath, 'rt') as f:
                return sum(1 for _ in f)
        except:
            # Fallback: just return a large number
            return 10000000
    
    def convert_to_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Avazu format to standard format used in this project
        
        Avazu columns -> Standard columns mapping:
        - click -> clicked
        - hour -> timestamp (converted from YYMMDDHH format)
        - C1-C21 -> kept as categorical features
        - banner_pos, site_id, app_id, device_id, etc. -> kept as features
        
        Args:
            df: Avazu format DataFrame
            
        Returns:
            DataFrame in standard format
        """
        df_std = df.copy()
        
        # Rename target column
        if 'click' in df_std.columns:
            df_std['clicked'] = df_std['click'].astype(int)
            df_std = df_std.drop('click', axis=1)
        
        # Convert hour to timestamp
        if 'hour' in df_std.columns:
            # Avazu hour format: YYMMDDHH (e.g., 14102100 = Oct 21, 2014, 00:00)
            df_std['hour_str'] = df_std['hour'].astype(str).str.zfill(8)
            
            # Parse to datetime
            df_std['timestamp'] = pd.to_datetime(
                df_std['hour_str'],
                format='%y%m%d%H',
                errors='coerce'
            )
            
            # Extract temporal features
            df_std['hour'] = df_std['timestamp'].dt.hour
            df_std['day_of_week'] = df_std['timestamp'].dt.dayofweek
            df_std['is_weekend'] = (df_std['day_of_week'] >= 5).astype(int)
            df_std['date'] = df_std['timestamp'].dt.date
            
            # Drop intermediate column
            df_std = df_std.drop('hour_str', axis=1)
        
        # Map device_type to our device categories
        if 'device_type' in df_std.columns:
            # Avazu: 1=Desktop, 2=Mobile, 4=Tablet, 5=Other
            device_map = {
                1: 'desktop',
                2: 'mobile',
                4: 'tablet',
                5: 'other'
            }
            df_std['device'] = df_std['device_type'].map(device_map).fillna('other')
        
        # Map banner_pos to placement
        if 'banner_pos' in df_std.columns:
            # Map banner positions to our placement categories
            # This is approximate - Avazu has numeric positions
            placement_map = {
                0: 'header',
                1: 'sidebar',
                2: 'footer',
                3: 'in_content',
                4: 'popup'
            }
            df_std['placement'] = df_std['banner_pos'].map(
                lambda x: placement_map.get(x % 5, 'other')
            )
        
        # Create user_id from device_id and device_ip (if available)
        if 'device_id' in df_std.columns and 'device_ip' in df_std.columns:
            # Combine to create a user identifier
            df_std['user_id'] = (
                df_std['device_id'].astype(str) + '_' + 
                df_std['device_ip'].astype(str)
            )
        
        # Create ad_id from combination of ad features
        if 'C14' in df_std.columns:  # C14 is often used as ad identifier
            df_std['ad_id'] = df_std['C14'].astype(str)
        elif 'C1' in df_std.columns:
            df_std['ad_id'] = df_std['C1'].astype(str)
        
        # Reorder columns to match standard format (if possible)
        standard_cols = ['user_id', 'ad_id', 'device', 'placement', 'timestamp', 
                        'hour', 'day_of_week', 'is_weekend', 'date', 'clicked']
        
        # Keep all original columns, but prioritize standard ones
        final_cols = []
        for col in standard_cols:
            if col in df_std.columns:
                final_cols.append(col)
        
        # Add remaining columns
        for col in df_std.columns:
            if col not in final_cols:
                final_cols.append(col)
        
        df_std = df_std[final_cols]
        
        return df_std
    
    def load_and_prepare(
        self,
        n_rows: Optional[int] = None,
        sample_frac: Optional[float] = None,
        convert_format: bool = True
    ) -> pd.DataFrame:
        """
        Load and prepare Avazu data in one step
        
        Args:
            n_rows: Number of rows to load
            sample_frac: Fraction of data to sample
            convert_format: Whether to convert to standard format
            
        Returns:
            Prepared DataFrame
        """
        df = self.load_train_data(n_rows=n_rows, sample_frac=sample_frac)
        
        if convert_format:
            df = self.convert_to_standard_format(df)
        
        return df
    
    def save_processed(self, df: pd.DataFrame, output_path: str):
        """Save processed data to CSV"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # For large datasets, save as parquet or compressed CSV
        if len(df) > 1000000:
            # Save as parquet for large datasets
            parquet_path = output_path.with_suffix('.parquet')
            df.to_parquet(parquet_path, compression='snappy', index=False)
            print(f"✓ Saved to {parquet_path} (Parquet format for large dataset)")
        else:
            df.to_csv(output_path, index=False)
            print(f"✓ Saved to {output_path}")


def download_avazu_dataset(data_dir: str = "data/raw/avazu", force: bool = False) -> bool:
    """
    Convenience function to download Avazu dataset
    
    Args:
        data_dir: Directory to store data
        force: Force re-download
        
    Returns:
        True if successful
    """
    loader = AvazuDataLoader(data_dir)
    return loader.download_dataset(force=force)


def load_avazu_data(
    data_dir: str = "data/raw/avazu",
    n_rows: Optional[int] = None,
    sample_frac: Optional[float] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to load Avazu data
    
    Args:
        data_dir: Directory containing Avazu data
        n_rows: Number of rows to load
        sample_frac: Fraction to sample (0.0 to 1.0)
        output_path: Optional path to save processed data
        
    Returns:
        DataFrame with Avazu data
    """
    loader = AvazuDataLoader(data_dir)
    df = loader.load_and_prepare(n_rows=n_rows, sample_frac=sample_frac)
    
    if output_path:
        loader.save_processed(df, output_path)
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and load Avazu dataset')
    parser.add_argument('--download', action='store_true', help='Download dataset from Kaggle')
    parser.add_argument('--force', action='store_true', help='Force re-download')
    parser.add_argument('--n-rows', type=int, help='Number of rows to load')
    parser.add_argument('--sample-frac', type=float, help='Fraction of data to sample (0.0-1.0)')
    parser.add_argument('--output', type=str, help='Output path for processed data')
    parser.add_argument('--data-dir', type=str, default='data/raw/avazu', help='Data directory')
    
    args = parser.parse_args()
    
    loader = AvazuDataLoader(args.data_dir)
    
    if args.download:
        success = loader.download_dataset(force=args.force)
        if not success:
            sys.exit(1)
    
    if args.n_rows or args.sample_frac or args.output:
        df = loader.load_and_prepare(
            n_rows=args.n_rows,
            sample_frac=args.sample_frac
        )
        
        if args.output:
            loader.save_processed(df, args.output)
        else:
            print(f"\nLoaded {len(df):,} rows")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst few rows:")
            print(df.head())

