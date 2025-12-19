"""
Feature Engineering Pipeline

Orchestrates all feature engineering steps:
1. Basic features
2. Advanced features (encoding)
3. Feature selection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import joblib

from .basic_features import BasicFeatureEngineer
from .advanced_features import AdvancedFeatureEngineer
from .feature_selector import FeatureSelector


class FeatureEngineeringPipeline:
    """Complete feature engineering pipeline"""
    
    def __init__(
        self,
        target_col: str = 'clicked',
        create_basic_features: bool = True,
        create_advanced_features: bool = True,
        apply_feature_selection: bool = True,
        random_state: int = 42
    ):
        """
        Initialize pipeline
        
        Args:
            target_col: Name of target column
            create_basic_features: Whether to create basic features
            create_advanced_features: Whether to create advanced features
            apply_feature_selection: Whether to apply feature selection
            random_state: Random seed
        """
        self.target_col = target_col
        self.create_basic_features = create_basic_features
        self.create_advanced_features = create_advanced_features
        self.apply_feature_selection = apply_feature_selection
        self.random_state = random_state
        
        # Initialize components
        self.basic_engineer = BasicFeatureEngineer(target_col=target_col)
        self.advanced_engineer = AdvancedFeatureEngineer(
            target_col=target_col,
            random_state=random_state
        )
        self.feature_selector = FeatureSelector(
            target_col=target_col,
            random_state=random_state
        )
        
        self.is_fitted = False
    
    def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Fit pipeline and transform data
        
        Args:
            df: Raw DataFrame
            **kwargs: Additional arguments for feature selection
            
        Returns:
            DataFrame with engineered features
        """
        print("=" * 60)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 60)
        print(f"Input shape: {df.shape}")
        print()
        
        # Step 1: Basic features
        if self.create_basic_features:
            print("Step 1: Creating basic features...")
            df = self.basic_engineer.fit_transform(df)
            print(f"  Shape after basic features: {df.shape}\n")
        
        # Step 2: Advanced features (encoding)
        if self.create_advanced_features:
            print("Step 2: Creating advanced features...")
            high_card_cols = kwargs.get('high_card_cols', None)
            df = self.advanced_engineer.fit_transform(df, high_card_cols=high_card_cols)
            print(f"  Shape after advanced features: {df.shape}\n")
        
        # Step 3: Feature selection
        if self.apply_feature_selection:
            print("Step 3: Applying feature selection...")
            df = self.feature_selector.fit_transform(
                df,
                remove_low_variance=kwargs.get('remove_low_variance', True),
                remove_correlated=kwargs.get('remove_correlated', True),
                select_by_importance=kwargs.get('select_by_importance', False),
                n_features=kwargs.get('n_features', None)
            )
            print(f"  Shape after feature selection: {df.shape}\n")
        
        self.is_fitted = True
        
        print("=" * 60)
        print(f"✓ Feature engineering complete!")
        print(f"  Final shape: {df.shape}")
        print(f"  Features: {len(df.columns)}")
        print("=" * 60)
        
        return df
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform new data using fitted pipeline
        
        Args:
            df: New DataFrame
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with engineered features
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform. Use fit_transform() first.")
        
        # Apply transformations in order
        if self.create_basic_features:
            df = self.basic_engineer.transform(df)
        
        if self.create_advanced_features:
            high_card_cols = kwargs.get('high_card_cols', None)
            df = self.advanced_engineer.transform(df, high_card_cols=high_card_cols)
        
        return df
    
    def save(self, filepath: str):
        """Save pipeline to disk"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self, filepath)
        print(f"✓ Pipeline saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'FeatureEngineeringPipeline':
        """Load pipeline from disk"""
        return joblib.load(filepath)


def create_features(
    input_path: str,
    output_path: str,
    target_col: str = 'clicked',
    create_basic: bool = True,
    create_advanced: bool = True,
    apply_selection: bool = True,
    sample_frac: Optional[float] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to create features from raw data
    
    Args:
        input_path: Path to input CSV/Parquet file
        output_path: Path to save processed data
        target_col: Name of target column
        create_basic: Whether to create basic features
        create_advanced: Whether to create advanced features
        apply_selection: Whether to apply feature selection
        sample_frac: Fraction of data to sample (0.0-1.0) for faster processing
        **kwargs: Additional arguments for pipeline
        
    Returns:
        DataFrame with engineered features
    """
    # Load data
    input_path = Path(input_path)
    if input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    
    print(f"Loaded data from {input_path}")
    print(f"Original shape: {df.shape}")
    
    # Sample data if requested (use stratified sampling to maintain class distribution)
    if sample_frac is not None:
        if 0 < sample_frac < 1:
            original_size = len(df)
            random_state = kwargs.get('random_state', 42)
            
            # Store original CTR for comparison
            original_ctr = None
            if target_col in df.columns:
                original_ctr = df[target_col].mean()
            
            # Use stratified sampling if target column exists
            if target_col in df.columns:
                from sklearn.model_selection import train_test_split
                # Use train_test_split for stratified sampling
                df_sampled, _ = train_test_split(
                    df,
                    test_size=1 - sample_frac,
                    stratify=df[target_col],
                    random_state=random_state
                )
                df = df_sampled
            else:
                # Simple random sampling if no target column
                df = df.sample(frac=sample_frac, random_state=random_state)
            
            print(f"Sampled {sample_frac*100:.1f}% of data: {len(df):,} rows (from {original_size:,})")
            if target_col in df.columns and original_ctr is not None:
                sampled_ctr = df[target_col].mean()
                print(f"  CTR maintained: {sampled_ctr:.4f} (original: {original_ctr:.4f})")
        elif sample_frac <= 0 or sample_frac >= 1:
            print(f"⚠️  Warning: sample_frac must be between 0 and 1, ignoring...")
    
    print(f"Working with shape: {df.shape}\n")
    
    # Create pipeline
    pipeline = FeatureEngineeringPipeline(
        target_col=target_col,
        create_basic_features=create_basic,
        create_advanced_features=create_advanced,
        apply_feature_selection=apply_selection
    )
    
    # Fit and transform
    df_processed = pipeline.fit_transform(df, **kwargs)
    
    # Save processed data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.parquet':
        df_processed.to_parquet(output_path, index=False)
    else:
        df_processed.to_csv(output_path, index=False)
    
    print(f"\n✓ Processed data saved to {output_path}")
    
    return df_processed

