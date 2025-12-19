"""
Advanced Feature Engineering

Handles high-cardinality categorical features using:
- Feature hashing (hashing trick)
- Target encoding (mean encoding) with cross-validation
- Frequency encoding
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from sklearn.model_selection import KFold
import hashlib


class AdvancedFeatureEngineer:
    """Create advanced features for high-cardinality categoricals"""
    
    def __init__(self, target_col: str = 'clicked', n_splits: int = 5, random_state: int = 42):
        """
        Initialize advanced feature engineer
        
        Args:
            target_col: Name of target column
            n_splits: Number of folds for cross-validation in target encoding
            random_state: Random seed
        """
        self.target_col = target_col
        self.n_splits = n_splits
        self.random_state = random_state
        self.encoding_maps = {}  # Store encoding mappings
        self.global_mean = None
    
    def feature_hashing(
        self, 
        df: pd.DataFrame, 
        col: str, 
        n_features: int = 10
    ) -> pd.DataFrame:
        """
        Apply feature hashing (hashing trick) to a column
        
        Args:
            df: DataFrame
            col: Column name to hash
            n_features: Number of hash features to create
            
        Returns:
            DataFrame with hash features added
        """
        df = df.copy()
        
        def hash_value(value, n_features):
            """Hash a value to n_features dimensions"""
            value_str = str(value)
            hash_obj = hashlib.md5(value_str.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            return hash_int % n_features
        
        # Create hash features
        for i in range(n_features):
            df[f'{col}_hash_{i}'] = df[col].apply(
                lambda x: 1 if hash_value(x, n_features) == i else 0
            )
        
        return df
    
    def frequency_encoding(
        self, 
        df: pd.DataFrame, 
        col: str,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Create frequency encoding for a categorical column
        
        Args:
            df: DataFrame
            col: Column name
            fit: Whether to fit (learn) or transform (use existing mapping)
            
        Returns:
            DataFrame with frequency encoding
        """
        df = df.copy()
        
        if fit:
            # Calculate frequencies
            freq_map = df[col].value_counts().to_dict()
            self.encoding_maps[f'{col}_freq'] = freq_map
        else:
            freq_map = self.encoding_maps.get(f'{col}_freq', {})
        
        # Apply frequency encoding
        df[f'{col}_freq'] = df[col].map(freq_map).fillna(0)
        
        # Also create normalized frequency (0-1)
        if fit:
            max_freq = df[f'{col}_freq'].max()
            self.encoding_maps[f'{col}_freq_max'] = max_freq
        else:
            max_freq = self.encoding_maps.get(f'{col}_freq_max', 1)
        
        df[f'{col}_freq_norm'] = df[f'{col}_freq'] / max_freq if max_freq > 0 else 0
        
        return df
    
    def target_encoding_cv(
        self,
        df: pd.DataFrame,
        col: str,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Create target encoding (mean encoding) with cross-validation
        
        This prevents data leakage by using out-of-fold means.
        
        Args:
            df: DataFrame
            col: Column name
            fit: Whether to fit or transform
            
        Returns:
            DataFrame with target encoding
        """
        df = df.copy()
        
        if fit:
            # Store global mean
            self.global_mean = df[self.target_col].mean()
            
            # Initialize encoding column
            df[f'{col}_target_enc'] = np.nan
            
            # Cross-validation for target encoding
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            
            for train_idx, val_idx in kf.split(df):
                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]
                
                # Calculate mean target for each category in training fold
                mean_encoding = train_df.groupby(col)[self.target_col].mean()
                
                # Apply to validation fold
                df.loc[val_idx, f'{col}_target_enc'] = val_df[col].map(mean_encoding)
            
            # Fill missing values with global mean
            df[f'{col}_target_enc'] = df[f'{col}_target_enc'].fillna(self.global_mean)
            
            # Store mean encoding for transform
            mean_encoding = df.groupby(col)[self.target_col].mean()
            self.encoding_maps[f'{col}_target_enc'] = mean_encoding.to_dict()
        else:
            # Use stored encoding
            encoding_map = self.encoding_maps.get(f'{col}_target_enc', {})
            global_mean = self.global_mean if self.global_mean is not None else df[self.target_col].mean()
            df[f'{col}_target_enc'] = df[col].map(encoding_map).fillna(global_mean)
        
        return df
    
    def encode_high_cardinality_features(
        self,
        df: pd.DataFrame,
        high_card_cols: Optional[List[str]] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Apply encoding to high-cardinality categorical features
        
        Args:
            df: DataFrame
            high_card_cols: List of high-cardinality columns (auto-detect if None)
            fit: Whether to fit or transform
            
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        # Auto-detect high-cardinality features if not provided
        if high_card_cols is None:
            high_card_cols = []
            for col in df.columns:
                if df[col].dtype in ['object', 'int64']:
                    n_unique = df[col].nunique()
                    # Consider high-cardinality if > 100 unique values
                    if n_unique > 100 and col != self.target_col:
                        high_card_cols.append(col)
        
        print(f"Encoding {len(high_card_cols)} high-cardinality features...")
        
        for col in high_card_cols:
            if col not in df.columns:
                continue
            
            # Apply frequency encoding
            df = self.frequency_encoding(df, col, fit=fit)
            
            # Apply target encoding (only if target column exists)
            if self.target_col in df.columns:
                df = self.target_encoding_cv(df, col, fit=fit)
            
            # Apply feature hashing (for very high cardinality)
            if df[col].nunique() > 1000:
                df = self.feature_hashing(df, col, n_features=10)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, high_card_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit and transform data
        
        Args:
            df: DataFrame
            high_card_cols: High-cardinality columns to encode
            
        Returns:
            DataFrame with advanced features
        """
        return self.encode_high_cardinality_features(df, high_card_cols, fit=True)
    
    def transform(self, df: pd.DataFrame, high_card_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Transform new data using fitted encodings
        
        Args:
            df: DataFrame
            high_card_cols: High-cardinality columns to encode
            
        Returns:
            DataFrame with advanced features
        """
        return self.encode_high_cardinality_features(df, high_card_cols, fit=False)

