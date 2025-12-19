"""
Basic Feature Engineering

Creates fundamental features from raw data:
- Historical CTR features (user-level, ad-level, device-level)
- Time-based features
- Impression count features
- Time since last click
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime


class BasicFeatureEngineer:
    """Create basic features from raw impression data"""
    
    def __init__(self, target_col: str = 'clicked'):
        """
        Initialize feature engineer
        
        Args:
            target_col: Name of target column
        """
        self.target_col = target_col
        self.feature_stats = {}  # Store statistics for inference
    
    def create_historical_ctr_features(
        self, 
        df: pd.DataFrame,
        group_cols: List[str],
        suffix: str = ""
    ) -> pd.DataFrame:
        """
        Create historical CTR features for given grouping
        
        Args:
            df: DataFrame with impression data
            group_cols: Columns to group by (e.g., ['user_id'])
            suffix: Suffix for feature names
            
        Returns:
            DataFrame with added CTR features
        """
        df = df.copy()
        
        # Sort by timestamp to ensure chronological order
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        # Calculate CTR for each group
        for group_col in group_cols:
            feature_name = f'ctr_{group_col}{suffix}'
            
            # Calculate cumulative CTR (using only previous data)
            df[feature_name] = (
                df.groupby(group_col)[self.target_col]
                .transform(lambda x: x.shift(1).expanding().mean())
                .fillna(df[self.target_col].mean())  # Fill with global mean
            )
            
            # Also create count features
            count_name = f'impressions_{group_col}{suffix}'
            df[count_name] = df.groupby(group_col).cumcount()
        
        return df
    
    def create_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user-level features"""
        df = df.copy()
        
        if 'user_id' not in df.columns:
            return df
        
        # Historical user CTR
        df = self.create_historical_ctr_features(df, ['user_id'], suffix='_user')
        
        # User activity level
        user_impressions = df.groupby('user_id').size()
        df['user_activity_level'] = df['user_id'].map(user_impressions)
        
        # User click rate (overall)
        user_ctr = df.groupby('user_id')[self.target_col].mean()
        df['user_ctr_overall'] = df['user_id'].map(user_ctr)
        
        return df
    
    def create_ad_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ad-level features"""
        df = df.copy()
        
        if 'ad_id' not in df.columns:
            return df
        
        # Historical ad CTR
        df = self.create_historical_ctr_features(df, ['ad_id'], suffix='_ad')
        
        # Ad popularity (impression count)
        ad_impressions = df.groupby('ad_id').size()
        df['ad_impression_count'] = df['ad_id'].map(ad_impressions)
        
        # Ad CTR (overall)
        ad_ctr = df.groupby('ad_id')[self.target_col].mean()
        df['ad_ctr_overall'] = df['ad_id'].map(ad_ctr)
        
        return df
    
    def create_device_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create device-level features"""
        df = df.copy()
        
        if 'device' not in df.columns:
            return df
        
        # Device-level CTR
        df = self.create_historical_ctr_features(df, ['device'], suffix='_device')
        
        # Device CTR (overall)
        device_ctr = df.groupby('device')[self.target_col].mean()
        df['device_ctr_overall'] = df['device'].map(device_ctr)
        
        return df
    
    def create_placement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create placement-level features"""
        df = df.copy()
        
        if 'placement' not in df.columns:
            return df
        
        # Placement-level CTR
        placement_ctr = df.groupby('placement')[self.target_col].mean()
        df['placement_ctr_overall'] = df['placement'].map(placement_ctr)
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional temporal features"""
        df = df.copy()
        
        if 'timestamp' not in df.columns:
            return df
        
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract additional temporal features
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        
        # Business hours (9 AM - 5 PM)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Morning (6-12), Afternoon (12-18), Evening (18-24), Night (0-6)
        df['time_of_day'] = pd.cut(
            df['hour'],
            bins=[-1, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening']
        ).astype(str)
        
        return df
    
    def create_time_since_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time since last click/impression features"""
        df = df.copy()
        
        if 'timestamp' not in df.columns or 'user_id' not in df.columns:
            return df
        
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by user and timestamp
        df = df.sort_values(['user_id', 'timestamp'])
        
        # Time since last impression (for same user)
        df['time_since_last_impression'] = (
            df.groupby('user_id')['timestamp']
            .diff()
            .dt.total_seconds()
            .fillna(0)
        )
        
        # Time since last click (for same user)
        user_clicks = df[df[self.target_col] == 1].copy()
        if len(user_clicks) > 0:
            user_clicks = user_clicks.sort_values(['user_id', 'timestamp'])
            user_clicks['time_since_last_click'] = (
                user_clicks.groupby('user_id')['timestamp']
                .diff()
                .dt.total_seconds()
                .fillna(0)
            )
            
            # Merge back
            df = df.merge(
                user_clicks[['user_id', 'timestamp', 'time_since_last_click']],
                on=['user_id', 'timestamp'],
                how='left'
            )
            # Use a large finite value instead of np.inf to avoid issues with sklearn
            # Max time span in seconds (e.g., 1 year = 31536000 seconds)
            max_time_span = 31536000
            df['time_since_last_click'] = df['time_since_last_click'].fillna(max_time_span)
        else:
            # Use a large finite value instead of np.inf
            max_time_span = 31536000
            df['time_since_last_click'] = max_time_span
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        df = df.copy()
        
        # Device-placement interaction
        if 'device' in df.columns and 'placement' in df.columns:
            df['device_placement'] = df['device'].astype(str) + '_' + df['placement'].astype(str)
            device_placement_ctr = df.groupby('device_placement')[self.target_col].mean()
            df['device_placement_ctr'] = df['device_placement'].map(device_placement_ctr)
        
        # User-device interaction
        if 'user_id' in df.columns and 'device' in df.columns:
            user_device_ctr = df.groupby(['user_id', 'device'])[self.target_col].mean()
            df['user_device_ctr'] = df.set_index(['user_id', 'device']).index.map(
                lambda x: user_device_ctr.get(x, df[self.target_col].mean())
            )
        
        # Hour-placement interaction
        if 'hour' in df.columns and 'placement' in df.columns:
            hour_placement_ctr = df.groupby(['hour', 'placement'])[self.target_col].mean()
            df['hour_placement_ctr'] = df.set_index(['hour', 'placement']).index.map(
                lambda x: hour_placement_ctr.get(x, df[self.target_col].mean())
            )
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all basic features
        
        Args:
            df: Raw DataFrame
            
        Returns:
            DataFrame with all basic features
        """
        print("Creating basic features...")
        
        df = self.create_temporal_features(df)
        df = self.create_user_features(df)
        df = self.create_ad_features(df)
        df = self.create_device_features(df)
        df = self.create_placement_features(df)
        df = self.create_time_since_features(df)
        df = self.create_interaction_features(df)
        
        print(f"✓ Created basic features. Total columns: {len(df.columns)}")
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to new data (using stored statistics)
        
        Args:
            df: New DataFrame
            
        Returns:
            DataFrame with features
        """
        # For now, just call fit_transform
        # In production, would use stored statistics
        return self.fit_transform(df)

