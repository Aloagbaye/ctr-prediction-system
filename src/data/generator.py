"""
Data Generator for Simulated Ad Impression Data

Generates realistic ad impression data with:
- user_id, ad_id, device, time, placement, clicked
- Realistic CTR patterns
- Temporal patterns
- User behavior patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import random


class CTRDataGenerator:
    """Generate simulated ad impression data for CTR prediction"""
    
    def __init__(
        self,
        n_users: int = 10000,
        n_ads: int = 1000,
        n_impressions: int = 100000,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        seed: int = 42
    ):
        """
        Initialize data generator
        
        Args:
            n_users: Number of unique users
            n_ads: Number of unique ads
            n_impressions: Total number of impressions to generate
            start_date: Start date for time series (default: 30 days ago)
            end_date: End date for time series (default: today)
            seed: Random seed for reproducibility
        """
        self.n_users = n_users
        self.n_ads = n_ads
        self.n_impressions = n_impressions
        self.seed = seed
        
        # Set default dates
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        self.start_date = start_date
        self.end_date = end_date
        
        # Set random seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Device types and placements
        self.devices = ['mobile', 'desktop', 'tablet']
        self.placements = ['header', 'sidebar', 'footer', 'in_content', 'popup']
        
        # Initialize user and ad characteristics
        self._initialize_user_profiles()
        self._initialize_ad_profiles()
    
    def _initialize_user_profiles(self):
        """Initialize user click propensity profiles"""
        # Create user segments with different CTR baselines
        self.user_profiles = {}
        for user_id in range(1, self.n_users + 1):
            # User segments: low, medium, high clickers
            segment = np.random.choice(['low', 'medium', 'high'], p=[0.6, 0.3, 0.1])
            
            # Base CTR by segment
            base_ctr = {
                'low': 0.01,      # 1% CTR
                'medium': 0.05,   # 5% CTR
                'high': 0.15      # 15% CTR
            }[segment]
            
            # Device preference (some users prefer certain devices)
            device_pref = np.random.choice(self.devices, p=[0.5, 0.3, 0.2])
            
            self.user_profiles[user_id] = {
                'base_ctr': base_ctr,
                'segment': segment,
                'device_pref': device_pref,
                'click_history': []
            }
    
    def _initialize_ad_profiles(self):
        """Initialize ad performance profiles"""
        self.ad_profiles = {}
        for ad_id in range(1, self.n_ads + 1):
            # Ad quality: poor, average, good
            quality = np.random.choice(['poor', 'average', 'good'], p=[0.3, 0.5, 0.2])
            
            # Base CTR by quality
            base_ctr = {
                'poor': 0.01,     # 1% CTR
                'average': 0.05,  # 5% CTR
                'good': 0.12      # 12% CTR
            }[quality]
            
            # Placement effectiveness
            placement_effect = {
                'header': 1.2,
                'sidebar': 0.8,
                'footer': 0.6,
                'in_content': 1.5,
                'popup': 0.4
            }
            
            self.ad_profiles[ad_id] = {
                'base_ctr': base_ctr,
                'quality': quality,
                'placement_effect': placement_effect
            }
    
    def _calculate_click_probability(
        self,
        user_id: int,
        ad_id: int,
        device: str,
        placement: str,
        hour: int,
        day_of_week: int
    ) -> float:
        """Calculate click probability based on features"""
        user_profile = self.user_profiles[user_id]
        ad_profile = self.ad_profiles[ad_id]
        
        # Base CTR from user and ad
        base_ctr = (user_profile['base_ctr'] + ad_profile['base_ctr']) / 2
        
        # Device effect (users prefer their preferred device)
        if device == user_profile['device_pref']:
            device_multiplier = 1.3
        else:
            device_multiplier = 1.0
        
        # Placement effect
        placement_multiplier = ad_profile['placement_effect'][placement]
        
        # Temporal effects
        # Higher CTR during business hours (9-17)
        if 9 <= hour <= 17:
            time_multiplier = 1.2
        else:
            time_multiplier = 0.9
        
        # Weekend effect (slightly lower CTR)
        if day_of_week >= 5:  # Saturday, Sunday
            day_multiplier = 0.95
        else:
            day_multiplier = 1.0
        
        # Calculate final probability
        probability = base_ctr * device_multiplier * placement_multiplier * time_multiplier * day_multiplier
        
        # Add some noise
        probability += np.random.normal(0, 0.01)
        
        # Clip to valid range
        probability = np.clip(probability, 0.001, 0.5)
        
        return probability
    
    def generate(self) -> pd.DataFrame:
        """Generate the complete dataset"""
        print(f"Generating {self.n_impressions} ad impressions...")
        
        data = []
        
        for i in range(self.n_impressions):
            # Random user and ad
            user_id = np.random.randint(1, self.n_users + 1)
            ad_id = np.random.randint(1, self.n_ads + 1)
            
            # Random device and placement
            device = np.random.choice(self.devices, p=[0.5, 0.3, 0.2])
            placement = np.random.choice(self.placements, p=[0.2, 0.2, 0.2, 0.3, 0.1])
            
            # Random timestamp
            time_delta = (self.end_date - self.start_date).total_seconds()
            random_seconds = np.random.randint(0, int(time_delta))
            timestamp = self.start_date + timedelta(seconds=random_seconds)
            
            # Extract temporal features
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Calculate click probability
            click_prob = self._calculate_click_probability(
                user_id, ad_id, device, placement, hour, day_of_week
            )
            
            # Sample click
            clicked = int(np.random.random() < click_prob)
            
            # Update user click history
            if clicked:
                self.user_profiles[user_id]['click_history'].append(timestamp)
            
            data.append({
                'user_id': user_id,
                'ad_id': ad_id,
                'device': device,
                'placement': placement,
                'timestamp': timestamp,
                'clicked': clicked
            })
            
            if (i + 1) % 10000 == 0:
                print(f"  Generated {i + 1:,} impressions...")
        
        df = pd.DataFrame(data)
        
        # Add derived temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['date'] = df['timestamp'].dt.date
        
        # Reorder columns
        df = df[['user_id', 'ad_id', 'device', 'placement', 'timestamp', 
                 'hour', 'day_of_week', 'is_weekend', 'date', 'clicked']]
        
        print(f"✓ Generated {len(df):,} impressions")
        print(f"✓ Overall CTR: {df['clicked'].mean():.4f} ({df['clicked'].sum():,} clicks)")
        
        return df
    
    def save(self, df: pd.DataFrame, filepath: str):
        """Save generated data to CSV"""
        df.to_csv(filepath, index=False)
        print(f"✓ Saved data to {filepath}")


def generate_dataset(
    output_path: str = "data/raw/impressions.csv",
    n_users: int = 10000,
    n_ads: int = 1000,
    n_impressions: int = 100000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Convenience function to generate and save dataset
    
    Args:
        output_path: Path to save the CSV file
        n_users: Number of unique users
        n_ads: Number of unique ads
        n_impressions: Total number of impressions
        seed: Random seed
        
    Returns:
        Generated DataFrame
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    generator = CTRDataGenerator(
        n_users=n_users,
        n_ads=n_ads,
        n_impressions=n_impressions,
        seed=seed
    )
    
    df = generator.generate()
    generator.save(df, output_path)
    
    return df


if __name__ == "__main__":
    # Generate sample dataset
    df = generate_dataset(
        output_path="data/raw/impressions.csv",
        n_users=10000,
        n_ads=1000,
        n_impressions=100000,
        seed=42
    )
    
    print("\nDataset Summary:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nColumn info:")
    print(df.info())

