"""
Exploratory Data Analysis (EDA) for CTR Prediction Dataset

Performs comprehensive data exploration including:
- Basic statistics
- Class imbalance analysis
- Feature distributions
- Temporal patterns
- Missing value analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class CTRExplorer:
    """Perform EDA on CTR prediction dataset"""
    
    def __init__(self, df: pd.DataFrame, output_dir: str = "data/eda"):
        """
        Initialize explorer
        
        Args:
            df: DataFrame with ad impression data
            output_dir: Directory to save plots and reports
        """
        self.df = df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure timestamp is datetime
        if 'timestamp' in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
    
    def basic_info(self) -> dict:
        """Get basic dataset information"""
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'date_range': {
                'start': self.df['timestamp'].min() if 'timestamp' in self.df.columns else None,
                'end': self.df['timestamp'].max() if 'timestamp' in self.df.columns else None
            }
        }
        
        print("=" * 60)
        print("DATASET BASIC INFORMATION")
        print("=" * 60)
        print(f"Shape: {info['shape']}")
        print(f"Memory usage: {info['memory_usage_mb']:.2f} MB")
        if info['date_range']['start']:
            print(f"Date range: {info['date_range']['start']} to {info['date_range']['end']}")
        print(f"\nColumns: {', '.join(info['columns'])}")
        print("\nData types:")
        for col, dtype in info['dtypes'].items():
            print(f"  {col}: {dtype}")
        
        return info
    
    def missing_values(self):
        """Analyze missing values"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'column': missing.index,
            'missing_count': missing.values,
            'missing_percentage': missing_pct.values
        }).sort_values('missing_count', ascending=False)
        
        print("\n" + "=" * 60)
        print("MISSING VALUES ANALYSIS")
        print("=" * 60)
        if missing.sum() == 0:
            print("✓ No missing values found!")
        else:
            print(missing_df[missing_df['missing_count'] > 0].to_string(index=False))
        
        return missing_df
    
    def class_imbalance(self):
        """Analyze class imbalance in target variable"""
        if 'clicked' not in self.df.columns:
            print("Warning: 'clicked' column not found")
            return None
        
        click_counts = self.df['clicked'].value_counts()
        click_pct = self.df['clicked'].value_counts(normalize=True) * 100
        
        print("\n" + "=" * 60)
        print("CLASS IMBALANCE ANALYSIS")
        print("=" * 60)
        print(f"Total impressions: {len(self.df):,}")
        print(f"Clicked: {click_counts.get(1, 0):,} ({click_pct.get(1, 0):.2f}%)")
        print(f"Not clicked: {click_counts.get(0, 0):,} ({click_pct.get(0, 0):.2f}%)")
        print(f"Overall CTR: {self.df['clicked'].mean():.4f}")
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        click_counts.plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'])
        axes[0].set_title('Click Distribution (Counts)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Clicked')
        axes[0].set_ylabel('Count')
        axes[0].set_xticklabels(['Not Clicked', 'Clicked'], rotation=0)
        for i, v in enumerate(click_counts.values):
            axes[0].text(i, v, f'{v:,}', ha='center', va='bottom')
        
        # Percentage plot
        click_pct.plot(kind='bar', ax=axes[1], color=['#3498db', '#e74c3c'])
        axes[1].set_title('Click Distribution (Percentages)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Clicked')
        axes[1].set_ylabel('Percentage (%)')
        axes[1].set_xticklabels(['Not Clicked', 'Clicked'], rotation=0)
        for i, v in enumerate(click_pct.values):
            axes[1].text(i, v, f'{v:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_imbalance.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot to {self.output_dir / 'class_imbalance.png'}")
        plt.close()
        
        return {
            'clicked': click_counts.get(1, 0),
            'not_clicked': click_counts.get(0, 0),
            'ctr': self.df['clicked'].mean()
        }
    
    def feature_distributions(self):
        """Analyze distributions of categorical and numerical features"""
        print("\n" + "=" * 60)
        print("FEATURE DISTRIBUTIONS")
        print("=" * 60)
        
        # Categorical features
        categorical_cols = ['device', 'placement']
        if 'device' in self.df.columns:
            for col in categorical_cols:
                if col in self.df.columns:
                    print(f"\n{col.upper()}:")
                    value_counts = self.df[col].value_counts()
                    for val, count in value_counts.items():
                        pct = (count / len(self.df)) * 100
                        print(f"  {val}: {count:,} ({pct:.2f}%)")
        
        # Numerical features
        numerical_cols = ['user_id', 'ad_id', 'hour', 'day_of_week']
        numerical_cols = [col for col in numerical_cols if col in self.df.columns]
        
        if numerical_cols:
            print(f"\nNumerical features statistics:")
            print(self.df[numerical_cols].describe())
    
    def temporal_patterns(self):
        """Analyze temporal patterns in clicks"""
        if 'timestamp' not in self.df.columns:
            print("Warning: 'timestamp' column not found")
            return
        
        print("\n" + "=" * 60)
        print("TEMPORAL PATTERNS ANALYSIS")
        print("=" * 60)
        
        # Hourly patterns
        if 'hour' in self.df.columns:
            hourly_ctr = self.df.groupby('hour')['clicked'].agg(['mean', 'count'])
            hourly_ctr.columns = ['ctr', 'impressions']
            
            print("\nHourly CTR:")
            print(hourly_ctr)
            
            # Plot
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # Hourly CTR
            axes[0].plot(hourly_ctr.index, hourly_ctr['ctr'], marker='o', linewidth=2, markersize=8)
            axes[0].set_title('CTR by Hour of Day', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Hour')
            axes[0].set_ylabel('CTR')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xticks(range(24))
            
            # Hourly impressions
            axes[1].bar(hourly_ctr.index, hourly_ctr['impressions'], alpha=0.7)
            axes[1].set_title('Impressions by Hour of Day', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Hour')
            axes[1].set_ylabel('Number of Impressions')
            axes[1].grid(True, alpha=0.3, axis='y')
            axes[1].set_xticks(range(24))
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'temporal_hourly.png', dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved plot to {self.output_dir / 'temporal_hourly.png'}")
            plt.close()
        
        # Day of week patterns
        if 'day_of_week' in self.df.columns:
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_ctr = self.df.groupby('day_of_week')['clicked'].agg(['mean', 'count'])
            daily_ctr.index = [day_names[i] for i in daily_ctr.index]
            daily_ctr.columns = ['ctr', 'impressions']
            
            print("\nDaily CTR:")
            print(daily_ctr)
            
            # Plot
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Daily CTR
            axes[0].plot(range(len(daily_ctr)), daily_ctr['ctr'], marker='o', linewidth=2, markersize=8)
            axes[0].set_title('CTR by Day of Week', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Day of Week')
            axes[0].set_ylabel('CTR')
            axes[0].set_xticks(range(len(daily_ctr)))
            axes[0].set_xticklabels(daily_ctr.index, rotation=45, ha='right')
            axes[0].grid(True, alpha=0.3)
            
            # Daily impressions
            axes[1].bar(range(len(daily_ctr)), daily_ctr['impressions'], alpha=0.7)
            axes[1].set_title('Impressions by Day of Week', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Day of Week')
            axes[1].set_ylabel('Number of Impressions')
            axes[1].set_xticks(range(len(daily_ctr)))
            axes[1].set_xticklabels(daily_ctr.index, rotation=45, ha='right')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'temporal_daily.png', dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved plot to {self.output_dir / 'temporal_daily.png'}")
            plt.close()
    
    def device_placement_analysis(self):
        """Analyze CTR by device and placement"""
        if 'device' not in self.df.columns or 'placement' not in self.df.columns:
            print("Warning: 'device' or 'placement' columns not found")
            return
        
        print("\n" + "=" * 60)
        print("DEVICE & PLACEMENT ANALYSIS")
        print("=" * 60)
        
        # Device CTR
        device_ctr = self.df.groupby('device')['clicked'].agg(['mean', 'count'])
        device_ctr.columns = ['ctr', 'impressions']
        print("\nCTR by Device:")
        print(device_ctr)
        
        # Placement CTR
        placement_ctr = self.df.groupby('placement')['clicked'].agg(['mean', 'count'])
        placement_ctr.columns = ['ctr', 'impressions']
        print("\nCTR by Placement:")
        print(placement_ctr)
        
        # Device x Placement interaction
        device_placement_ctr = self.df.groupby(['device', 'placement'])['clicked'].mean().unstack()
        print("\nCTR by Device x Placement:")
        print(device_placement_ctr)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Device CTR
        device_ctr['ctr'].plot(kind='bar', ax=axes[0], color='#3498db')
        axes[0].set_title('CTR by Device', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Device')
        axes[0].set_ylabel('CTR')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Placement CTR
        placement_ctr['ctr'].plot(kind='bar', ax=axes[1], color='#e74c3c')
        axes[1].set_title('CTR by Placement', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Placement')
        axes[1].set_ylabel('CTR')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'device_placement_ctr.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot to {self.output_dir / 'device_placement_ctr.png'}")
        plt.close()
        
        # Heatmap
        if device_placement_ctr.shape[0] > 0:
            plt.figure(figsize=(10, 6))
            sns.heatmap(device_placement_ctr, annot=True, fmt='.4f', cmap='YlOrRd', cbar_kws={'label': 'CTR'})
            plt.title('CTR Heatmap: Device x Placement', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'device_placement_heatmap.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {self.output_dir / 'device_placement_heatmap.png'}")
            plt.close()
    
    def user_ad_analysis(self):
        """Analyze user and ad statistics"""
        print("\n" + "=" * 60)
        print("USER & AD ANALYSIS")
        print("=" * 60)
        
        if 'user_id' in self.df.columns:
            n_unique_users = self.df['user_id'].nunique()
            avg_impressions_per_user = len(self.df) / n_unique_users
            print(f"\nUnique users: {n_unique_users:,}")
            print(f"Average impressions per user: {avg_impressions_per_user:.2f}")
            
            # Top users by impressions
            top_users = self.df['user_id'].value_counts().head(10)
            print(f"\nTop 10 users by impressions:")
            print(top_users)
        
        if 'ad_id' in self.df.columns:
            n_unique_ads = self.df['ad_id'].nunique()
            avg_impressions_per_ad = len(self.df) / n_unique_ads
            print(f"\nUnique ads: {n_unique_ads:,}")
            print(f"Average impressions per ad: {avg_impressions_per_ad:.2f}")
            
            # Top ads by impressions
            top_ads = self.df['ad_id'].value_counts().head(10)
            print(f"\nTop 10 ads by impressions:")
            print(top_ads)
    
    def run_full_eda(self):
        """Run complete EDA analysis"""
        print("\n" + "=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}\n")
        
        # Run all analyses
        self.basic_info()
        self.missing_values()
        self.class_imbalance()
        self.feature_distributions()
        self.temporal_patterns()
        self.device_placement_analysis()
        self.user_ad_analysis()
        
        print("\n" + "=" * 60)
        print("EDA COMPLETE!")
        print("=" * 60)
        print(f"All plots saved to: {self.output_dir}")


def explore_dataset(filepath: str, output_dir: str = "data/eda"):
    """
    Convenience function to load and explore dataset
    
    Args:
        filepath: Path to CSV file
        output_dir: Directory to save plots
    """
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Convert timestamp if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    explorer = CTRExplorer(df, output_dir)
    explorer.run_full_eda()
    
    return explorer


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "data/raw/impressions.csv"
    
    explore_dataset(filepath)

