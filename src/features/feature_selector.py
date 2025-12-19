"""
Feature Selection

Removes redundant and low-importance features using:
- Correlation analysis
- Feature importance from models
- Variance threshold
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """Select important features and remove redundant ones"""
    
    def __init__(self, target_col: str = 'clicked', random_state: int = 42):
        """
        Initialize feature selector
        
        Args:
            target_col: Name of target column
            random_state: Random seed
        """
        self.target_col = target_col
        self.random_state = random_state
        self.selected_features = []
        self.variance_threshold = VarianceThreshold(threshold=0.0)
    
    def remove_low_variance_features(
        self, 
        df: pd.DataFrame, 
        threshold: float = 0.01
    ) -> pd.DataFrame:
        """
        Remove features with low variance
        
        Args:
            df: DataFrame
            threshold: Variance threshold (features below this are removed)
            
        Returns:
            DataFrame with low-variance features removed
        """
        df = df.copy()
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != self.target_col]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[self.target_col] if self.target_col in df.columns else None
        
        # Apply variance threshold
        self.variance_threshold = VarianceThreshold(threshold=threshold)
        X_selected = self.variance_threshold.fit_transform(X)
        
        # Get selected feature names
        selected_mask = self.variance_threshold.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Keep non-numeric columns and selected numeric columns
        non_numeric_cols = [col for col in df.columns if col not in X.columns]
        df_selected = df[non_numeric_cols + selected_features].copy()
        
        if y is not None:
            df_selected[self.target_col] = y
        
        print(f"Removed {len(X.columns) - len(selected_features)} low-variance features")
        
        return df_selected
    
    def remove_highly_correlated_features(
        self, 
        df: pd.DataFrame, 
        threshold: float = 0.95
    ) -> pd.DataFrame:
        """
        Remove highly correlated features
        
        Args:
            df: DataFrame
            threshold: Correlation threshold (features above this are removed)
            
        Returns:
            DataFrame with highly correlated features removed
        """
        df = df.copy()
        
        # Get numeric features only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)
        
        if len(numeric_cols) < 2:
            return df
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > threshold)]
        
        # Drop features
        df = df.drop(columns=to_drop)
        
        print(f"Removed {len(to_drop)} highly correlated features (threshold={threshold})")
        
        return df
    
    def select_by_importance(
        self,
        df: pd.DataFrame,
        n_features: int = 50,
        max_depth: int = 10
    ) -> pd.DataFrame:
        """
        Select features based on importance from a Random Forest model
        
        Args:
            df: DataFrame
            n_features: Number of top features to keep
            max_depth: Max depth for Random Forest (to speed up)
            
        Returns:
            DataFrame with selected features
        """
        df = df.copy()
        
        if self.target_col not in df.columns:
            print("Target column not found, skipping importance-based selection")
            return df
        
        # Get numeric features
        feature_cols = [col for col in df.columns if col != self.target_col]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            return df
        
        X = df[numeric_cols]
        y = df[self.target_col]
        
        # Train Random Forest to get feature importance
        print(f"Training Random Forest to get feature importance...")
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': numeric_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top features
        top_features = feature_importance.head(n_features)['feature'].tolist()
        
        # Keep non-numeric columns and top numeric features
        non_numeric_cols = [col for col in df.columns 
                          if col not in numeric_cols and col != self.target_col]
        selected_cols = non_numeric_cols + top_features + [self.target_col]
        
        df_selected = df[selected_cols].copy()
        
        print(f"Selected top {len(top_features)} features by importance")
        
        # Store selected features
        self.selected_features = selected_cols
        
        return df_selected
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        remove_low_variance: bool = True,
        remove_correlated: bool = True,
        select_by_importance: bool = False,
        n_features: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Apply all feature selection steps
        
        Args:
            df: DataFrame
            remove_low_variance: Whether to remove low-variance features
            remove_correlated: Whether to remove highly correlated features
            select_by_importance: Whether to select by importance
            n_features: Number of features to keep (if select_by_importance=True)
            
        Returns:
            DataFrame with selected features
        """
        print("Applying feature selection...")
        
        if remove_low_variance:
            df = self.remove_low_variance_features(df)
        
        if remove_correlated:
            df = self.remove_highly_correlated_features(df)
        
        if select_by_importance and n_features:
            df = self.select_by_importance(df, n_features=n_features)
        
        print(f"✓ Feature selection complete. Final columns: {len(df.columns)}")
        
        return df

