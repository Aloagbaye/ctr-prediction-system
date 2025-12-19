#!/usr/bin/env python3
"""
Script to train baseline models

Usage:
    python scripts/train_models.py [--input PATH] [--output-dir DIR] [--models MODEL_LIST]
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description='Train baseline models for CTR prediction')
    parser.add_argument('--input', type=str, default='data/processed/features.csv',
                       help='Input CSV/Parquet file with features')
    parser.add_argument('--output-dir', type=str, default='models/',
                       help='Output directory for trained models')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['logistic', 'xgboost', 'lightgbm'],
                       choices=['logistic', 'xgboost', 'lightgbm'],
                       help='Models to train')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data for test set')
    parser.add_argument('--val-size', type=float, default=0.2,
                       help='Proportion of train data for validation set')
    parser.add_argument('--target-col', type=str, default='clicked',
                       help='Name of target column')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Models: {', '.join(args.models)}")
    print("=" * 60)
    print()
    
    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        print("\nPlease run feature engineering first:")
        print("  python scripts/create_features.py")
        sys.exit(1)
    
    print(f"Loading data from {input_path}...")
    if input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Check target column
    if args.target_col not in df.columns:
        print(f"❌ Target column '{args.target_col}' not found!")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Initialize trainer
    trainer = ModelTrainer(
        target_col=args.target_col,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )
    
    # Train models
    print("\nTraining models...")
    trained_models = trainer.train_all_models(df, models=args.models)
    
    # Save models
    print(f"\nSaving models to {args.output_dir}...")
    trainer.save_models(args.output_dir)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)
    
    # Prepare test set
    feature_cols = [col for col in df.columns if col != args.target_col]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols]
    y = df[args.target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )
    
    evaluator = ModelEvaluator()
    results = {}
    
    for model_name, model in trained_models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Get predictions
        if model_name == 'logistic' and 'logistic' in trainer.scalers:
            # Scale features for logistic regression
            X_test_scaled = trainer.scalers['logistic'].transform(X_test)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        y_pred_binary = (y_pred_proba >= 0.5).astype(int)
        
        # Evaluate
        metrics = evaluator.evaluate(y_test, y_pred_proba, y_pred_binary)
        results[model_name] = metrics
        
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  Log Loss: {metrics['log_loss']:.4f}")
        print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        if 'accuracy' in metrics:
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    # Compare models
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    comparison_df = pd.DataFrame({
        model: {
            'ROC-AUC': results[model]['roc_auc'],
            'Log Loss': results[model]['log_loss'],
            'PR-AUC': results[model]['pr_auc']
        }
        for model in results.keys()
    }).T
    print(comparison_df.round(4))
    
    # Save evaluation results
    eval_output = Path(args.output_dir) / "evaluation_results.csv"
    comparison_df.to_csv(eval_output)
    print(f"\n✓ Evaluation results saved to {eval_output}")
    
    print("\n✓ Model training complete!")


if __name__ == "__main__":
    main()

