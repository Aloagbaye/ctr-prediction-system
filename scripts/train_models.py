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
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from sklearn.model_selection import train_test_split

# MLflow integration
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️  MLflow not installed. Training will continue without MLflow logging.")
    print("   Install with: pip install mlflow")


def setup_mlflow(experiment_name: str = "ctr_prediction_training", enable_mlflow: bool = True):
    """Setup MLflow tracking"""
    if not MLFLOW_AVAILABLE or not enable_mlflow:
        return False
    
    try:
        # Set tracking URI (Windows-compatible)
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        mlruns_dir = project_root / "mlruns"
        mlruns_dir.mkdir(exist_ok=True)
        
        # Use absolute path for Windows compatibility
        abs_path = mlruns_dir.absolute()
        if os.name == 'nt':  # Windows
            tracking_uri = f"file:///{str(abs_path).replace(chr(92), '/')}"
        else:
            tracking_uri = f"file://{abs_path.as_posix()}"
        
        mlflow.set_tracking_uri(tracking_uri)
        
        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✓ Created MLflow experiment: {experiment_name} (ID: {experiment_id})")
        else:
            print(f"✓ Using MLflow experiment: {experiment_name} (ID: {experiment.experiment_id})")
        
        mlflow.set_experiment(experiment_name)
        
        # Enable autologging for XGBoost and LightGBM
        try:
            mlflow.xgboost.autolog()
            print("✓ MLflow XGBoost autologging enabled")
        except:
            pass
        
        try:
            mlflow.lightgbm.autolog()
            print("✓ MLflow LightGBM autologging enabled")
        except:
            pass
        
        return True
    except Exception as e:
        print(f"⚠️  MLflow setup failed: {e}")
        print("   Training will continue without MLflow logging.")
        return False


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
    parser.add_argument('--mlflow-experiment', type=str, default='ctr_prediction_training',
                       help='MLflow experiment name')
    parser.add_argument('--no-mlflow', action='store_true',
                       help='Disable MLflow logging')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Models: {', '.join(args.models)}")
    print("=" * 60)
    
    # Setup MLflow
    mlflow_enabled = setup_mlflow(args.mlflow_experiment, enable_mlflow=not args.no_mlflow)
    if mlflow_enabled:
        print(f"✓ MLflow tracking enabled: {mlflow.get_tracking_uri()}")
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
    
    # Prepare data once for evaluation
    feature_cols = [col for col in df.columns if col != args.target_col]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols]
    y = df[args.target_col]
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )
    
    # Train models with MLflow logging
    print("\nTraining models...")
    trained_models = {}
    mlflow_run_ids = {}  # Store run IDs for later test metric logging
    
    # Get train/val split for training
    X_train, X_val, y_train, y_val = trainer.prepare_data(df, feature_cols)
    
    # Train each model with MLflow
    
    for model_name in args.models:
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if mlflow_enabled:
            with mlflow.start_run(run_name=run_name, nested=False):
                run_id = mlflow.active_run().info.run_id
                mlflow_run_ids[model_name] = run_id
                
                # Set tags
                mlflow.set_tag("model_type", model_name)
                mlflow.set_tag("task", "ctr_prediction")
                mlflow.set_tag("dataset", Path(args.input).stem)
                mlflow.set_tag("training_date", datetime.now().isoformat())
                
                # Log data parameters
                mlflow.log_params({
                    "test_size": args.test_size,
                    "val_size": args.val_size,
                    "random_state": args.random_state,
                    "num_features": len(numeric_cols),
                    "train_samples": len(X_train),
                    "val_samples": len(X_val),
                    "test_samples": len(X_test)
                })
                
                # Train model
                if model_name == 'logistic':
                    model = trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
                    # Log model manually (no autologging for sklearn LogisticRegression)
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path=f"{model_name}_model",
                        registered_model_name=f"ctr_{model_name}"
                    )
                elif model_name == 'xgboost':
                    # XGBoost autologging will handle model logging
                    model = trainer.train_xgboost(X_train, y_train, X_val, y_val)
                elif model_name == 'lightgbm':
                    # LightGBM autologging will handle model logging
                    model = trainer.train_lightgbm(X_train, y_train, X_val, y_val)
                
                trained_models[model_name] = model
                
                # Evaluate on validation set and log metrics
                if model_name == 'logistic' and 'logistic' in trainer.scalers:
                    X_val_scaled = trainer.scalers['logistic'].transform(X_val)
                    val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
                else:
                    val_pred_proba = model.predict_proba(X_val)[:, 1]
                
                val_pred_binary = (val_pred_proba >= 0.5).astype(int)
                evaluator = ModelEvaluator()
                val_metrics = evaluator.evaluate(y_val, val_pred_proba, val_pred_binary)
                
                # Log validation metrics
                mlflow.log_metrics({
                    "val_roc_auc": val_metrics['roc_auc'],
                    "val_log_loss": val_metrics['log_loss'],
                    "val_pr_auc": val_metrics['pr_auc']
                })
                
                if 'accuracy' in val_metrics:
                    mlflow.log_metric("val_accuracy", val_metrics['accuracy'])
                
                print(f"✓ {model_name} trained and logged to MLflow (Run ID: {run_id[:8]}...)")
        else:
            # Train without MLflow
            if model_name == 'logistic':
                trained_models[model_name] = trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
            elif model_name == 'xgboost':
                trained_models[model_name] = trainer.train_xgboost(X_train, y_train, X_val, y_val)
            elif model_name == 'lightgbm':
                trained_models[model_name] = trainer.train_lightgbm(X_train, y_train, X_val, y_val)
    
    # Save models
    print(f"\nSaving models to {args.output_dir}...")
    trainer.save_models(args.output_dir)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)
    
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
        
        # Log test metrics to MLflow (resume the run)
        if mlflow_enabled and model_name in mlflow_run_ids:
            try:
                run_id = mlflow_run_ids[model_name]
                # Resume the run to log test metrics
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_metrics({
                        "test_roc_auc": metrics['roc_auc'],
                        "test_log_loss": metrics['log_loss'],
                        "test_pr_auc": metrics['pr_auc']
                    })
                    if 'accuracy' in metrics:
                        mlflow.log_metric("test_accuracy", metrics['accuracy'])
                print(f"  ✓ Test metrics logged to MLflow")
            except Exception as e:
                print(f"  ⚠️  Could not log test metrics to MLflow: {e}")
    
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
    if mlflow_enabled:
        print(f"\nView results in MLflow UI:")
        print(f"  1. Start MLflow UI: mlflow ui")
        print(f"  2. Open http://localhost:5000")
        print(f"  3. Select experiment: {args.mlflow_experiment}")


if __name__ == "__main__":
    main()

