# MLflow Tutorial: Complete Guide to Experiment Tracking and Model Management

## Table of Contents
1. [Introduction to MLflow](#introduction)
2. [Why Use MLflow?](#why-mlflow)
3. [Core Concepts](#core-concepts)
4. [Installation and Setup](#installation)
5. [Basic Usage](#basic-usage)
6. [Advanced Features](#advanced-features)
7. [Integration with Our System](#integration)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Real-World Examples](#real-world-examples)

---

## 1. Introduction to MLflow {#introduction}

### What is MLflow?

**MLflow** is an open-source platform for managing the machine learning lifecycle, including:
- **Experiment Tracking**: Log and query experiments (parameters, metrics, artifacts)
- **Model Registry**: Centralized model store with versioning and stage management
- **Model Packaging**: Package models for deployment
- **Model Deployment**: Deploy models to various platforms

### Key Benefits

- **Reproducibility**: Track all parameters and code versions
- **Collaboration**: Share experiments with team members
- **Organization**: Organize experiments by project or team
- **Comparison**: Easily compare model performance
- **Deployment**: Streamline model deployment process

### MLflow Components

1. **MLflow Tracking**: Logging API and UI for recording and querying experiments
2. **MLflow Projects**: Packaging format for reproducible runs
3. **MLflow Models**: Model packaging format and tools
4. **Model Registry**: Centralized model store and lifecycle management

---

## 2. Why Use MLflow? {#why-mlflow}

### Problem: Without MLflow

**Scenario**: You're training multiple models with different hyperparameters

**Challenges**:
- ❌ Hard to remember which parameters you used
- ❌ Difficult to compare model performance
- ❌ No easy way to reproduce results
- ❌ Models scattered across different files
- ❌ No versioning or history
- ❌ Team members can't see your experiments

### Solution: With MLflow

**Benefits**:
- ✅ Automatic logging of parameters, metrics, and models
- ✅ Easy comparison of experiments in UI
- ✅ Reproducible experiments with code snapshots
- ✅ Centralized model storage
- ✅ Version control for models
- ✅ Team collaboration and sharing

### Real-World Example

**Before MLflow**:
```python
# Training script
model = train_xgboost(learning_rate=0.1, max_depth=6)
accuracy = evaluate(model)
print(f"Accuracy: {accuracy}")  # Where do I save this?
# Model saved to: models/xgboost_v1.pkl  # Which version is this?
```

**After MLflow**:
```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 6)
    
    model = train_xgboost(learning_rate=0.1, max_depth=6)
    accuracy = evaluate(model)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    
    # Everything tracked automatically!
```

---

## 3. Core Concepts {#core-concepts}

### 3.1 Tracking URI

**Location where MLflow stores experiment data**

**Types**:
- **Local File**: `file:///path/to/mlruns` (default)
- **SQLite**: `sqlite:///mlflow.db`
- **Database**: `postgresql://user:pass@host/db`
- **HTTP Server**: `http://mlflow-server:5000`
- **Databricks**: `databricks://profile`

**Example**:
```python
import mlflow

# Local file store (default)
mlflow.set_tracking_uri("file:///path/to/mlruns")

# SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Remote server
mlflow.set_tracking_uri("http://mlflow-server:5000")
```

### 3.2 Experiments

**Container for organizing runs**

**Structure**:
```
Experiment: "ctr_prediction_training"
├── Run 1: logistic_20240101_120000
├── Run 2: xgboost_20240101_120500
└── Run 3: lightgbm_20240101_121000
```

**Creating/Using Experiments**:
```python
# Create or get experiment
experiment = mlflow.get_experiment_by_name("my_experiment")
if experiment is None:
    experiment_id = mlflow.create_experiment("my_experiment")
else:
    experiment_id = experiment.experiment_id

# Set active experiment
mlflow.set_experiment("my_experiment")
```

### 3.3 Runs

**Single execution of your code**

**Run contains**:
- **Parameters**: Inputs to your code (hyperparameters, config)
- **Metrics**: Outputs you want to track (accuracy, loss, latency)
- **Artifacts**: Files (models, plots, data)
- **Tags**: Metadata (team, project, status)
- **Code Version**: Git commit hash (if available)

**Example**:
```python
with mlflow.start_run(run_name="xgboost_experiment_1"):
    # Log parameters
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 6)
    
    # Train model
    model = train_model()
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("roc_auc", 0.92)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### 3.4 Parameters vs Metrics

**Parameters**: Inputs that don't change during training
- Hyperparameters (learning_rate, max_depth)
- Configuration (batch_size, epochs)
- Data paths

**Metrics**: Outputs that may change during training
- Performance metrics (accuracy, loss)
- Training time
- Resource usage

**Key Difference**:
- Parameters: Logged once per run
- Metrics: Can be logged multiple times (e.g., per epoch)

**Example**:
```python
# Parameters (logged once)
mlflow.log_param("learning_rate", 0.1)
mlflow.log_param("max_depth", 6)

# Metrics (can log multiple times)
for epoch in range(10):
    loss = train_epoch()
    mlflow.log_metric("loss", loss, step=epoch)  # step for time series
```

### 3.5 Artifacts

**Files associated with a run**

**Common Artifacts**:
- Trained models
- Plots and visualizations
- Data files
- Configuration files
- Logs

**Example**:
```python
# Log model
mlflow.sklearn.log_model(model, "model")

# Log plot
import matplotlib.pyplot as plt
plt.plot(history)
plt.savefig("training_curve.png")
mlflow.log_artifact("training_curve.png")

# Log directory
mlflow.log_artifacts("outputs/", "outputs")
```

### 3.6 Tags

**Key-value pairs for organizing runs**

**Common Tags**:
- `team`: Which team owns the experiment
- `project`: Project name
- `status`: "completed", "failed", "in_progress"
- `model_type`: "xgboost", "lightgbm", "neural_net"

**Example**:
```python
mlflow.set_tag("team", "ml-engineering")
mlflow.set_tag("project", "ctr-prediction")
mlflow.set_tag("model_type", "xgboost")
mlflow.set_tag("status", "completed")
```

---

## 4. Installation and Setup {#installation}

### 4.1 Installation

**Basic Installation**:
```bash
pip install mlflow
```

**With Extra Dependencies**:
```bash
# For specific frameworks
pip install mlflow[extras]  # Includes many ML libraries

# For specific framework
pip install mlflow[sklearn]  # scikit-learn
pip install mlflow[xgboost]  # XGBoost
pip install mlflow[lightgbm]  # LightGBM
pip install mlflow[tensorflow]  # TensorFlow
pip install mlflow[pytorch]  # PyTorch
```

**Full Installation** (for our project):
```bash
pip install mlflow scikit-learn xgboost lightgbm
```

### 4.2 Basic Setup

**1. Create MLflow Directory**:
```bash
mkdir mlruns
```

**2. Set Tracking URI** (optional, defaults to `./mlruns`):
```python
import mlflow
mlflow.set_tracking_uri("file:///path/to/mlruns")
```

**3. Create Experiment**:
```python
experiment = mlflow.get_experiment_by_name("my_experiment")
if experiment is None:
    mlflow.create_experiment("my_experiment")
mlflow.set_experiment("my_experiment")
```

### 4.3 Windows-Specific Setup

**Issue**: Windows paths need special handling

**Solution**:
```python
from pathlib import Path
import os

# Get absolute path
project_root = Path(__file__).parent.parent
mlruns_dir = project_root / "mlruns"
mlruns_dir.mkdir(exist_ok=True)

# Convert to Windows-compatible URI
abs_path = mlruns_dir.absolute()
if os.name == 'nt':  # Windows
    tracking_uri = f"file:///{str(abs_path).replace(chr(92), '/')}"
else:
    tracking_uri = f"file://{abs_path.as_posix()}"

mlflow.set_tracking_uri(tracking_uri)
```

### 4.4 Starting MLflow UI

**Command**:
```bash
mlflow ui
```

**Options**:
```bash
# Specify port
mlflow ui --port 5000

# Specify tracking URI
mlflow ui --backend-store-uri file:///path/to/mlruns

# Specify host (for remote access)
mlflow ui --host 0.0.0.0
```

**Access**:
- Local: http://localhost:5000
- Remote: http://your-server:5000

---

## 5. Basic Usage {#basic-usage}

### 5.1 Simple Example

**Minimal Tracking**:
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set experiment
mlflow.set_experiment("my_first_experiment")

# Start run
with mlflow.start_run(run_name="rf_baseline"):
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Accuracy: {accuracy}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

### 5.2 Autologging

**Automatic logging for popular frameworks**

**Supported Frameworks**:
- scikit-learn
- XGBoost
- LightGBM
- TensorFlow/Keras
- PyTorch
- Spark MLlib

**Example (scikit-learn)**:
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Enable autologging
mlflow.sklearn.autolog()

# Train model (automatically logged!)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Metrics, parameters, and model automatically logged!
```

**Example (XGBoost)**:
```python
import mlflow
import mlflow.xgboost
import xgboost as xgb

# Enable autologging
mlflow.xgboost.autolog()

# Train model
model = xgb.XGBClassifier()
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# Everything logged automatically!
```

**Example (LightGBM)**:
```python
import mlflow
import mlflow.lightgbm
import lightgbm as lgb

# Enable autologging
mlflow.lightgbm.autolog()

# Train model
model = lgb.LGBMClassifier()
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# Automatically logged!
```

### 5.3 Manual Logging

**When to use manual logging**:
- Custom metrics not captured by autologging
- Additional parameters
- Custom artifacts
- Framework not supported by autologging

**Example**:
```python
with mlflow.start_run():
    # Parameters
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("n_estimators", 100)
    
    # Multiple metrics
    mlflow.log_metrics({
        "train_accuracy": 0.95,
        "val_accuracy": 0.92,
        "test_accuracy": 0.91
    })
    
    # Time series metrics
    for epoch in range(10):
        loss = train_epoch()
        mlflow.log_metric("loss", loss, step=epoch)
    
    # Tags
    mlflow.set_tags({
        "team": "ml-engineering",
        "project": "ctr-prediction",
        "model_type": "xgboost"
    })
    
    # Artifacts
    mlflow.log_artifact("config.yaml")
    mlflow.log_artifacts("plots/", "plots")
    
    # Model
    mlflow.sklearn.log_model(model, "model")
```

### 5.4 Logging Models

**Different model flavors**:

**scikit-learn**:
```python
import mlflow.sklearn

mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    registered_model_name="ctr_predictor"  # Optional: register in Model Registry
)
```

**XGBoost**:
```python
import mlflow.xgboost

mlflow.xgboost.log_model(
    xgb_model=model,
    artifact_path="model"
)
```

**LightGBM**:
```python
import mlflow.lightgbm

mlflow.lightgbm.log_model(
    lgb_model=model,
    artifact_path="model"
)
```

**PyFunc (Generic)**:
```python
import mlflow.pyfunc

class MyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        # Custom prediction logic
        return predictions

mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=MyModel()
)
```

### 5.5 Loading Models

**Load for inference**:
```python
# Load model from run
model = mlflow.sklearn.load_model("runs:/<run_id>/model")

# Load from Model Registry
model = mlflow.sklearn.load_model("models:/ctr_predictor/Production")

# Load as PyFunc (works for any flavor)
model = mlflow.pyfunc.load_model("runs:/<run_id>/model")
predictions = model.predict(X_test)
```

### 5.6 Querying Runs

**Search runs**:
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get experiment
experiment = client.get_experiment_by_name("my_experiment")

# Search runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.accuracy > 0.9",
    max_results=10,
    order_by=["metrics.accuracy DESC"]
)

# Access run data
for run in runs:
    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {run.data.metrics['accuracy']}")
    print(f"Parameters: {run.data.params}")
```

---

## 6. Advanced Features {#advanced-features}

### 6.1 Nested Runs

**Organize related runs**:
```python
with mlflow.start_run(run_name="hyperparameter_tuning"):
    mlflow.log_param("algorithm", "xgboost")
    
    # Nested run for each hyperparameter combination
    for lr in [0.01, 0.1, 0.3]:
        with mlflow.start_run(nested=True):
            mlflow.log_param("learning_rate", lr)
            model = train_model(learning_rate=lr)
            accuracy = evaluate(model)
            mlflow.log_metric("accuracy", accuracy)
```

### 6.2 Model Registry

**Centralized model management**:

**Register Model**:
```python
# After training
mlflow.sklearn.log_model(
    model,
    "model",
    registered_model_name="ctr_predictor"
)
```

**Transition Stages**:
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Transition to Staging
client.transition_model_version_stage(
    name="ctr_predictor",
    version=1,
    stage="Staging"
)

# Transition to Production
client.transition_model_version_stage(
    name="ctr_predictor",
    version=1,
    stage="Production"
)
```

**Load from Registry**:
```python
# Load from specific stage
model = mlflow.sklearn.load_model(
    "models:/ctr_predictor/Production"
)

# Load specific version
model = mlflow.sklearn.load_model(
    "models:/ctr_predictor/1"
)
```

### 6.3 Projects

**Reproducible ML code packaging**:

**Create MLproject file**:
```yaml
# MLproject
name: ctr_prediction_training

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      learning_rate: {type: float, default: 0.1}
      max_depth: {type: int, default: 6}
    command: "python train.py --learning-rate {learning_rate} --max-depth {max_depth}"
```

**Run Project**:
```bash
mlflow run . -P learning_rate=0.1 -P max_depth=6
```

### 6.4 Custom Metrics with Timestamps

**Time series metrics**:
```python
import time

for epoch in range(10):
    loss = train_epoch()
    timestamp = int(time.time() * 1000)  # milliseconds
    mlflow.log_metric("loss", loss, timestamp=timestamp, step=epoch)
```

### 6.5 Logging Images

**Visualizations**:
```python
import matplotlib.pyplot as plt
import mlflow

# Create plot
fig, ax = plt.subplots()
ax.plot(history['loss'], label='Training Loss')
ax.plot(history['val_loss'], label='Validation Loss')
ax.legend()
plt.savefig("loss_curve.png")

# Log image
mlflow.log_artifact("loss_curve.png", "plots")
```

### 6.6 Environment Logging

**Capture environment**:
```python
# Autologging captures:
# - Python version
# - Package versions
# - Git commit (if available)

# Manual logging
mlflow.log_param("python_version", sys.version)
mlflow.log_param("platform", platform.platform())
```

---

## 7. Integration with Our System {#integration}

### 7.1 Model Training Integration

**Our Implementation** (`scripts/train_models.py`):

```python
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

# Setup MLflow
def setup_mlflow(experiment_name="ctr_prediction_training"):
    # Windows-compatible tracking URI
    project_root = Path(__file__).parent.parent
    mlruns_dir = project_root / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)
    
    abs_path = mlruns_dir.absolute()
    if os.name == 'nt':  # Windows
        tracking_uri = f"file:///{str(abs_path).replace(chr(92), '/')}"
    else:
        tracking_uri = f"file://{abs_path.as_posix()}"
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # Enable autologging
    mlflow.xgboost.autolog()
    mlflow.lightgbm.autolog()

# Training with MLflow
with mlflow.start_run(run_name=f"{model_name}_{timestamp}"):
    mlflow.set_tag("model_type", model_name)
    mlflow.log_params({
        "test_size": 0.2,
        "val_size": 0.2,
        "random_state": 42
    })
    
    # Train model (autologging captures everything)
    model = trainer.train_xgboost(X_train, y_train, X_val, y_val)
    
    # Log additional metrics
    mlflow.log_metrics({
        "val_roc_auc": val_metrics['roc_auc'],
        "test_roc_auc": test_metrics['roc_auc']
    })
```

### 7.2 A/B Testing Integration

**Our Implementation** (`src/api/ab_test_manager.py`):

```python
import mlflow

class ABTestManager:
    def __init__(self, mlflow_tracking_uri=None):
        # Setup MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri or "file:///mlruns")
        mlflow.set_experiment("ctr_prediction_ab_tests")
    
    def create_test(self, test_id, model_a, model_b, traffic_split):
        # Create MLflow run for A/B test
        with mlflow.start_run(run_name=f"ab_test_{test_id}"):
            mlflow.log_params({
                "test_id": test_id,
                "model_a": model_a,
                "model_b": model_b,
                "traffic_split": traffic_split
            })
            mlflow.set_tags({
                "test_type": "ab_test",
                "model_a": model_a,
                "model_b": model_b
            })
    
    def record_prediction(self, test_id, model_name, latency, error):
        # Log metrics periodically
        if self.enable_mlflow:
            with mlflow.start_run(run_id=self.mlflow_run_ids[test_id]):
                mlflow.log_metrics({
                    f"{model_name}_avg_latency_ms": latency,
                    f"{model_name}_error_rate": error_rate
                })
```

### 7.3 Monitoring Integration

**Our Implementation** (`src/api/monitoring.py`):

```python
import mlflow

class MonitoringManager:
    def __init__(self, mlflow_tracking_uri=None):
        mlflow.set_tracking_uri(mlflow_tracking_uri or "file:///mlruns")
        mlflow.set_experiment("ctr_prediction_production")
    
    def log_prediction(self, model_name, latency, error):
        # Log to MLflow
        with mlflow.start_run(run_name="production_monitoring"):
            mlflow.log_metrics({
                f"{model_name}_latency_ms": latency,
                f"{model_name}_error_count": 1 if error else 0
            }, step=int(time.time()))
```

### 7.4 Viewing Results

**Start MLflow UI**:
```bash
mlflow ui
```

**Access**:
- http://localhost:5000

**Experiments**:
- `ctr_prediction_training`: Model training experiments
- `ctr_prediction_ab_tests`: A/B test experiments
- `ctr_prediction_production`: Production monitoring

---

## 8. Best Practices {#best-practices}

### 8.1 Organizing Experiments

**Use Descriptive Names**:
```python
# Good
mlflow.set_experiment("ctr_prediction_training")
mlflow.set_experiment("ctr_prediction_ab_tests")

# Bad
mlflow.set_experiment("exp1")
mlflow.set_experiment("test")
```

**Use Tags for Organization**:
```python
mlflow.set_tags({
    "team": "ml-engineering",
    "project": "ctr-prediction",
    "model_type": "xgboost",
    "status": "completed"
})
```

### 8.2 Logging Parameters

**Log All Hyperparameters**:
```python
# Log all parameters that affect results
mlflow.log_params({
    "learning_rate": 0.1,
    "max_depth": 6,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
})
```

**Log Data Parameters**:
```python
mlflow.log_params({
    "train_size": len(X_train),
    "test_size": len(X_test),
    "num_features": X_train.shape[1],
    "data_version": "v1.2.3"
})
```

### 8.3 Logging Metrics

**Log Multiple Metrics**:
```python
mlflow.log_metrics({
    "train_accuracy": train_acc,
    "val_accuracy": val_acc,
    "test_accuracy": test_acc,
    "train_roc_auc": train_roc,
    "val_roc_auc": val_roc,
    "test_roc_auc": test_roc
})
```

**Use Consistent Metric Names**:
```python
# Good: Consistent naming
mlflow.log_metric("val_roc_auc", val_roc)
mlflow.log_metric("test_roc_auc", test_roc)

# Bad: Inconsistent
mlflow.log_metric("val_roc_auc", val_roc)
mlflow.log_metric("test_auc", test_roc)  # Different naming
```

### 8.4 Model Versioning

**Register Models**:
```python
# Register with descriptive name
mlflow.sklearn.log_model(
    model,
    "model",
    registered_model_name="ctr_predictor_xgboost"
)
```

**Use Stages**:
```python
# Development → Staging → Production
client.transition_model_version_stage(
    name="ctr_predictor_xgboost",
    version=1,
    stage="Staging"
)
```

### 8.5 Reproducibility

**Log Code Version**:
```python
# MLflow automatically logs Git commit if available
# Or manually:
mlflow.log_param("git_commit", get_git_commit())
mlflow.log_param("code_version", "v1.2.3")
```

**Log Environment**:
```python
# Autologging captures package versions
# Or manually:
import sys
mlflow.log_param("python_version", sys.version)
```

**Log Random Seeds**:
```python
mlflow.log_param("random_state", 42)
np.random.seed(42)
```

### 8.6 Performance Considerations

**Batch Metric Logging**:
```python
# Good: Batch logging
metrics = {
    "accuracy": 0.95,
    "roc_auc": 0.92,
    "log_loss": 0.15
}
mlflow.log_metrics(metrics)

# Less efficient: Individual logging
mlflow.log_metric("accuracy", 0.95)
mlflow.log_metric("roc_auc", 0.92)
mlflow.log_metric("log_loss", 0.15)
```

**Log Artifacts Efficiently**:
```python
# Log directory instead of individual files
mlflow.log_artifacts("outputs/", "outputs")

# Instead of:
for file in os.listdir("outputs/"):
    mlflow.log_artifact(f"outputs/{file}")
```

### 8.7 Error Handling

**Handle MLflow Errors Gracefully**:
```python
try:
    mlflow.log_metric("accuracy", accuracy)
except Exception as e:
    logger.warning(f"Failed to log to MLflow: {e}")
    # Continue execution without MLflow
```

**Check MLflow Availability**:
```python
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available, continuing without tracking")

if MLFLOW_AVAILABLE:
    mlflow.log_metric("accuracy", accuracy)
```

---

## 9. Troubleshooting {#troubleshooting}

### 9.1 Common Issues

#### Issue: "MLflow run not showing in UI"

**Causes**:
- Tracking URI mismatch
- Run not ended properly
- Nested runs not visible

**Solutions**:
```python
# Ensure tracking URI matches
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# Ensure run is ended
with mlflow.start_run():
    # Your code
    pass  # Run automatically ends here

# Or manually
run = mlflow.start_run()
# Your code
mlflow.end_run()
```

#### Issue: "Windows path errors"

**Problem**: `file://./mlruns` doesn't work on Windows

**Solution**:
```python
from pathlib import Path
import os

mlruns_dir = Path("mlruns").absolute()
mlruns_dir.mkdir(exist_ok=True)

if os.name == 'nt':  # Windows
    tracking_uri = f"file:///{str(mlruns_dir).replace(chr(92), '/')}"
else:
    tracking_uri = f"file://{mlruns_dir.as_posix()}"

mlflow.set_tracking_uri(tracking_uri)
```

#### Issue: "Autologging not working"

**Causes**:
- Autologging not enabled
- Framework not supported
- Version incompatibility

**Solutions**:
```python
# Enable autologging before training
mlflow.sklearn.autolog()

# Check if framework is supported
import mlflow.sklearn
print(mlflow.sklearn.__version__)

# Update MLflow
pip install --upgrade mlflow
```

#### Issue: "Model loading fails"

**Causes**:
- Model flavor mismatch
- Missing dependencies
- Wrong run ID

**Solutions**:
```python
# Use correct flavor
model = mlflow.sklearn.load_model("runs:/<run_id>/model")

# Or use PyFunc (more flexible)
model = mlflow.pyfunc.load_model("runs:/<run_id>/model")

# Check run exists
from mlflow.tracking import MlflowClient
client = MlflowClient()
run = client.get_run(run_id)
print(run.info)
```

### 9.2 Debugging Tips

**Enable Debug Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check Active Run**:
```python
run = mlflow.active_run()
if run:
    print(f"Active run: {run.info.run_id}")
else:
    print("No active run")
```

**Verify Tracking URI**:
```python
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experiment: {mlflow.get_experiment_by_name('my_experiment')}")
```

**List All Runs**:
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("my_experiment")
runs = client.search_runs([experiment.experiment_id])
for run in runs:
    print(f"{run.info.run_id}: {run.data.metrics}")
```

---

## 10. Real-World Examples {#real-world-examples}

### Example 1: Model Training with Full Tracking

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd

# Setup
mlflow.set_experiment("ctr_prediction_training")
mlflow.sklearn.autolog()

# Load data
df = pd.read_csv("data/processed/features.csv")
X = df.drop("clicked", axis=1)
y = df["clicked"]

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Training with MLflow
with mlflow.start_run(run_name="rf_baseline"):
    # Log data info
    mlflow.log_params({
        "train_size": len(X_train),
        "test_size": len(X_test),
        "num_features": X_train.shape[1],
        "random_state": 42
    })
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Log metrics (autologging also logs these)
    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_roc_auc": roc_auc
    })
    
    # Log model
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="ctr_predictor_rf"
    )
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
```

### Example 2: Hyperparameter Tuning

```python
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

mlflow.set_experiment("hyperparameter_tuning")
mlflow.sklearn.autolog()

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}

with mlflow.start_run(run_name="grid_search"):
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='roc_auc'
    )
    grid_search.fit(X_train, y_train)
    
    # Log best parameters
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    
    # Log best model
    mlflow.sklearn.log_model(
        grid_search.best_estimator_,
        "best_model"
    )
```

### Example 3: Comparing Multiple Models

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

mlflow.set_experiment("model_comparison")

models = {
    "random_forest": RandomForestClassifier(n_estimators=100),
    "logistic_regression": LogisticRegression(max_iter=1000),
    "xgboost": xgb.XGBClassifier()
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model_type", name)
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model
        if name == "xgboost":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
```

### Example 4: Production Monitoring

```python
import mlflow
import time
from datetime import datetime

mlflow.set_experiment("production_monitoring")

# Start a long-running run for monitoring
run = mlflow.start_run(run_name="production_monitoring")

try:
    while True:
        # Collect metrics
        latency = get_average_latency()
        error_rate = get_error_rate()
        request_count = get_request_count()
        
        # Log metrics with timestamp
        timestamp = int(time.time())
        mlflow.log_metrics({
            "latency_ms": latency,
            "error_rate": error_rate,
            "request_count": request_count
        }, step=timestamp)
        
        time.sleep(60)  # Log every minute
except KeyboardInterrupt:
    mlflow.end_run()
```

---

## Conclusion

MLflow is a powerful tool for managing the machine learning lifecycle. By tracking experiments, organizing models, and enabling collaboration, MLflow helps you build better models faster.

### Key Takeaways

1. **Start Simple**: Begin with basic tracking, add complexity as needed
2. **Use Autologging**: Saves time and ensures consistency
3. **Organize Well**: Use descriptive names, tags, and experiments
4. **Log Everything**: Parameters, metrics, models, artifacts
5. **Monitor Continuously**: Track production metrics
6. **Version Models**: Use Model Registry for deployment
7. **Reproduce Results**: Log code versions and environments

### Next Steps

1. **Try Basic Tracking**: Run the simple example
2. **Explore UI**: Start `mlflow ui` and explore
3. **Integrate with Your Code**: Add MLflow to your training scripts
4. **Set Up Model Registry**: Organize your models
5. **Monitor Production**: Track real-time metrics

### Resources

- **Documentation**: https://mlflow.org/docs/latest/index.html
- **GitHub**: https://github.com/mlflow/mlflow
- **Examples**: https://github.com/mlflow/mlflow/tree/master/examples

Happy Tracking! 📊🚀

