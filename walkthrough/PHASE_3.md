# Phase 3: Baseline Model Development - Walkthrough

## 📚 Overview

In this phase, you'll learn how to:
- Train baseline models (Logistic Regression, XGBoost, LightGBM)
- Split data into train/validation/test sets
- Evaluate model performance
- Compare different models
- Save trained models for later use

**Estimated Time**: 3-4 hours  
**Difficulty**: Intermediate

---

## 🎯 Learning Objectives

By the end of this phase, you will:
1. Understand how to train baseline models for CTR prediction
2. Know how to handle imbalanced datasets
3. Be able to evaluate models using appropriate metrics
4. Understand the trade-offs between different model types
5. Know how to save and load trained models

---

## 📋 Prerequisites

- Completed Phase 2 (Feature Engineering)
- Understanding of machine learning basics
- Familiarity with scikit-learn, XGBoost, LightGBM

---

## 🚀 Step-by-Step Tutorial

### Step 1: Prepare Your Data

Make sure you have engineered features ready:

```bash
# If you haven't created features yet
python scripts/create_features.py \
  --input data/raw/avazu_processed.parquet \
  --output data/processed/features.csv
```

---

### Step 2: Understanding Baseline Models

#### 2.1 Logistic Regression

**Pros:**
- Simple and interpretable
- Fast training
- Good baseline
- Handles linear relationships well

**Cons:**
- May not capture complex patterns
- Requires feature scaling
- Limited for non-linear relationships

**Use case:** Quick baseline to compare against

#### 2.2 XGBoost

**Pros:**
- Excellent performance
- Handles non-linear relationships
- Built-in regularization
- Feature importance

**Cons:**
- Slower training than logistic regression
- More hyperparameters to tune
- Less interpretable

**Use case:** Primary model for production

#### 2.3 LightGBM

**Pros:**
- Faster than XGBoost
- Similar performance
- Good for large datasets
- Memory efficient

**Cons:**
- May overfit on small datasets
- Less mature than XGBoost

**Use case:** Alternative to XGBoost, especially for large datasets

---

### Step 3: Train Models

#### 3.1 Train All Models

```bash
python scripts/train_models.py \
  --input data/processed/features.csv \
  --output-dir models/ \
  --models logistic xgboost lightgbm
```

This will:
1. Load your features
2. Split into train/validation/test sets
3. Train all specified models
4. Evaluate on test set
5. Save models and results

#### 3.2 Train Specific Models

```bash
# Train only XGBoost
python scripts/train_models.py --models xgboost

# Train Logistic Regression and LightGBM
python scripts/train_models.py --models logistic lightgbm
```

#### 3.3 Customize Training

```bash
# Custom test/validation split
python scripts/train_models.py \
  --test-size 0.15 \
  --val-size 0.15

# Use different random seed
python scripts/train_models.py --random-state 123
```

---

### Step 4: Understanding Model Training

#### 4.1 Data Splitting

The trainer uses stratified splitting to maintain class distribution:

- **Train set**: 64% (80% of 80%)
- **Validation set**: 16% (20% of 80%)
- **Test set**: 20%

Stratified splitting ensures each set has similar CTR distribution.

#### 4.2 Handling Imbalanced Data

CTR prediction is highly imbalanced (~1-5% positive class). Models handle this by:

- **Logistic Regression**: Uses class weights
- **XGBoost**: Uses `scale_pos_weight` parameter
- **LightGBM**: Uses `scale_pos_weight` parameter

#### 4.3 Early Stopping

XGBoost and LightGBM use early stopping on validation set to prevent overfitting:
- Stops training if validation performance doesn't improve
- Default: 10 rounds without improvement

---

### Step 5: Model Evaluation

#### 5.1 Evaluation Metrics

**ROC-AUC (Area Under ROC Curve)**
- Measures ability to distinguish between classes
- Range: 0 to 1 (higher is better)
- Good for imbalanced data
- **Target**: > 0.7 for baseline, > 0.8 for good model

**Log Loss (Logarithmic Loss)**
- Measures probability prediction quality
- Range: 0 to ∞ (lower is better)
- Penalizes confident wrong predictions
- **Target**: < 0.5 for baseline, < 0.4 for good model

**PR-AUC (Precision-Recall AUC)**
- Better for imbalanced data than ROC-AUC
- Focuses on positive class
- **Target**: > 0.3 for baseline, > 0.5 for good model

#### 5.2 Understanding Results

After training, you'll see:

```
MODEL COMPARISON
============================================================
          ROC-AUC  Log Loss   PR-AUC
logistic   0.7234    0.4521   0.3421
xgboost    0.8123    0.3892   0.5234
lightgbm   0.8098    0.3912   0.5187
```

**Interpretation:**
- XGBoost and LightGBM typically perform better than Logistic Regression
- Differences between XGBoost and LightGBM are usually small
- Choose based on speed vs. performance trade-off

---

### Step 6: Understanding Calibration Curves

#### 6.1 What is Model Calibration?

**Calibration** measures how well a model's predicted probabilities match the actual frequencies of events. A well-calibrated model means:
- When the model predicts 0.3 probability, ~30% of those cases should actually be positive
- When it predicts 0.7 probability, ~70% should be positive

**Why it matters:**
- For CTR prediction, you need accurate probabilities, not just rankings
- Business decisions depend on probability estimates (e.g., bid amounts)
- Poor calibration can lead to suboptimal decisions

#### 6.2 Interpreting Calibration Plots

Calibration plots show:
- **X-axis**: Mean predicted probability (binned)
- **Y-axis**: Fraction of positives (actual rate)
- **Diagonal line**: Perfect calibration (predicted = actual)

**How to read the plot:**
- **Above diagonal**: Model over-predicts (probabilities too high)
- **Below diagonal**: Model under-predicts (probabilities too low)
- **Close to diagonal**: Well-calibrated model

#### 6.3 Typical Calibration Patterns

**Logistic Regression:**
- Usually well-calibrated out of the box
- Probabilities are naturally calibrated due to sigmoid function
- May need slight adjustments for extreme probabilities

**XGBoost & LightGBM:**
- Often show **under-prediction** (probabilities too conservative)
- Common pattern: Curve stays below diagonal
- Example: Model predicts 0.6 probability, but actual rate is only 0.2
- This is normal for tree-based models!

**Why tree-based models are miscalibrated:**
- They optimize for ranking (separation), not probability estimation
- Leaf values are averages, not calibrated probabilities
- They tend to be conservative, especially with imbalanced data

#### 6.4 What Your Calibration Curves Tell You

Based on typical results:

**Logistic Regression:**
```
✓ Well-calibrated: Curve follows diagonal closely
✓ Good for: Direct probability interpretation
✓ Use when: You need accurate probability estimates
```

**XGBoost & LightGBM:**
```
⚠️ Under-prediction: Curve below diagonal
✓ Still useful: Good ranking (high ROC-AUC)
✓ Solution: Apply calibration (Platt scaling or isotonic regression)
✓ Use when: Ranking is more important than exact probabilities
```

#### 6.5 Improving Calibration

If calibration is poor, you can:

**Option 1: Platt Scaling (Logistic Regression)**
```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrate XGBoost predictions
calibrated_model = CalibratedClassifierCV(
    model, method='sigmoid', cv=3
)
calibrated_model.fit(X_train, y_train)
```

**Option 2: Isotonic Regression**
```python
calibrated_model = CalibratedClassifierCV(
    model, method='isotonic', cv=3
)
```

**Option 3: Adjust Threshold**
- If probabilities are systematically low, adjust decision threshold
- Instead of 0.5, use 0.3 or 0.4 based on calibration curve

#### 6.6 When to Worry About Calibration

**Good calibration is important when:**
- Making business decisions based on probabilities
- Setting bid amounts in ad auctions
- Calculating expected revenue
- Risk assessment

**Calibration less critical when:**
- Only need ranking (which ads to show)
- Using probabilities for feature engineering
- Relative comparisons matter more than absolute values

**For CTR prediction:**
- **Ranking** (ROC-AUC) is often more important than calibration
- You can improve calibration later if needed
- Focus on model performance first, calibration second

---

### Step 7: Model Inspection

#### 6.1 Feature Importance

Feature importance shows which features the model uses most:

```python
from src.models.evaluator import ModelEvaluator
from src.models.trainer import ModelTrainer
import joblib

# Load model
model = joblib.load('models/xgboost_model.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Plot feature importance
evaluator = ModelEvaluator()
importance_df = evaluator.plot_feature_importance(
    model, feature_names, top_n=20, model_name='XGBoost'
)
```

#### 6.2 Prediction Examples

```python
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('models/xgboost_model.pkl')
scaler = joblib.load('models/logistic_scaler.pkl')  # If using logistic

# Load test data
df_test = pd.read_csv('data/processed/features.csv')
X_test = df_test.drop('clicked', axis=1).select_dtypes(include=[np.number])

# Make predictions
predictions = model.predict_proba(X_test)[:, 1]

# Show examples
results = pd.DataFrame({
    'actual': df_test['clicked'],
    'predicted_prob': predictions,
    'predicted_class': (predictions >= 0.5).astype(int)
})
print(results.head(20))
```

---

### Step 8: Model Comparison

#### 7.1 Compare Models

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load evaluation results
results = pd.read_csv('models/evaluation_results.csv', index_col=0)

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['ROC-AUC', 'Log Loss', 'PR-AUC']
for i, metric in enumerate(metrics):
    results[metric].plot(kind='bar', ax=axes[i])
    axes[i].set_title(metric)
    axes[i].set_ylabel('Score')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

#### 7.2 Choose Best Model

Consider:
1. **Performance**: ROC-AUC, Log Loss
2. **Speed**: Training time, prediction latency
3. **Interpretability**: Feature importance, model complexity
4. **Production requirements**: Memory, dependencies

**Recommendation:**
- **Baseline**: Logistic Regression (fast, interpretable)
- **Production**: XGBoost or LightGBM (best performance)
- **Large datasets**: LightGBM (faster training)

---

### Step 9: Save and Load Models

#### 8.1 Models Are Automatically Saved

After training, models are saved to `models/` directory:
- `logistic_model.pkl`
- `xgboost_model.pkl`
- `lightgbm_model.pkl`
- `logistic_scaler.pkl` (if using logistic regression)
- `feature_names.pkl`

#### 8.2 Load Models for Inference

```python
import joblib

# Load model
model = joblib.load('models/xgboost_model.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Prepare new data (must have same features)
new_data = pd.DataFrame({
    # ... your features ...
})

# Ensure same feature order
new_data = new_data[feature_names]

# Predict
predictions = model.predict_proba(new_data)[:, 1]
```

---

## 🎓 Key Concepts Learned

### 1. Model Selection
- Different models have different strengths
- Start with simple baselines
- Progress to more complex models

### 2. Evaluation Metrics
- Use appropriate metrics for imbalanced data
- ROC-AUC and PR-AUC are key
- Log Loss measures probability quality
- Calibration measures probability accuracy

### 3. Data Splitting
- Train/validation/test split
- Stratified splitting maintains class distribution
- Validation set for early stopping

### 4. Model Persistence
- Save models for reuse
- Save feature names for consistency
- Version control model artifacts

---

## 🔍 Common Issues & Solutions

### Issue 1: Out of Memory

**Problem**: Training fails due to memory issues

**Solution**:
- Use smaller dataset sample
- Reduce number of features
- Use LightGBM (more memory efficient)

### Issue 2: Poor Performance

**Problem**: Low ROC-AUC (< 0.7)

**Solution**:
- Check feature quality
- Try different models
- Tune hyperparameters (Phase 4)
- Add more features

### Issue 3: Overfitting

**Problem**: Large gap between train and validation performance

**Solution**:
- Use early stopping (already enabled)
- Increase regularization
- Reduce model complexity
- Get more data

### Issue 4: Slow Training

**Problem**: Models take too long to train

**Solution**:
- Use smaller dataset for development
- Reduce number of estimators
- Use LightGBM instead of XGBoost
- Enable parallel processing (n_jobs=-1)

### Issue 5: Poor Calibration

**Problem**: Model probabilities don't match actual rates (e.g., predicts 0.6 but only 20% actually click)

**Solution**:
- This is normal for tree-based models (XGBoost, LightGBM)
- Apply Platt scaling or isotonic regression for calibration
- Logistic Regression is usually well-calibrated
- For ranking tasks, calibration is less critical than ROC-AUC

---

## 📊 Expected Results

After completing this phase, you should have:

1. **Trained Models**: 
   - Logistic Regression baseline
   - XGBoost model
   - LightGBM model

2. **Evaluation Results**:
   - ROC-AUC: > 0.7 (baseline), > 0.8 (good)
   - Log Loss: < 0.5 (baseline), < 0.4 (good)
   - PR-AUC: > 0.3 (baseline), > 0.5 (good)

3. **Saved Artifacts**:
   - Model files (.pkl)
   - Evaluation results (CSV)
   - Feature names

---

## ✅ Phase 3 Checklist

- [ ] Created engineered features
- [ ] Trained Logistic Regression baseline
- [ ] Trained XGBoost model
- [ ] Trained LightGBM model
- [ ] Evaluated all models on test set
- [ ] Compared model performance
- [ ] Analyzed calibration curves
- [ ] Saved models and results
- [ ] Inspected feature importance
- [ ] Ready for Phase 4 (Model Optimization)

---

## 🚀 Next Steps

Once you've completed Phase 3, you're ready for:

**Phase 4: Model Optimization**
- Hyperparameter tuning
- Feature selection
- Model ensemble
- Advanced techniques

---

## 📚 Additional Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Scikit-learn Model Selection](https://scikit-learn.org/stable/model_selection.html)
- [ROC-AUC Explained](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)
- [Model Calibration Explained](https://scikit-learn.org/stable/modules/calibration.html)
- [Why Tree-Based Models Need Calibration](https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/)

---

## 💡 Tips for Success

1. **Start Simple**: Begin with Logistic Regression as baseline
2. **Compare Models**: Always compare multiple models
3. **Use Validation Set**: Monitor validation performance during training
4. **Save Everything**: Save models, results, and feature names
5. **Document**: Keep notes on model performance and decisions

---

**Congratulations on completing Phase 3!** 🎉

You now have trained baseline models ready for optimization and deployment.

