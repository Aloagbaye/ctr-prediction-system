# Phase 2: Feature Engineering - Walkthrough

## 📚 Overview

In this phase, you'll learn how to:
- Create basic features from raw data (historical CTR, temporal features, counts)
- Handle high-cardinality categorical features using advanced techniques
- Create interaction features
- Select the most important features

**Estimated Time**: 3-4 hours  
**Difficulty**: Intermediate

---

## 🎯 Learning Objectives

By the end of this phase, you will:
1. Understand different types of features for CTR prediction
2. Know how to create historical/aggregate features without data leakage
3. Master techniques for handling high-cardinality categoricals
4. Be able to create interaction features
5. Understand feature selection methods

---

## 📋 Prerequisites

- Completed Phase 1 (Data Preparation & Exploration)
- Understanding of pandas and data manipulation
- Basic knowledge of machine learning concepts
- Familiarity with cross-validation (helpful)

---

## 🚀 Step-by-Step Tutorial

### Step 1: Understanding Feature Types

Before creating features, let's understand what we need:

#### 1.1 Basic Features

**Historical CTR Features:**
- User-level CTR: How often does this user click?
- Ad-level CTR: How often is this ad clicked?
- Device-level CTR: CTR by device type
- Placement CTR: CTR by placement

**Temporal Features:**
- Hour, day of week, month
- Business hours indicator
- Time of day (morning/afternoon/evening/night)

**Count Features:**
- Number of impressions per user
- Number of impressions per ad
- Time since last click/impression

#### 1.2 Advanced Features

**High-Cardinality Categorical Handling:**
- User ID and Ad ID have many unique values
- Need special encoding techniques:
  - Feature hashing (hashing trick)
  - Target encoding (mean encoding)
  - Frequency encoding

**Interaction Features:**
- Device × Placement combinations
- User × Device interactions
- Hour × Placement interactions

#### 1.3 Feature Selection

- Remove low-variance features
- Remove highly correlated features
- Select by importance (using model)

---

### Step 2: Basic Feature Engineering

#### 2.1 Create Basic Features

Run the feature engineering pipeline:

```bash
python scripts/create_features.py \
  --input data/raw/impressions.csv \
  --output data/processed/features.csv
```

This will:
1. Create basic features (historical CTR, temporal, counts)
2. Create advanced features (encoding for high-cardinality)
3. Apply feature selection

#### 2.2 Understanding Basic Features

Let's examine what features were created:

```python
import pandas as pd

# Load processed data
df = pd.read_csv('data/processed/features.csv')

# See all feature columns
feature_cols = [col for col in df.columns if col != 'clicked']
print(f"Total features: {len(feature_cols)}")
print("\nFeature categories:")

# Historical CTR features
ctr_features = [col for col in feature_cols if 'ctr' in col.lower()]
print(f"\nHistorical CTR features ({len(ctr_features)}):")
for feat in ctr_features[:10]:
    print(f"  - {feat}")

# Temporal features
temporal_features = [col for col in feature_cols if col in ['hour', 'day_of_week', 'month', 'is_business_hours']]
print(f"\nTemporal features ({len(temporal_features)}):")
for feat in temporal_features:
    print(f"  - {feat}")

# Count features
count_features = [col for col in feature_cols if 'count' in col.lower() or 'impressions' in col.lower()]
print(f"\nCount features ({len(count_features)}):")
for feat in count_features:
    print(f"  - {feat}")
```

#### 2.3 Verify Feature Quality

Check for issues:

```python
# Check for missing values
missing = df[feature_cols].isnull().sum()
print("Missing values:")
print(missing[missing > 0])

# Check for infinite values
inf_cols = []
for col in feature_cols:
    if np.isinf(df[col]).any():
        inf_cols.append(col)
print(f"\nColumns with infinite values: {inf_cols}")

# Check feature distributions
print("\nFeature statistics:")
print(df[feature_cols].describe())
```

---

### Step 3: Advanced Feature Engineering

#### 3.1 Understanding High-Cardinality Features

High-cardinality features (like `user_id`, `ad_id`) have many unique values:
- Can't use one-hot encoding (too many columns)
- Need special techniques

**Techniques we use:**

1. **Frequency Encoding**: How often does this value appear?
   - `user_id_freq`: Number of times this user appears
   - `ad_id_freq`: Number of times this ad appears

2. **Target Encoding**: What's the average CTR for this value?
   - `user_id_target_enc`: Average CTR for this user
   - `ad_id_target_enc`: Average CTR for this ad
   - Uses cross-validation to prevent data leakage

3. **Feature Hashing**: Hash values to fixed number of features
   - `user_id_hash_0` to `user_id_hash_9`: Binary hash features

#### 3.2 Inspect Advanced Features

```python
# Check encoding features
encoding_features = [col for col in feature_cols 
                    if 'freq' in col or 'target_enc' in col or 'hash' in col]
print(f"Encoding features ({len(encoding_features)}):")
for feat in encoding_features[:15]:
    print(f"  - {feat}")

# Check target encoding values
if 'user_id_target_enc' in df.columns:
    print("\nUser target encoding statistics:")
    print(df['user_id_target_enc'].describe())
    
    # Visualize distribution
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(df['user_id_target_enc'], bins=50, edgecolor='black')
    plt.title('Distribution of User Target Encoding')
    plt.xlabel('Target Encoding Value')
    plt.ylabel('Frequency')
    plt.show()
```

---

### Step 4: Interaction Features

#### 4.1 Understanding Interactions

Interaction features capture relationships between features:
- `device_placement`: Which device-placement combinations work best?
- `user_device_ctr`: User's CTR on specific devices
- `hour_placement_ctr`: CTR by hour and placement

#### 4.2 Inspect Interaction Features

```python
# Check interaction features
interaction_features = [col for col in feature_cols 
                       if '_' in col and ('device' in col or 'placement' in col or 'hour' in col)]
print(f"Interaction features ({len(interaction_features)}):")
for feat in interaction_features:
    print(f"  - {feat}")

# Example: Device-placement combinations
if 'device_placement_ctr' in df.columns:
    print("\nDevice-Placement CTR:")
    print(df.groupby('device_placement')['device_placement_ctr'].first())
```

---

### Step 5: Feature Selection

#### 5.1 Understanding Feature Selection

Feature selection removes:
- **Low-variance features**: Features that don't vary much (not informative)
- **Highly correlated features**: Redundant features
- **Low-importance features**: Features that don't help prediction

#### 5.2 Feature Selection Results

The pipeline automatically:
1. Removes features with variance < 0.01
2. Removes features with correlation > 0.95
3. Optionally selects top N features by importance

Check what was removed:

```python
# Compare original vs processed
# (You'd need to track this, but for now, just check final features)

print(f"Final number of features: {len(feature_cols)}")
print(f"Target column: clicked")

# Check feature importance (if available)
# This would require training a model first (Phase 3)
```

---

### Step 6: Custom Feature Engineering

#### 6.1 Create Custom Features

You can add your own features. Example:

```python
import pandas as pd
from src.features.pipeline import FeatureEngineeringPipeline

# Load raw data
df = pd.read_csv('data/raw/impressions.csv')

# Create custom feature
df['user_ad_interaction'] = df['user_id'].astype(str) + '_' + df['ad_id'].astype(str)

# Count interactions
interaction_counts = df.groupby('user_ad_interaction').size()
df['user_ad_interaction_count'] = df['user_ad_interaction'].map(interaction_counts)

# Now run pipeline
pipeline = FeatureEngineeringPipeline()
df_processed = pipeline.fit_transform(df)

# Your custom feature will be included
```

#### 6.2 Feature Engineering Best Practices

1. **Avoid Data Leakage**:
   - Use only past data for historical features
   - Use cross-validation for target encoding
   - Don't use future information

2. **Handle Missing Values**:
   - Fill with appropriate defaults (global mean, 0, etc.)
   - Consider creating "missing" indicator features

3. **Normalize/Scale** (if needed):
   - Some models (neural networks) need scaling
   - Tree-based models (XGBoost, LightGBM) don't need scaling

4. **Document Features**:
   - Keep track of what each feature means
   - Note any transformations applied

---

### Step 7: Feature Analysis

#### 7.1 Feature Importance (Preview)

While full feature importance comes in Phase 3, you can get a preview:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load features
df = pd.read_csv('data/processed/features.csv')

# Prepare data
X = df.drop('clicked', axis=1).select_dtypes(include=[np.number])
y = df['clicked']

# Train simple model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 20 most important features:")
print(feature_importance.head(20))
```

#### 7.2 Feature Correlation Analysis

Check for highly correlated features:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Select numeric features
numeric_features = df[feature_cols].select_dtypes(include=[np.number])

# Calculate correlation matrix (sample for visualization)
corr_matrix = numeric_features.sample(min(1000, len(df))).corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('data/eda/feature_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
```

---

### Step 8: Save and Document

#### 8.1 Save Processed Data

The pipeline automatically saves processed data, but you can also save intermediate steps:

```python
# Save after basic features only
from src.features.basic_features import BasicFeatureEngineer

df = pd.read_csv('data/raw/impressions.csv')
engineer = BasicFeatureEngineer()
df_basic = engineer.fit_transform(df)
df_basic.to_csv('data/processed/basic_features.csv', index=False)
```

#### 8.2 Document Features

Create a feature documentation file:

```python
# Create feature documentation
feature_docs = {
    'Historical CTR Features': [
        'ctr_user_id_user: User historical CTR',
        'ctr_ad_id_ad: Ad historical CTR',
        'ctr_device_device: Device historical CTR',
    ],
    'Temporal Features': [
        'hour: Hour of day (0-23)',
        'day_of_week: Day of week (0=Monday)',
        'is_weekend: Weekend indicator',
        'is_business_hours: Business hours (9-17) indicator',
    ],
    'Encoding Features': [
        'user_id_freq: Frequency of user',
        'user_id_target_enc: User target encoding',
        'ad_id_freq: Frequency of ad',
        'ad_id_target_enc: Ad target encoding',
    ],
    # ... add more
}

# Save to file
import json
with open('data/processed/feature_documentation.json', 'w') as f:
    json.dump(feature_docs, f, indent=2)
```

---

## 🎓 Key Concepts Learned

### 1. Historical Features
- Use past data to predict future
- Must avoid data leakage (use shift/expanding windows)
- Important for capturing user/ad behavior

### 2. Target Encoding
- Mean encoding with cross-validation
- Prevents data leakage
- Very effective for high-cardinality categoricals

### 3. Feature Hashing
- Maps many values to fixed number of features
- Useful for extremely high-cardinality features
- May have collisions but often works well

### 4. Feature Selection
- Reduces dimensionality
- Removes noise and redundancy
- Can improve model performance and speed

---

## 🔍 Common Issues & Solutions

### Issue 1: Memory Error

**Problem**: Running out of memory during feature creation

**Solution**:
- Process data in chunks
- Use smaller dataset for development
- Disable some feature creation steps

```bash
# Skip advanced features
python scripts/create_features.py --no-advanced
```

### Issue 2: Data Leakage

**Problem**: Features using future information

**Solution**:
- Always use `shift()` or `expanding()` for historical features
- Use cross-validation for target encoding
- Sort by timestamp before creating features

### Issue 3: Too Many Features

**Problem**: Created too many features, model is slow

**Solution**:
- Enable feature selection
- Use `--select-by-importance --n-features 50`
- Remove low-variance features

### Issue 4: Missing Values in Features

**Problem**: Some features have NaN values

**Solution**:
- Fill with appropriate defaults (mean, 0, etc.)
- Check why values are missing
- Consider creating "missing" indicator

---

## 📊 Expected Results

After completing this phase, you should have:

1. **Processed Dataset**: `data/processed/features.csv` with engineered features
2. **Feature Count**: 50-200 features (depending on options)
3. **Feature Types**:
   - Historical CTR features
   - Temporal features
   - Encoding features (frequency, target encoding)
   - Interaction features
4. **Documentation**: Understanding of what each feature represents

---

## ✅ Phase 2 Checklist

- [ ] Created basic features (historical CTR, temporal, counts)
- [ ] Created advanced features (encoding for high-cardinality)
- [ ] Created interaction features
- [ ] Applied feature selection
- [ ] Verified no data leakage
- [ ] Checked for missing values
- [ ] Analyzed feature importance (preview)
- [ ] Documented features
- [ ] Ready for Phase 3 (Model Training)

---

## 🚀 Next Steps

Once you've completed Phase 2, you're ready for:

**Phase 3: Baseline Model Development**
- Train Logistic Regression baseline
- Train XGBoost model
- Train LightGBM model
- Evaluate model performance

---

## 📚 Additional Resources

- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Target Encoding Explained](https://maxhalford.github.io/blog/target-encoding/)
- [Feature Hashing](https://en.wikipedia.org/wiki/Feature_hashing)
- [Feature Selection in Scikit-learn](https://scikit-learn.org/stable/modules/feature_selection.html)

---

## 💡 Tips for Success

1. **Start Simple**: Create basic features first, then add advanced ones
2. **Avoid Leakage**: Always think about temporal order
3. **Monitor Memory**: Large feature sets can use significant memory
4. **Test Incrementally**: Test each feature type separately
5. **Document Everything**: Keep track of what features you create and why

---

**Congratulations on completing Phase 2!** 🎉

You now have a rich set of features ready for model training.

