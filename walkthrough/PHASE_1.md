# Phase 1: Data Preparation & Exploration - Walkthrough

## 📚 Overview

In this phase, you'll learn how to:
- Set up the project environment
- Generate simulated ad impression data
- Perform comprehensive exploratory data analysis (EDA)
- Understand the characteristics of CTR prediction datasets

**Estimated Time**: 2-3 hours  
**Difficulty**: Beginner to Intermediate

---

## 🎯 Learning Objectives

By the end of this phase, you will:
1. Understand the structure of ad impression data
2. Know how to generate realistic synthetic datasets for ML projects
3. Be able to perform EDA on imbalanced classification datasets
4. Identify key patterns in temporal and categorical features
5. Recognize the importance of class imbalance in CTR prediction

---

## 📋 Prerequisites

- Python 3.9 or higher installed
- Basic understanding of Python and pandas
- Familiarity with data science concepts (optional but helpful)

---

## 🚀 Step-by-Step Tutorial

### Step 1: Environment Setup

#### 1.1 Install Dependencies

First, create a virtual environment (recommended):

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

This will install:
- **pandas** & **numpy**: Data manipulation
- **matplotlib** & **seaborn**: Visualization
- **scikit-learn**: Machine learning utilities
- **xgboost** & **lightgbm**: Gradient boosting models (for later phases)
- **fastapi**: API framework (for later phases)

#### 1.2 Verify Installation

Verify that everything is installed correctly:

```bash
python -c "import pandas, numpy, matplotlib, seaborn; print('✓ All packages installed successfully!')"
```

---

### Step 2: Understanding the Data Structure

Before generating data, let's understand what we're working with.

#### 2.1 Ad Impression Data Schema

An ad impression record contains:

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | Integer | Unique identifier for the user |
| `ad_id` | Integer | Unique identifier for the advertisement |
| `device` | Categorical | Device type: mobile, desktop, or tablet |
| `placement` | Categorical | Ad placement: header, sidebar, footer, in_content, or popup |
| `timestamp` | DateTime | When the impression occurred |
| `hour` | Integer | Hour of day (0-23) |
| `day_of_week` | Integer | Day of week (0=Monday, 6=Sunday) |
| `is_weekend` | Binary | Whether it's a weekend (0 or 1) |
| `date` | Date | Date of the impression |
| `clicked` | Binary | **Target variable**: Whether user clicked (0 or 1) |

#### 2.2 Why Simulate Data?

For this tutorial, we're using simulated data because:
- **Real ad data is proprietary** and hard to obtain
- **Simulated data is reproducible** - same seed = same results
- **We control the patterns** - can create realistic scenarios
- **No privacy concerns** - safe to share and experiment

Real-world alternatives:
- Kaggle Avazu CTR dataset
- Kaggle Criteo CTR dataset
- Your company's ad data (if available)

---

### Step 3: Generate Simulated Data

#### 3.1 Basic Data Generation

Generate a default dataset (100,000 impressions):

```bash
python scripts/generate_data.py
```

This will:
- Create 10,000 unique users
- Create 1,000 unique ads
- Generate 100,000 impressions
- Save to `data/raw/impressions.csv`

#### 3.2 Custom Data Generation

Generate a larger or smaller dataset:

```bash
# Generate 500,000 impressions
python scripts/generate_data.py --n-impressions 500000

# Generate with custom parameters
python scripts/generate_data.py \
  --n-users 50000 \
  --n-ads 5000 \
  --n-impressions 1000000 \
  --output data/raw/large_dataset.csv \
  --seed 123
```

**Parameters explained:**
- `--n-users`: Number of unique users (affects user diversity)
- `--n-ads`: Number of unique ads (affects ad diversity)
- `--n-impressions`: Total impressions (affects dataset size)
- `--output`: Where to save the CSV file
- `--seed`: Random seed for reproducibility

#### 3.3 Understanding the Generator

The data generator (`src/data/generator.py`) creates realistic patterns:

1. **User Segments**:
   - Low clickers (60%): ~1% CTR
   - Medium clickers (30%): ~5% CTR
   - High clickers (10%): ~15% CTR

2. **Ad Quality**:
   - Poor ads (30%): ~1% CTR
   - Average ads (50%): ~5% CTR
   - Good ads (20%): ~12% CTR

3. **Temporal Effects**:
   - Business hours (9-17): +20% CTR boost
   - Weekends: -5% CTR reduction

4. **Device & Placement**:
   - Users prefer certain devices
   - Different placements have different effectiveness

#### 3.4 Verify Generated Data

Quick check of the generated data:

```python
import pandas as pd

df = pd.read_csv('data/raw/impressions.csv')
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nBasic stats:")
print(df.describe())
```

Expected output:
- Shape: (100000, 10) for default settings
- Overall CTR: ~0.03-0.05 (3-5%)
- No missing values

---

### Step 4: Exploratory Data Analysis (EDA)

#### 4.1 Run Complete EDA

Run the automated EDA script:

```bash
python scripts/explore_data.py --input data/raw/impressions.csv --output-dir data/eda
```

This will generate:
- Class imbalance analysis
- Temporal pattern visualizations
- Device and placement analysis
- Feature distributions
- All plots saved to `data/eda/`

#### 4.2 Understanding EDA Outputs

The EDA script produces several analyses:

**1. Basic Information**
- Dataset shape and memory usage
- Date range covered
- Column types

**2. Missing Values**
- Should show no missing values for simulated data
- Important to check for real datasets

**3. Class Imbalance Analysis**
- Shows the distribution of clicks vs. no clicks
- CTR prediction is typically highly imbalanced (1-5% CTR)
- This affects model choice and evaluation metrics

**4. Temporal Patterns**
- **Hourly patterns**: CTR by hour of day
- **Daily patterns**: CTR by day of week
- Helps identify peak engagement times

**5. Device & Placement Analysis**
- Which devices have higher CTR
- Which placements are most effective
- Interaction effects (device × placement)

**6. User & Ad Statistics**
- Distribution of impressions per user
- Distribution of impressions per ad
- Identifies power users and popular ads

#### 4.3 Manual EDA (Optional)

You can also explore the data manually in Python:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/raw/impressions.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 1. Overall CTR
print(f"Overall CTR: {df['clicked'].mean():.4f}")
print(f"Total clicks: {df['clicked'].sum():,}")

# 2. CTR by device
device_ctr = df.groupby('device')['clicked'].mean()
print("\nCTR by Device:")
print(device_ctr)

# 3. CTR by placement
placement_ctr = df.groupby('placement')['clicked'].mean()
print("\nCTR by Placement:")
print(placement_ctr)

# 4. Hourly CTR
hourly_ctr = df.groupby('hour')['clicked'].mean()
plt.figure(figsize=(12, 6))
plt.plot(hourly_ctr.index, hourly_ctr.values, marker='o')
plt.title('CTR by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('CTR')
plt.grid(True)
plt.show()

# 5. User activity distribution
user_impressions = df['user_id'].value_counts()
print(f"\nAverage impressions per user: {user_impressions.mean():.2f}")
print(f"Most active user: {user_impressions.max()} impressions")
```

---

### Step 5: Key Insights from EDA

After running EDA, you should observe:

#### 5.1 Class Imbalance

**Observation**: CTR is typically 1-5%, meaning:
- ~95-99% of impressions are negative (no click)
- ~1-5% are positive (click)

**Implications**:
- Need appropriate metrics (ROC-AUC, not accuracy)
- May need sampling techniques (undersampling, oversampling)
- Consider calibration for probability estimates

#### 5.2 Temporal Patterns

**Common patterns**:
- Higher CTR during business hours (9 AM - 5 PM)
- Lower CTR on weekends
- Peak hours vary by industry

**Action items**:
- Create time-based features (hour, day_of_week, is_weekend)
- Consider time-of-day interactions
- Model temporal trends if data spans long periods

#### 5.3 Device & Placement Effects

**Typical findings**:
- Mobile often has different CTR than desktop
- In-content placements usually perform best
- Popups have lowest CTR (but may have other goals)

**Action items**:
- Include device and placement as categorical features
- Consider device × placement interactions
- May need device-specific models

#### 5.4 User & Ad Heterogeneity

**Observations**:
- Some users click more than others
- Some ads perform better than others
- Long tail distribution (few power users, many casual users)

**Action items**:
- Create user-level features (historical CTR, activity level)
- Create ad-level features (historical CTR, age)
- Consider user-ad interaction features

---

### Step 6: Data Quality Checks

Before moving to feature engineering, verify data quality:

```python
import pandas as pd

df = pd.read_csv('data/raw/impressions.csv')

# 1. Check for missing values
print("Missing values:")
print(df.isnull().sum())

# 2. Check for duplicates
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# 3. Check data types
print("\nData types:")
print(df.dtypes)

# 4. Check value ranges
print("\nValue ranges:")
print(f"User IDs: {df['user_id'].min()} to {df['user_id'].max()}")
print(f"Ad IDs: {df['ad_id'].min()} to {df['ad_id'].max()}")
print(f"Hours: {df['hour'].min()} to {df['hour'].max()}")

# 5. Check for invalid values
assert df['clicked'].isin([0, 1]).all(), "Clicked must be 0 or 1"
assert df['device'].isin(['mobile', 'desktop', 'tablet']).all(), "Invalid device"
assert df['placement'].isin(['header', 'sidebar', 'footer', 'in_content', 'popup']).all(), "Invalid placement"

print("\n✓ All data quality checks passed!")
```

---

### Step 7: Prepare for Next Phase

#### 7.1 Organize Your Findings

Create a summary document of your findings:

```markdown
# EDA Summary

## Dataset Overview
- Total impressions: X
- Date range: Y to Z
- Overall CTR: X%

## Key Findings
1. Class imbalance: X% positive class
2. Temporal patterns: [describe]
3. Device effects: [describe]
4. Placement effects: [describe]

## Data Quality
- Missing values: None
- Duplicates: X
- Data types: All correct

## Recommendations for Feature Engineering
1. [Feature idea 1]
2. [Feature idea 2]
3. [Feature idea 3]
```

#### 7.2 Save Your Work

Make sure you have:
- ✅ Generated dataset in `data/raw/impressions.csv`
- ✅ EDA plots in `data/eda/`
- ✅ Understanding of data characteristics
- ✅ Notes on potential features

---

## 🎓 Key Concepts Learned

### 1. Ad Impression Data
- Structure of ad impression logs
- Target variable: clicked (binary)
- Context features: user, ad, device, placement, time

### 2. Class Imbalance
- CTR prediction is highly imbalanced
- Need appropriate evaluation metrics
- May require special handling techniques

### 3. Temporal Patterns
- Time-based features are important
- Business hours vs. off-hours
- Day of week effects

### 4. Categorical Features
- Device type (low cardinality)
- Placement (low cardinality)
- User ID and Ad ID (high cardinality - will need special handling)

---

## 🔍 Common Issues & Solutions

### Issue 1: Memory Error with Large Datasets

**Solution**: Process data in chunks or use smaller dataset for development:

```bash
python scripts/generate_data.py --n-impressions 50000
```

### Issue 2: EDA Plots Not Saving

**Solution**: Ensure output directory exists:

```bash
mkdir -p data/eda
python scripts/explore_data.py
```

### Issue 3: Import Errors

**Solution**: Make sure you're in the project root and packages are installed:

```bash
pip install -r requirements.txt
```

### Issue 4: Different Results Each Run

**Solution**: Use a fixed seed:

```bash
python scripts/generate_data.py --seed 42
```

---

## 📊 Expected Results

After completing this phase, you should have:

1. **Dataset**: `data/raw/impressions.csv` with 100,000+ impressions
2. **EDA Plots**: 
   - `class_imbalance.png`
   - `temporal_hourly.png`
   - `temporal_daily.png`
   - `device_placement_ctr.png`
   - `device_placement_heatmap.png`
3. **Understanding**: 
   - Dataset structure
   - Class imbalance (~3-5% CTR)
   - Temporal patterns
   - Feature distributions

---

## ✅ Phase 1 Checklist

- [ ] Installed all dependencies
- [ ] Generated simulated dataset
- [ ] Verified data quality
- [ ] Ran complete EDA
- [ ] Reviewed all EDA plots
- [ ] Documented key findings
- [ ] Understood class imbalance
- [ ] Identified temporal patterns
- [ ] Ready for Phase 2 (Feature Engineering)

---

## 🚀 Next Steps

Once you've completed Phase 1, you're ready for:

**Phase 2: Feature Engineering**
- Create user-level features
- Create ad-level features
- Handle high-cardinality categoricals
- Create interaction features
- Feature selection

---

## 📚 Additional Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Tutorial](https://matplotlib.org/stable/tutorials/index.html)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Class Imbalance in ML](https://machinelearningmastery.com/what-is-imbalanced-classification/)

---

## 💡 Tips for Success

1. **Start Small**: Use smaller datasets (50K-100K) for initial exploration
2. **Visualize Everything**: Plots reveal patterns numbers don't
3. **Document Findings**: Take notes on what you discover
4. **Ask Questions**: Why does CTR vary by hour? What explains device differences?
5. **Think Ahead**: Consider what features would help a model predict clicks

---

**Congratulations on completing Phase 1!** 🎉

You now have a solid understanding of your data and are ready to engineer features that will help predict CTR.

