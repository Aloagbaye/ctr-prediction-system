# CTR Prediction System

A complete end-to-end machine learning system for predicting Click-Through Rate (CTR) of ad impressions. This project covers data generation, feature engineering, model training, evaluation, and production deployment.

## 🎯 Project Status

- ✅ **Phase 1**: Data Preparation & Exploration
- ✅ **Phase 2**: Feature Engineering (Current)
- ⏳ Phase 3: Baseline Model Development
- ⏳ Phase 4: Model Optimization
- ⏳ Phase 5: Model Evaluation
- ⏳ Phase 6: API Development
- ⏳ Phase 7: Containerization
- ⏳ Phase 8: Cloud Deployment
- ⏳ Phase 9: Online Evaluation & Monitoring

## 📋 Phase 1: Data Preparation & Exploration

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Option A: Use Simulated Data (Quick Start)

2. **Generate simulated data:**
   ```bash
   python scripts/generate_data.py --n-impressions 100000 --output data/raw/impressions.csv
   ```

   Or use custom parameters:
   ```bash
   python scripts/generate_data.py \
     --n-users 10000 \
     --n-ads 1000 \
     --n-impressions 100000 \
     --output data/raw/impressions.csv \
     --seed 42
   ```

3. **Run exploratory data analysis:**
   ```bash
   python scripts/explore_data.py --input data/raw/impressions.csv --output-dir data/eda
   ```

### Option B: Use Real Kaggle Avazu Data (Recommended for Production)

2. **Set up Kaggle API credentials:**
   - Go to https://www.kaggle.com/account
   - Scroll to 'API' section and click 'Create New API Token'
   - Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac)

3. **Download Avazu dataset:**
   ```bash
   python scripts/download_avazu.py --download
   ```

4. **Load and process Avazu data:**
   ```bash
   # Load a sample (10% of data - recommended for initial exploration)
   python scripts/download_avazu.py --load --sample-frac 0.1 --output data/raw/avazu_processed.csv
   
   # Or load specific number of rows (for testing)
   python scripts/download_avazu.py --load --n-rows 100000 --output data/raw/avazu_processed.csv
   ```

5. **Run exploratory data analysis:**
   ```bash
   python scripts/explore_data.py --input data/raw/avazu_processed.csv --output-dir data/eda
   ```

### Dataset Structure

The generated dataset contains the following columns:

- `user_id`: Unique user identifier
- `ad_id`: Unique advertisement identifier
- `device`: Device type (mobile, desktop, tablet)
- `placement`: Ad placement location (header, sidebar, footer, in_content, popup)
- `timestamp`: Timestamp of the impression
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `is_weekend`: Binary indicator (0 or 1)
- `date`: Date of impression
- `clicked`: Target variable (0 or 1)

### EDA Outputs

The EDA script generates:
- Class imbalance analysis
- Temporal patterns (hourly, daily)
- Device and placement CTR analysis
- Feature distributions
- All visualizations saved to `data/eda/`

### Project Structure

```
ctr-prediction-system/
├── data/
│   ├── raw/              # Raw data files
│   │   └── avazu/        # Avazu dataset (from Kaggle)
│   ├── processed/        # Processed features
│   └── eda/              # EDA plots and reports
├── src/
│   ├── data/
│   │   ├── generator.py      # Simulated data generation
│   │   ├── kaggle_loader.py  # Kaggle Avazu data loader
│   │   └── explore.py        # EDA analysis
│   └── features/
│       ├── basic_features.py    # Basic feature engineering
│       ├── advanced_features.py # Advanced encoding features
│       ├── feature_selector.py  # Feature selection
│       └── pipeline.py          # Complete pipeline
├── scripts/
│   ├── generate_data.py     # Simulated data generation script
│   ├── download_avazu.py    # Avazu dataset downloader
│   ├── explore_data.py      # EDA script
│   └── create_features.py   # Feature engineering script
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
└── README.md
```

## 🚀 Quick Start

### Using Simulated Data

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data
python scripts/generate_data.py

# 3. Explore data
python scripts/explore_data.py
```

### Using Real Avazu Data

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up Kaggle credentials (see instructions above)

# 3. Download and load Avazu data
python scripts/download_avazu.py --download
python scripts/download_avazu.py --load --sample-frac 0.1

# 4. Explore data
python scripts/explore_data.py --input data/raw/avazu_processed.csv
```

## 📋 Phase 2: Feature Engineering

### Create Features

After preparing your data (Phase 1), create engineered features:

```bash
# Create all features (basic + advanced + selection)
python scripts/create_features.py \
  --input data/raw/impressions.csv \
  --output data/processed/features.csv

# Create features with custom options
python scripts/create_features.py \
  --input data/raw/impressions.csv \
  --output data/processed/features.csv \
  --select-by-importance \
  --n-features 50
```

### Feature Types Created

- **Historical CTR**: User-level, ad-level, device-level CTR
- **Temporal**: Hour, day, month, business hours, time of day
- **Counts**: Impression counts, time since last click
- **Encoding**: Frequency encoding, target encoding for high-cardinality features
- **Interactions**: Device×placement, user×device, hour×placement

## 📚 Documentation

- **[TUTORIAL_PLAN.md](TUTORIAL_PLAN.md)**: Complete tutorial plan covering all phases
- **[Walkthrough Tutorials](walkthrough/)**: Detailed step-by-step guides for each phase
  - [Phase 1: Data Preparation & Exploration](walkthrough/PHASE_1.md) ✅
  - [Phase 2: Feature Engineering](walkthrough/PHASE_2.md) ✅

## 🛠 Technology Stack

- **Python 3.9+**
- **Pandas, NumPy**: Data manipulation
- **Matplotlib, Seaborn**: Visualization
- **Scikit-learn, XGBoost, LightGBM**: Machine learning
- **FastAPI**: API framework (Phase 6+)
- **Docker**: Containerization (Phase 7+)
- **GCP/AWS**: Cloud deployment (Phase 8+)

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.