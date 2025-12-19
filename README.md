# CTR Prediction System

A complete end-to-end machine learning system for predicting Click-Through Rate (CTR) of ad impressions. This project covers data generation, feature engineering, model training, evaluation, and production deployment.

## 🎯 Project Status

- ✅ **Phase 1**: Data Preparation & Exploration
- ✅ **Phase 2**: Feature Engineering
- ✅ **Phase 3**: Baseline Model Development
- ⏳ Phase 4: Model Optimization
- ⏳ Phase 5: Model Evaluation
- ✅ **Phase 6**: API Development
- ✅ **Phase 6.2**: Streamlit UI with LLM Explanations
- ✅ **Phase 7**: Containerization
- ✅ **Phase 8**: Cloud Deployment (GCP)
- ✅ **Phase 8.2**: Vertex AI & BigQuery Integration (Current)
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
│   ├── features/
│   │   ├── basic_features.py    # Basic feature engineering
│   │   ├── advanced_features.py # Advanced encoding features
│   │   ├── feature_selector.py  # Feature selection
│   │   └── pipeline.py          # Complete pipeline
│   ├── models/
│   │   ├── trainer.py        # Model training
│   │   └── evaluator.py      # Model evaluation
│   ├── api/
│   │   ├── main.py           # FastAPI application
│   │   ├── models.py         # Pydantic request/response models
│   │   └── predictor.py      # Model prediction logic
│   └── app/
│       ├── streamlit_app.py  # Streamlit UI application
│       └── llm_explainer.py  # LLM explanation generator
├── scripts/
│   ├── generate_data.py     # Simulated data generation script
│   ├── download_avazu.py    # Avazu dataset downloader
│   ├── explore_data.py      # EDA script
│   ├── create_features.py   # Feature engineering script
│   ├── train_models.py      # Model training script
│   ├── run_api.py           # API server script
│   ├── run_streamlit.py     # Streamlit app script
│   ├── docker_build.sh      # Docker build script (Linux/Mac)
│   ├── docker_build.ps1    # Docker build script (Windows)
│   ├── docker_run.sh        # Docker run script (Linux/Mac)
│   ├── docker_run.ps1       # Docker run script (Windows)
│   ├── gcp_deploy.sh        # GCP deployment script (Linux/Mac)
│   ├── gcp_deploy.ps1       # GCP deployment script (Windows)
│   ├── gcp_cleanup.sh       # GCP cleanup script (Linux/Mac)
│   ├── gcp_cleanup.ps1      # GCP cleanup script (Windows)
│   ├── vertex_ai_upload.sh  # Vertex AI model upload (Linux/Mac)
│   ├── vertex_ai_upload.ps1 # Vertex AI model upload (Windows)
│   ├── bigquery_setup.sh    # BigQuery setup (Linux/Mac)
│   └── bigquery_setup.ps1   # BigQuery setup (Windows)
├── Dockerfile.api           # Dockerfile for API service
├── Dockerfile.streamlit     # Dockerfile for Streamlit UI
├── docker-compose.yml      # Docker Compose configuration
├── .dockerignore           # Docker ignore file
└── models/                  # Trained models (generated)
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

# Create features with sampling for faster processing (10% of data)
python scripts/create_features.py \
  --input data/raw/avazu_processed.parquet \
  --output data/processed/features.csv \
  --sample-frac 0.1

# Create features with custom options
python scripts/create_features.py \
  --input data/raw/impressions.csv \
  --output data/processed/features.csv \
  --select-by-importance \
  --n-features 50 \
  --sample-frac 0.1
```

### Feature Types Created

- **Historical CTR**: User-level, ad-level, device-level CTR
- **Temporal**: Hour, day, month, business hours, time of day
- **Counts**: Impression counts, time since last click
- **Encoding**: Frequency encoding, target encoding for high-cardinality features
- **Interactions**: Device×placement, user×device, hour×placement

## 📋 Phase 3: Baseline Model Development

### Train Models

After creating features (Phase 2), train baseline models:

```bash
# Train all models (Logistic Regression, XGBoost, LightGBM)
python scripts/train_models.py \
  --input data/processed/features.csv \
  --output-dir models/

# Train specific models
python scripts/train_models.py --models xgboost lightgbm

# Customize training parameters
python scripts/train_models.py \
  --test-size 0.15 \
  --val-size 0.15
```

### Models Trained

- **Logistic Regression**: Simple baseline, fast and interpretable
- **XGBoost**: High performance, handles non-linear patterns
- **LightGBM**: Fast training, similar performance to XGBoost

### Evaluation Metrics

- **ROC-AUC**: Ability to distinguish between classes
- **Log Loss**: Probability prediction quality
- **PR-AUC**: Better for imbalanced data

## 📚 Documentation

- **[TUTORIAL_PLAN.md](TUTORIAL_PLAN.md)**: Complete tutorial plan covering all phases
- **[Walkthrough Tutorials](walkthrough/)**: Detailed step-by-step guides for each phase
  - [Phase 1: Data Preparation & Exploration](walkthrough/PHASE_1.md) ✅
  - [Phase 2: Feature Engineering](walkthrough/PHASE_2.md) ✅
  - [Phase 3: Baseline Model Development](walkthrough/PHASE_3.md) ✅
  - [Phase 6: API Development](walkthrough/PHASE_6.md) ✅
  - [Phase 6.2: Streamlit UI with LLM Explanations](walkthrough/PHASE_6_2.md) ✅
  - [Phase 7: Containerization](walkthrough/PHASE_7.md) ✅
  - [Phase 8: Cloud Deployment (GCP)](walkthrough/PHASE_8.md) ✅
  - [Phase 8.2: Vertex AI & BigQuery Integration](walkthrough/PHASE_8_2.md) ✅

## 📋 Phase 8: Cloud Deployment (GCP)

### Prerequisites

1. **GCP Account**: Create account at [cloud.google.com](https://cloud.google.com)
2. **Install gcloud CLI**: [Installation Guide](https://cloud.google.com/sdk/docs/install)
3. **Authenticate**:
   ```bash
   gcloud auth login
   gcloud auth configure-docker
   ```

### Deploy to GCP

**Set Environment Variables:**
```bash
# Linux/Mac
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-central1

# Windows PowerShell
$env:GCP_PROJECT_ID = "your-project-id"
$env:GCP_REGION = "us-central1"
```

**Deploy:**
```bash
# Linux/Mac
chmod +x scripts/gcp_deploy.sh
./scripts/gcp_deploy.sh

# Windows PowerShell
.\scripts\gcp_deploy.ps1
```

**What Gets Deployed:**
- ✅ Cloud Run API service
- ✅ Cloud Run Streamlit UI
- ✅ Cloud Storage bucket (for models)
- ✅ Artifact Registry (for Docker images)

**All resources are labeled for easy cleanup!**

### Clean Up Resources

```bash
# Linux/Mac
chmod +x scripts/gcp_cleanup.sh
./scripts/gcp_cleanup.sh

# Windows PowerShell
.\scripts\gcp_cleanup.ps1
```

This deletes all resources created by the deployment to save costs.

## 📋 Phase 8.2: Vertex AI & BigQuery Integration

### Prerequisites

1. **Enable APIs**:
   ```bash
   gcloud services enable aiplatform.googleapis.com
   gcloud services enable bigquery.googleapis.com
   ```

### Upload Models to Vertex AI

```bash
# Linux/Mac
chmod +x scripts/vertex_ai_upload.sh
./scripts/vertex_ai_upload.sh

# Windows PowerShell
.\scripts\vertex_ai_upload.ps1
```

### Set Up BigQuery

```bash
# Linux/Mac
chmod +x scripts/bigquery_setup.sh
./scripts/bigquery_setup.sh

# Windows PowerShell
.\scripts\bigquery_setup.ps1
```

**What Gets Created:**
- ✅ BigQuery dataset: `ctr_predictions`
- ✅ Predictions table (partitioned by timestamp)
- ✅ Training data table (optional)

**API Integration:**
- Predictions are automatically logged to BigQuery
- Analytics queries available for model performance tracking

### Access Services

After deployment, you'll get:
- **API URL**: `https://ctr-prediction-api-<hash>-uc.a.run.app`
- **Streamlit UI**: `https://ctr-prediction-ui-<hash>-uc.a.run.app`

## 🛠 Technology Stack

- **Python 3.9+**
- **Pandas, NumPy**: Data manipulation
- **Matplotlib, Seaborn**: Visualization
- **Scikit-learn, XGBoost, LightGBM**: Machine learning
- **FastAPI**: API framework (Phase 6+)
- **Docker**: Containerization (Phase 7+)
- **GCP Cloud Run**: Serverless container deployment (Phase 8+)
- **GCP Cloud Storage**: Model artifact storage (Phase 8+)
- **GCP Vertex AI**: Model registry and training pipelines (Phase 8.2+)
- **GCP BigQuery**: Data warehouse and analytics (Phase 8.2+)

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.