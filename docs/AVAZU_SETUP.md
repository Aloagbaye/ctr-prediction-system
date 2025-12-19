# Avazu Dataset Setup Guide

This guide explains how to download and use the real Avazu CTR prediction dataset from Kaggle.

## 📥 Prerequisites

1. **Kaggle Account**: Sign up at https://www.kaggle.com (free)
2. **Kaggle API Token**: Required for downloading datasets

## 🔑 Setting Up Kaggle API

### Step 1: Get Your API Token

1. Go to https://www.kaggle.com/account
2. Scroll down to the "API" section
3. Click "Create New API Token"
4. This downloads a file called `kaggle.json`

### Step 2: Place the Token

**On Windows:**
```
C:\Users\<your-username>\.kaggle\kaggle.json
```

**On Linux/Mac:**
```
~/.kaggle/kaggle.json
```

### Step 3: Set Permissions (Linux/Mac only)

```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Step 4: Verify Installation

```bash
pip install kaggle
kaggle --version
```

## 📊 Downloading the Dataset

### Method 1: Using the Script (Recommended)

```bash
# Download the dataset
python scripts/download_avazu.py --download
```

This will:
- Download `train.gz` and `test.gz` from Kaggle
- Extract them to `data/raw/avazu/`
- The dataset is ~1.5GB compressed, ~6GB uncompressed

### Method 2: Manual Download

If the script doesn't work:

1. Go to https://www.kaggle.com/c/avazu-ctr-prediction/data
2. Download:
   - `train.gz` (~1.5GB)
   - `test.gz` (~500MB)
3. Place them in `data/raw/avazu/`

## 📈 Loading the Dataset

The Avazu dataset is **very large** (~40 million rows). For most purposes, you'll want to sample it:

### Load a Sample (Recommended)

```bash
# Load 10% of the data (good for development)
python scripts/download_avazu.py --load --sample-frac 0.1 --output data/raw/avazu_processed.csv

# Load 1% of the data (for quick testing)
python scripts/download_avazu.py --load --sample-frac 0.01 --output data/raw/avazu_processed.csv
```

### Load Specific Number of Rows

```bash
# Load first 100,000 rows
python scripts/download_avazu.py --load --n-rows 100000 --output data/raw/avazu_processed.csv

# Load first 1,000,000 rows
python scripts/download_avazu.py --load --n-rows 1000000 --output data/raw/avazu_processed.csv
```

### Load Full Dataset (⚠️ Requires Significant Memory)

```bash
# This will load all ~40 million rows - use with caution!
python scripts/download_avazu.py --load --output data/raw/avazu_processed.parquet
```

**Note**: For large datasets, the output will be saved as Parquet format (more efficient than CSV).

## 🔍 Understanding the Avazu Dataset

### Dataset Structure

The Avazu dataset contains:

- **~40 million training samples**
- **Target**: `click` (0 or 1)
- **Features**:
  - `hour`: Timestamp in YYMMDDHH format
  - `C1-C21`: Categorical features (anonymized)
  - `banner_pos`: Banner position
  - `site_id`, `site_domain`, `site_category`: Site information
  - `app_id`, `app_domain`, `app_category`: App information
  - `device_id`, `device_ip`, `device_model`, `device_type`, `device_conn_type`: Device information

### Data Conversion

The loader automatically converts Avazu format to our standard format:

- `click` → `clicked`
- `hour` → `timestamp`, `hour`, `day_of_week`, `is_weekend`, `date`
- `device_type` → `device` (mobile, desktop, tablet)
- `banner_pos` → `placement`
- Creates `user_id` and `ad_id` from available features

## 📊 Running EDA on Avazu Data

```bash
# After loading and processing
python scripts/explore_data.py --input data/raw/avazu_processed.csv --output-dir data/eda/avazu
```

## 💡 Tips

1. **Start Small**: Use `--sample-frac 0.01` or `--n-rows 100000` for initial exploration
2. **Memory Management**: Large datasets require significant RAM. Consider:
   - Using Parquet format (more efficient)
   - Processing in chunks
   - Using a machine with 16GB+ RAM for full dataset
3. **Processing Time**: Loading the full dataset can take 10-30 minutes
4. **Storage**: Ensure you have at least 10GB free space

## 🔄 Workflow Example

```bash
# 1. Download dataset (one-time)
python scripts/download_avazu.py --download

# 2. Load sample for development
python scripts/download_avazu.py --load --sample-frac 0.1 --output data/raw/avazu_sample.csv

# 3. Explore the sample
python scripts/explore_data.py --input data/raw/avazu_sample.csv --output-dir data/eda/avazu_sample

# 4. When ready, load larger sample or full dataset
python scripts/download_avazu.py --load --sample-frac 0.5 --output data/raw/avazu_50pct.csv
```

## ❓ Troubleshooting

### "Kaggle API credentials not found"
- Make sure `kaggle.json` is in the correct location
- Check file permissions (Linux/Mac: `chmod 600`)

### "Kaggle CLI not found"
```bash
pip install kaggle
```

### "Out of memory" error
- Use `--sample-frac` or `--n-rows` to load less data
- Close other applications
- Consider using a machine with more RAM

### "File not found" error
- Make sure you've downloaded the dataset first
- Check that files are in `data/raw/avazu/`

### Download is slow
- The dataset is large (~1.5GB). This is normal.
- Check your internet connection
- Consider downloading manually from Kaggle website

## 📚 Additional Resources

- [Kaggle Avazu Competition](https://www.kaggle.com/c/avazu-ctr-prediction)
- [Kaggle API Documentation](https://www.kaggle.com/docs/api)
- [Avazu Dataset Description](https://www.kaggle.com/c/avazu-ctr-prediction/data)

