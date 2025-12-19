# Phase 6.2: Streamlit UI with LLM Explanations

## 📋 Overview

Phase 6.2 extends Phase 6 by adding an interactive Streamlit web application that provides a user-friendly interface for CTR predictions, along with AI-powered explanations using Large Language Models (LLMs).

**Goals:**
- Create an interactive Streamlit UI for CTR predictions
- Integrate LLM-based explanations for predictions
- Provide batch prediction capabilities
- Enable model comparison visualization
- Make predictions accessible to non-technical users

**Deliverable:** Working Streamlit application with LLM explanations

---

## 🎯 Learning Objectives

By the end of this phase, you'll understand:
- How to build interactive web UIs with Streamlit
- How to integrate LLMs for generating explanations
- How to create visualizations for ML predictions
- How to make ML models accessible to end users
- Best practices for ML explainability

---

## 📦 Prerequisites

Before starting Phase 6.2, ensure you have:
- ✅ Completed Phase 6 (API is running)
- ✅ Streamlit installed (`pip install streamlit plotly`)
- ✅ (Optional) OpenAI API key for LLM explanations

---

## 🚀 Step-by-Step Guide

### Step 1: Install Dependencies

```bash
# Install Streamlit and visualization libraries
pip install streamlit plotly openai

# Or install all requirements
pip install -r requirements.txt
```

### Step 2: Set Up OpenAI API Key (Optional)

For LLM-powered explanations, you'll need an OpenAI API key:

**Option A: Environment Variable**
```bash
# Windows
set OPENAI_API_KEY=your_api_key_here

# Linux/Mac
export OPENAI_API_KEY=your_api_key_here
```

**Option B: In Streamlit App**
- Enter API key in the sidebar when running the app
- The app will use rule-based explanations if no key is provided

### Step 3: Start the API Server

Make sure the FastAPI server is running:

```bash
# In one terminal
python scripts/run_api.py
```

The API should be running on `http://localhost:8000`

### Step 4: Start the Streamlit App

```bash
# In another terminal
python scripts/run_streamlit.py

# Or directly with Streamlit
streamlit run src/app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Step 5: Use the Single Prediction Tab

1. **Enter Impression Details:**
   - User ID
   - Ad ID
   - Device type (mobile, desktop, tablet)
   - Ad placement (header, sidebar, footer, etc.)
   - Time information (hour, day of week)

2. **Click "Predict CTR"**

3. **View Results:**
   - Predicted CTR percentage
   - Model used
   - AI-generated explanation
   - Visual gauge chart

**Example Input:**
```
User ID: user_123
Ad ID: ad_456
Device: mobile
Placement: header
Hour: 14
Day of Week: 0 (Monday)
```

**Example Output:**
```
Predicted CTR: 2.34%
Model: XGBOOST

💡 Explanation:
This ad impression has a predicted CTR of 2.34%. Mobile devices 
typically show higher engagement. Header placements often receive 
more visibility. Business hours typically show higher engagement. 
This is a moderate CTR, indicating reasonable user interest.
```

### Step 6: Use Batch Prediction Tab

**Option A: CSV Upload**
1. Prepare a CSV file with columns:
   - `user_id`, `ad_id`, `device`, `placement`, `hour`, `day_of_week`
2. Upload the file
3. Click "Predict Batch"
4. View results table and download CSV

**Option B: Manual Entry**
1. Select number of impressions (1-10)
2. Enter details for each impression
3. Click "Predict Batch"
4. View individual predictions

### Step 7: Use Model Comparison Tab

1. Enter impression details
2. Click "Compare Models"
3. View predictions from all models:
   - Logistic Regression
   - XGBoost
   - LightGBM
4. See comparison chart and explanation

---

## 🔍 Understanding the Components

### 1. LLM Explainer (`src/app/llm_explainer.py`)

Generates human-readable explanations for predictions:

**Features:**
- **LLM Mode**: Uses OpenAI GPT models for natural language explanations
- **Rule-Based Mode**: Falls back to rule-based explanations if no API key
- **Comparison Explanations**: Explains differences between models

**LLM Explanation Example:**
```
This ad impression received a CTR prediction of 2.34% because:

1. Mobile devices typically show higher engagement rates due to 
   increased user attention and touch interactions.

2. Header placements receive prime visibility, appearing at the top 
   of the page where users naturally look first.

3. The business hours timing (2 PM) aligns with peak user activity 
   periods, increasing the likelihood of engagement.

The XGBoost model identified these factors as key contributors to 
the prediction, with device type and placement being the most 
influential features.
```

**Rule-Based Explanation Example:**
```
This ad impression has a predicted CTR of 2.34%. Mobile devices 
typically show higher engagement. Header placements often receive 
more visibility. Business hours typically show higher engagement. 
This is a moderate CTR, indicating reasonable user interest.
```

### 2. Streamlit App (`src/app/streamlit_app.py`)

Main application with three tabs:

**Tab 1: Single Prediction**
- Input form for single impression
- Real-time prediction
- Visual gauge chart
- LLM explanation

**Tab 2: Batch Prediction**
- CSV upload or manual entry
- Batch processing
- Results table and download
- Distribution histogram

**Tab 3: Model Comparison**
- Compare all models
- Side-by-side comparison
- Bar chart visualization
- Comparison explanation

### 3. Visualizations

**Gauge Chart:**
- Shows CTR as a gauge meter
- Color-coded thresholds
- Easy to interpret

**Bar Chart:**
- Model comparison visualization
- Side-by-side CTR values

**Histogram:**
- Distribution of batch predictions
- Shows prediction spread

---

## 🤖 LLM Integration Details

### How It Works

1. **Prediction Made**: API returns CTR prediction
2. **Context Gathered**: Request data, model info, feature importance
3. **Prompt Built**: Creates structured prompt for LLM
4. **LLM Called**: Sends prompt to OpenAI API
5. **Explanation Generated**: Returns human-readable explanation

### Prompt Structure

```
Explain why this ad impression received a CTR prediction of 2.34%.

Context:
- User ID: user_123
- Ad ID: ad_456
- Device: mobile
- Placement: header
- Hour: 14
- Day of Week: 0
- Model Used: xgboost

Top contributing factors:
- device_mobile: 0.234
- placement_header: 0.189
- hour_14: 0.156

Provide a brief, user-friendly explanation (2-3 sentences).
```

### Cost Considerations

- **GPT-3.5-turbo**: ~$0.002 per explanation (very affordable)
- **GPT-4**: ~$0.03 per explanation (higher quality)
- **Rule-based**: Free (always available as fallback)

### Fallback Behavior

If LLM is unavailable:
- Automatically uses rule-based explanations
- No errors or interruptions
- Still provides useful insights

---

## 🎨 UI Features

### Sidebar Configuration

- **API URL**: Configure API endpoint
- **LLM Settings**: Enable/disable LLM, enter API key
- **Model Selection**: Choose default model
- **Quick Links**: Health check, API docs

### Main Interface

- **Clean Design**: Modern, professional UI
- **Responsive Layout**: Works on different screen sizes
- **Color-Coded Results**: Visual indicators for CTR levels
- **Interactive Charts**: Plotly visualizations

### User Experience

- **Real-time Feedback**: Loading spinners, progress indicators
- **Error Handling**: Clear error messages
- **Helpful Tooltips**: Contextual help text
- **Download Options**: Export results as CSV

---

## 📊 Example Use Cases

### Use Case 1: Marketing Team

**Scenario**: Marketing team wants to understand why an ad has low CTR

**Workflow:**
1. Enter ad details in Single Prediction tab
2. Get prediction and explanation
3. Understand factors affecting CTR
4. Adjust targeting or placement based on insights

**Value**: Non-technical users can understand ML predictions

### Use Case 2: Ad Operations

**Scenario**: Ad ops team needs to evaluate multiple ad placements

**Workflow:**
1. Upload CSV with multiple impressions
2. Get batch predictions
3. Analyze distribution
4. Download results for reporting

**Value**: Efficient batch processing with visualizations

### Use Case 3: Model Evaluation

**Scenario**: Data scientist wants to compare model performance

**Workflow:**
1. Enter test impression
2. Compare all models
3. See side-by-side predictions
4. Understand model differences

**Value**: Quick model comparison and selection

---

## 🔧 Configuration Options

### Environment Variables

```bash
# OpenAI API Key (optional)
export OPENAI_API_KEY=your_key_here

# API URL (default: http://localhost:8000)
export API_URL=http://localhost:8000

# Streamlit port (default: 8501)
export STREAMLIT_PORT=8501
```

### Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "localhost"

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
```

---

## 🧪 Testing the Application

### Test Single Prediction

1. Start API: `python scripts/run_api.py`
2. Start Streamlit: `python scripts/run_streamlit.py`
3. Enter test data
4. Verify prediction appears
5. Check explanation is generated

### Test Batch Prediction

1. Create test CSV:
```csv
user_id,ad_id,device,placement,hour,day_of_week
user_1,ad_1,mobile,header,14,0
user_2,ad_2,desktop,sidebar,18,5
```

2. Upload and verify results

### Test Model Comparison

1. Enter impression details
2. Verify all models return predictions
3. Check comparison chart
4. Verify explanation

---

## 🔍 Common Issues & Solutions

### Issue 1: API Connection Error

**Problem**: "Cannot connect to API"

**Solution:**
```bash
# Make sure API is running
python scripts/run_api.py

# Check API URL in sidebar
# Default: http://localhost:8000
```

### Issue 2: LLM Not Working

**Problem**: No LLM explanations, only rule-based

**Solutions:**
- Check API key is set correctly
- Verify OpenAI account has credits
- Check internet connection
- Rule-based explanations still work!

### Issue 3: Import Errors

**Problem**: ModuleNotFoundError

**Solution:**
```bash
# Install missing dependencies
pip install streamlit plotly openai requests pandas
```

### Issue 4: Port Already in Use

**Problem**: Port 8501 already in use

**Solution:**
```bash
# Use different port
streamlit run src/app/streamlit_app.py --server.port 8502

# Or kill existing process
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8501 | xargs kill
```

---

## 📊 Expected Results

After completing Phase 6.2, you should have:

1. **Working Streamlit App:**
   - Accessible at `http://localhost:8501`
   - All three tabs functional
   - Visualizations working

2. **LLM Explanations:**
   - Natural language explanations
   - Context-aware insights
   - Fallback to rule-based if needed

3. **User-Friendly Interface:**
   - Easy to use for non-technical users
   - Clear visualizations
   - Helpful error messages

---

## ✅ Phase 6.2 Checklist

- [ ] Streamlit app starts successfully
- [ ] Single prediction tab works
- [ ] Batch prediction tab works (CSV upload)
- [ ] Model comparison tab works
- [ ] LLM explanations generate (with API key)
- [ ] Rule-based explanations work (without API key)
- [ ] Visualizations display correctly
- [ ] CSV download works
- [ ] Error handling works properly
- [ ] Ready for production use

---

## 🚀 Next Steps

Once you've completed Phase 6.2, you can:

1. **Deploy Streamlit App:**
   - Streamlit Cloud (free hosting)
   - Docker containerization
   - Cloud platforms (GCP, AWS)

2. **Enhance Explanations:**
   - Add feature importance visualization
   - Include SHAP values
   - Add counterfactual explanations

3. **Add Features:**
   - User authentication
   - Prediction history
   - A/B testing interface
   - Analytics dashboard

---

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [ML Explainability Guide](https://christophm.github.io/interpretable-ml-book/)
- [Streamlit Cloud](https://streamlit.io/cloud)

---

## 💡 Tips for Success

1. **Start Simple**: Test with rule-based explanations first
2. **Add LLM Later**: Get basic functionality working, then add LLM
3. **Test Thoroughly**: Try different inputs and edge cases
4. **Monitor Costs**: Track LLM API usage if using OpenAI
5. **User Feedback**: Get feedback from non-technical users
6. **Iterate**: Improve explanations based on user needs

---

## 🎓 Key Concepts Learned

### 1. ML Explainability

- **Why it matters**: Users need to understand predictions
- **LLM approach**: Natural language explanations
- **Rule-based approach**: Structured, deterministic explanations
- **Hybrid approach**: Combine both for best results

### 2. User Interface Design

- **Accessibility**: Make ML accessible to non-technical users
- **Visualization**: Charts help users understand data
- **Feedback**: Clear error messages and loading states
- **Flexibility**: Multiple input methods (forms, CSV upload)

### 3. LLM Integration

- **API Integration**: Connect to external LLM services
- **Prompt Engineering**: Structure prompts for good results
- **Error Handling**: Graceful fallbacks
- **Cost Management**: Monitor and optimize API usage

---

**Congratulations on completing Phase 6.2!** 🎉

You now have a complete, user-friendly interface for CTR predictions with AI-powered explanations, making your ML models accessible to everyone!

