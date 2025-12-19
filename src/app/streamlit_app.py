"""
Streamlit Application for CTR Prediction with LLM Explanations

Provides an interactive UI for making CTR predictions and getting
AI-powered explanations of the results.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.app.llm_explainer import LLMExplainer

# Page configuration
st.set_page_config(
    page_title="CTR Prediction System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
# Check for API_URL environment variable (set by Docker Compose)
default_api_url = os.getenv("API_URL", "http://localhost:8000")
if 'api_url' not in st.session_state:
    st.session_state.api_url = default_api_url

if 'explainer' not in st.session_state:
    st.session_state.explainer = LLMExplainer()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .explanation-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2ecc71;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🎯 CTR Prediction System</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API URL configuration
    # Always check environment variable first (for Docker), then fall back to session state
    env_api_url = os.getenv("API_URL")
    if env_api_url:
        # Environment variable takes precedence (Docker mode)
        default_url = env_api_url
        if st.session_state.api_url != env_api_url:
            st.session_state.api_url = env_api_url
    else:
        # Local development mode
        default_url = st.session_state.api_url
    
    api_url = st.text_input(
        "API URL",
        value=default_url,
        help="URL of the CTR Prediction API (use 'http://api:8000' in Docker, 'http://localhost:8000' locally)"
    )
    # Update session state, but respect environment variable if set
    if not env_api_url:
        st.session_state.api_url = api_url
    else:
        # In Docker, use env var but allow override for testing
        if api_url != env_api_url:
            st.warning(f"⚠️ Using custom URL. Default (Docker): {env_api_url}")
        st.session_state.api_url = api_url
    
    # LLM Configuration
    st.subheader("🤖 LLM Explanation")
    use_llm = st.checkbox(
        "Enable LLM Explanations",
        value=os.getenv("OPENAI_API_KEY") is not None,
        help="Requires OpenAI API key (set OPENAI_API_KEY environment variable)"
    )
    
    if use_llm:
        api_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Your OpenAI API key"
        )
        if api_key:
            st.session_state.explainer = LLMExplainer(api_key=api_key)
            st.success("✓ LLM enabled")
        else:
            st.warning("⚠️ Enter API key to enable LLM explanations")
    else:
        st.info("Using rule-based explanations")
    
    st.markdown("---")
    
    # Model selection
    st.subheader("📊 Model Selection")
    model_choice = st.selectbox(
        "Select Model",
        options=["xgboost", "lightgbm", "logistic"],
        index=0,
        help="Choose which model to use for predictions"
    )
    
    st.markdown("---")
    
    # API Status Indicator
    st.subheader("🔗 API Status")
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            st.success("✓ API is healthy")
            # Try to get model info
            try:
                model_response = requests.get(f"{api_url}/model_info", timeout=5)
                if model_response.status_code == 200:
                    model_info = model_response.json()
                    st.caption(f"Model: {model_info.get('model_name', 'unknown').upper()} | Features: {model_info.get('num_features', 0)}")
            except:
                pass
        else:
            st.error("✗ API is not responding")
    except requests.exceptions.ConnectionError:
        st.error("✗ Cannot connect to API")
        st.caption(f"Trying to connect to: {api_url}")
    except Exception as e:
        st.warning(f"⚠️ API check failed: {str(e)}")
    
    st.markdown("---")
    
    # Quick links
    st.subheader("🔗 Quick Links")
    if st.button("📖 API Documentation"):
        st.markdown(f"[Open Swagger UI]({api_url}/docs)")
    if st.button("🔄 Refresh API Status"):
        st.rerun()

# Main content
tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📊 Batch Prediction", "📈 Model Comparison"])

# Tab 1: Single Prediction
with tab1:
    st.header("Single CTR Prediction")
    st.markdown("Enter ad impression details to get a CTR prediction with AI explanation.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Input Details")
        
        user_id = st.text_input("User ID", value="user_123", help="Unique user identifier")
        ad_id = st.text_input("Ad ID", value="ad_456", help="Unique advertisement identifier")
        
        device = st.selectbox(
            "Device Type",
            options=["mobile", "desktop", "tablet"],
            index=0
        )
        
        placement = st.selectbox(
            "Ad Placement",
            options=["header", "sidebar", "footer", "in_content", "popup"],
            index=0
        )
        
        col_time1, col_time2 = st.columns(2)
        with col_time1:
            hour = st.number_input("Hour", min_value=0, max_value=23, value=datetime.now().hour)
        with col_time2:
            day_of_week = st.number_input("Day of Week", min_value=0, max_value=6, value=datetime.now().weekday())
        
        # Use date_input and time_input instead of datetime_input for compatibility
        col_date, col_time = st.columns(2)
        with col_date:
            date_input = st.date_input(
                "Date",
                value=datetime.now().date(),
                help="Date of the impression"
            )
        with col_time:
            time_input = st.time_input(
                "Time",
                value=datetime.now().time(),
                help="Time of the impression"
            )
        
        # Combine date and time into datetime
        timestamp = datetime.combine(date_input, time_input)
    
    with col2:
        st.subheader("🎯 Prediction")
        
        if st.button("🔮 Predict CTR", type="primary", use_container_width=True):
            with st.spinner("Making prediction..."):
                try:
                    # Prepare request
                    request_data = {
                        "user_id": user_id,
                        "ad_id": ad_id,
                        "device": device,
                        "placement": placement,
                        "hour": int(hour),
                        "day_of_week": int(day_of_week),
                        "timestamp": timestamp.isoformat()
                    }
                    
                    # Make API call
                    response = requests.post(
                        f"{api_url}/predict_ctr",
                        json=request_data,
                        params={"model_name": model_choice},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        prediction = response.json()
                        ctr = prediction.get('predicted_ctr', 0)
                        model_name = prediction.get('model_name', 'unknown')
                        
                        # Display prediction
                        st.markdown(f'<div class="prediction-box">', unsafe_allow_html=True)
                        st.metric("Predicted CTR", f"{ctr:.2%}")
                        st.caption(f"Model: {model_name.upper()}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Generate explanation
                        with st.spinner("Generating explanation..."):
                            explanation = st.session_state.explainer.explain_prediction(
                                prediction,
                                request_data
                            )
                        
                        st.markdown(f'<div class="explanation-box">', unsafe_allow_html=True)
                        st.subheader("💡 Explanation")
                        st.markdown(explanation)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Visualization
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=ctr * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "CTR (%)"},
                            gauge={
                                'axis': {'range': [None, 5]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 1], 'color': "lightgray"},
                                    {'range': [1, 2], 'color': "gray"},
                                    {'range': [2, 5], 'color': "lightblue"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 2
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure the API server is running.")
                    st.info(f"Start the API with: `python scripts/run_api.py`")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Tab 2: Batch Prediction
with tab2:
    st.header("Batch CTR Prediction")
    st.markdown("Upload a CSV file or enter multiple impressions to get batch predictions.")
    
    input_method = st.radio(
        "Input Method",
        options=["CSV Upload", "Manual Entry"],
        horizontal=True
    )
    
    if input_method == "CSV Upload":
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV file with columns: user_id, ad_id, device, placement, hour, day_of_week"
        )
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(10))
            
            if st.button("🔮 Predict Batch", type="primary"):
                with st.spinner("Processing batch predictions..."):
                    try:
                        # Convert to API format
                        impressions = df.to_dict('records')
                        
                        # Make API call
                        response = requests.post(
                            f"{api_url}/batch_predict",
                            json={"impressions": impressions},
                            params={"model_name": model_choice},
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            predictions = result.get('predictions', [])
                            processing_time = result.get('processing_time_ms', 0)
                            
                            # Display results
                            st.success(f"✓ Processed {len(predictions)} predictions in {processing_time:.2f}ms")
                            
                            # Create results DataFrame
                            results_df = pd.DataFrame(predictions)
                            results_df['predicted_ctr'] = results_df['predicted_ctr'].apply(lambda x: f"{x:.2%}")
                            
                            st.dataframe(results_df)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name=f"ctr_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                            # Visualization
                            ctr_values = [p['predicted_ctr'] for p in predictions]
                            fig = px.histogram(
                                x=ctr_values,
                                nbins=20,
                                title="CTR Distribution",
                                labels={'x': 'Predicted CTR', 'y': 'Count'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    else:  # Manual Entry
        st.subheader("Enter Impressions")
        num_impressions = st.number_input("Number of Impressions", min_value=1, max_value=10, value=2)
        
        impressions = []
        for i in range(num_impressions):
            with st.expander(f"Impression {i+1}"):
                col1, col2 = st.columns(2)
                with col1:
                    user_id = st.text_input(f"User ID {i+1}", value=f"user_{i+1}", key=f"user_{i}")
                    ad_id = st.text_input(f"Ad ID {i+1}", value=f"ad_{i+1}", key=f"ad_{i}")
                with col2:
                    device = st.selectbox(f"Device {i+1}", ["mobile", "desktop", "tablet"], key=f"device_{i}")
                    placement = st.selectbox(f"Placement {i+1}", ["header", "sidebar", "footer", "in_content", "popup"], key=f"placement_{i}")
                
                impressions.append({
                    "user_id": user_id,
                    "ad_id": ad_id,
                    "device": device,
                    "placement": placement
                })
        
        if st.button("🔮 Predict Batch", type="primary"):
            with st.spinner("Processing batch predictions..."):
                try:
                    response = requests.post(
                        f"{api_url}/batch_predict",
                        json={"impressions": impressions},
                        params={"model_name": model_choice},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        predictions = result.get('predictions', [])
                        
                        st.success(f"✓ Processed {len(predictions)} predictions")
                        
                        # Display results
                        for i, pred in enumerate(predictions):
                            with st.expander(f"Prediction {i+1}: {pred['predicted_ctr']:.2%}"):
                                st.json(pred)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Tab 3: Model Comparison
with tab3:
    st.header("Model Comparison")
    st.markdown("Compare predictions from different models for the same impression.")
    
    st.subheader("📝 Input Details")
    
    col1, col2 = st.columns(2)
    with col1:
        user_id = st.text_input("User ID", value="user_123", key="comp_user")
        ad_id = st.text_input("Ad ID", value="ad_456", key="comp_ad")
        device = st.selectbox("Device", ["mobile", "desktop", "tablet"], key="comp_device")
    with col2:
        placement = st.selectbox("Placement", ["header", "sidebar", "footer", "in_content", "popup"], key="comp_placement")
        hour = st.number_input("Hour", min_value=0, max_value=23, value=14, key="comp_hour")
        day_of_week = st.number_input("Day of Week", min_value=0, max_value=6, value=0, key="comp_dow")
    
    if st.button("🔍 Compare Models", type="primary"):
        with st.spinner("Comparing models..."):
            try:
                request_data = {
                    "user_id": user_id,
                    "ad_id": ad_id,
                    "device": device,
                    "placement": placement,
                    "hour": int(hour),
                    "day_of_week": int(day_of_week)
                }
                
                # Get predictions from all models
                models = ["logistic", "xgboost", "lightgbm"]
                predictions = {}
                
                for model in models:
                    try:
                        response = requests.post(
                            f"{api_url}/predict_ctr",
                            json=request_data,
                            params={"model_name": model},
                            timeout=10
                        )
                        if response.status_code == 200:
                            predictions[model] = response.json()
                    except:
                        pass
                
                if predictions:
                    # Display comparison
                    st.subheader("📊 Comparison Results")
                    
                    comparison_data = {
                        "Model": [],
                        "Predicted CTR": [],
                        "Probability": []
                    }
                    
                    for model, pred in predictions.items():
                        comparison_data["Model"].append(model.upper())
                        comparison_data["Predicted CTR"].append(pred['predicted_ctr'])
                        comparison_data["Probability"].append(pred['probability'])
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df['Predicted CTR'] = comparison_df['Predicted CTR'].apply(lambda x: f"{x:.2%}")
                    
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Visualization
                    fig = px.bar(
                        comparison_df,
                        x="Model",
                        y=[float(x.replace('%', '')) for x in comparison_df['Predicted CTR']],
                        title="Model Comparison",
                        labels={'y': 'Predicted CTR (%)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Generate comparison explanation
                    explanation = st.session_state.explainer.explain_comparison(predictions, request_data)
                    st.markdown(f'<div class="explanation-box">', unsafe_allow_html=True)
                    st.subheader("💡 Comparison Explanation")
                    st.markdown(explanation)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("No models available for comparison")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>CTR Prediction System | Built with Streamlit & FastAPI</div>",
    unsafe_allow_html=True
)

