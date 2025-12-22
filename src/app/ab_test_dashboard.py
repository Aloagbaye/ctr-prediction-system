"""
Streamlit Dashboard for A/B Test Visualization

Displays A/B test results with:
- Test overview and status
- Model comparison metrics
- Statistical analysis
- Visualizations
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, Any, Optional
import os

# Configuration
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")


def check_api_health() -> bool:
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_ab_tests() -> list:
    """Get list of all A/B tests"""
    try:
        response = requests.get(f"{API_BASE_URL}/ab_test/list", timeout=5)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Error fetching A/B tests: {e}")
        return []


def get_test_stats(test_id: str) -> Optional[Dict[str, Any]]:
    """Get statistics for a specific A/B test"""
    try:
        response = requests.get(f"{API_BASE_URL}/ab_test/stats/{test_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching test stats: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching test stats: {e}")
        return None


def calculate_duration(start_time: str, end_time: Optional[str] = None) -> str:
    """Calculate test duration"""
    try:
        start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_time.replace('Z', '+00:00')) if end_time else datetime.now()
        
        delta = end - start
        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        if days > 0:
            return f"{days} day{'s' if days != 1 else ''}, {hours} hour{'s' if hours != 1 else ''}"
        elif hours > 0:
            return f"{hours} hour{'s' if hours != 1 else ''}, {minutes} minute{'s' if minutes != 1 else ''}"
        else:
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
    except:
        return "Unknown"


def format_number(num: float, decimals: int = 2) -> str:
    """Format number with commas"""
    if num is None:
        return "N/A"
    return f"{num:,.{decimals}f}"


def create_comparison_table(stats: Dict[str, Any]) -> pd.DataFrame:
    """Create comparison table DataFrame"""
    config = stats.get('config', {})
    model_a_stats = stats.get('model_a_stats', {})
    model_b_stats = stats.get('model_b_stats', {})
    comparison = stats.get('comparison', {})
    
    data = {
        'Metric': [
            'Total Requests',
            'Total Predictions',
            'Avg Latency (ms)',
            'P95 Latency (ms)',
            'Min Latency (ms)',
            'Max Latency (ms)',
            'Error Count',
            'Error Rate (%)',
            'Success Rate (%)'
        ],
        'Model A': [
            format_number(model_a_stats.get('total_requests', 0), 0),
            format_number(model_a_stats.get('total_predictions', 0), 0),
            format_number(model_a_stats.get('avg_prediction_time_ms', 0), 2),
            format_number(model_a_stats.get('max_prediction_time_ms', 0), 2),  # Using max as proxy for P95
            format_number(model_a_stats.get('min_prediction_time_ms', 0), 2),
            format_number(model_a_stats.get('max_prediction_time_ms', 0), 2),
            format_number(model_a_stats.get('error_count', 0), 0),
            format_number(
                (model_a_stats.get('error_count', 0) / model_a_stats.get('total_requests', 1) * 100)
                if model_a_stats.get('total_requests', 0) > 0 else 0,
                2
            ),
            format_number(
                100 - (model_a_stats.get('error_count', 0) / model_a_stats.get('total_requests', 1) * 100)
                if model_a_stats.get('total_requests', 0) > 0 else 100,
                2
            )
        ],
        'Model B': [
            format_number(model_b_stats.get('total_requests', 0), 0),
            format_number(model_b_stats.get('total_predictions', 0), 0),
            format_number(model_b_stats.get('avg_prediction_time_ms', 0), 2),
            format_number(model_b_stats.get('max_prediction_time_ms', 0), 2),
            format_number(model_b_stats.get('min_prediction_time_ms', 0), 2),
            format_number(model_b_stats.get('max_prediction_time_ms', 0), 2),
            format_number(model_b_stats.get('error_count', 0), 0),
            format_number(
                (model_b_stats.get('error_count', 0) / model_b_stats.get('total_requests', 1) * 100)
                if model_b_stats.get('total_requests', 0) > 0 else 0,
                2
            ),
            format_number(
                100 - (model_b_stats.get('error_count', 0) / model_b_stats.get('total_requests', 1) * 100)
                if model_b_stats.get('total_requests', 0) > 0 else 100,
                2
            )
        ]
    }
    
    # Add improvement column if comparison data available
    if comparison:
        latency_comp = comparison.get('avg_latency_ms', {})
        error_comp = comparison.get('error_rate', {})
        
        improvements = [
            '',  # Total Requests
            '',  # Total Predictions
            f"{latency_comp.get('improvement_pct', 0):.2f}%" if latency_comp.get('improvement_pct') else '',
            '',  # P95 Latency
            '',  # Min Latency
            '',  # Max Latency
            '',  # Error Count
            f"{error_comp.get('model_a', 0) - error_comp.get('model_b', 0):.2f}%" if error_comp else '',
            ''   # Success Rate
        ]
        data['Improvement'] = improvements
    
    return pd.DataFrame(data)


def create_latency_comparison_chart(stats: Dict[str, Any]):
    """Create latency comparison bar chart"""
    model_a_stats = stats.get('model_a_stats', {})
    model_b_stats = stats.get('model_b_stats', {})
    config = stats.get('config', {})
    
    model_a_name = config.get('model_a', 'Model A')
    model_b_name = config.get('model_b', 'Model B')
    
    metrics = ['Avg Latency', 'Min Latency', 'Max Latency']
    model_a_values = [
        model_a_stats.get('avg_prediction_time_ms', 0),
        model_a_stats.get('min_prediction_time_ms', 0),
        model_a_stats.get('max_prediction_time_ms', 0)
    ]
    model_b_values = [
        model_b_stats.get('avg_prediction_time_ms', 0),
        model_b_stats.get('min_prediction_time_ms', 0),
        model_b_stats.get('max_prediction_time_ms', 0)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=model_a_name,
        x=metrics,
        y=model_a_values,
        marker_color='#1f77b4'
    ))
    
    fig.add_trace(go.Bar(
        name=model_b_name,
        x=metrics,
        y=model_b_values,
        marker_color='#ff7f0e'
    ))
    
    fig.update_layout(
        title='Latency Comparison',
        xaxis_title='Metric',
        yaxis_title='Latency (ms)',
        barmode='group',
        height=400
    )
    
    return fig


def create_error_rate_chart(stats: Dict[str, Any]):
    """Create error rate comparison chart"""
    model_a_stats = stats.get('model_a_stats', {})
    model_b_stats = stats.get('model_b_stats', {})
    config = stats.get('config', {})
    
    model_a_name = config.get('model_a', 'Model A')
    model_b_name = config.get('model_b', 'Model B')
    
    model_a_requests = model_a_stats.get('total_requests', 0)
    model_b_requests = model_b_stats.get('total_requests', 0)
    model_a_errors = model_a_stats.get('error_count', 0)
    model_b_errors = model_b_stats.get('error_count', 0)
    
    if model_a_requests == 0 and model_b_requests == 0:
        return None
    
    model_a_error_rate = (model_a_errors / model_a_requests * 100) if model_a_requests > 0 else 0
    model_b_error_rate = (model_b_errors / model_b_requests * 100) if model_b_requests > 0 else 0
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[model_a_name, model_b_name],
        y=[model_a_error_rate, model_b_error_rate],
        marker_color=['#1f77b4', '#ff7f0e'],
        text=[f"{model_a_error_rate:.2f}%", f"{model_b_error_rate:.2f}%"],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Error Rate Comparison',
        xaxis_title='Model',
        yaxis_title='Error Rate (%)',
        height=400
    )
    
    return fig


def create_traffic_distribution_chart(stats: Dict[str, Any]):
    """Create traffic distribution pie chart"""
    comparison = stats.get('comparison', {})
    traffic_split = comparison.get('traffic_split', {})
    total_requests = comparison.get('total_requests', {})
    
    if not traffic_split or not total_requests:
        return None
    
    model_a_pct = traffic_split.get('model_a', '0%').replace('%', '')
    model_b_pct = traffic_split.get('model_b', '0%').replace('%', '')
    config = stats.get('config', {})
    
    model_a_name = config.get('model_a', 'Model A')
    model_b_name = config.get('model_b', 'Model B')
    
    fig = go.Figure(data=[go.Pie(
        labels=[model_a_name, model_b_name],
        values=[float(model_a_pct), float(model_b_pct)],
        hole=0.4,
        marker_colors=['#1f77b4', '#ff7f0e']
    )])
    
    fig.update_layout(
        title='Traffic Distribution',
        height=400
    )
    
    return fig


def main():
    st.set_page_config(
        page_title="A/B Test Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 A/B Test Dashboard")
    st.markdown("Visualize and compare A/B test results for CTR prediction models")
    
    # Check API health
    if not check_api_health():
        st.error(f"⚠️ API is not running. Please start the API at {API_BASE_URL}")
        st.info("Start the API with: `python scripts/run_api.py`")
        return
    
    st.success("✓ API is running")
    
    # Sidebar for test selection
    st.sidebar.header("Test Selection")
    
    # Get list of tests
    tests = get_ab_tests()
    
    if not tests:
        st.warning("No A/B tests found. Create a test first using the API or test script.")
        st.info("Create a test with: `python scripts/test_ab_testing.py`")
        return
    
    # Test selector
    test_options = {f"{t.get('test_id', 'unknown')} - {t.get('description', 'No description')}": t.get('test_id') 
                    for t in tests}
    
    selected_test_label = st.sidebar.selectbox(
        "Select A/B Test",
        options=list(test_options.keys()),
        index=0
    )
    
    selected_test_id = test_options[selected_test_label]
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Get test statistics
    stats = get_test_stats(selected_test_id)
    
    if not stats:
        st.error("Could not fetch test statistics")
        return
    
    config = stats.get('config', {})
    model_a_stats = stats.get('model_a_stats', {})
    model_b_stats = stats.get('model_b_stats', {})
    comparison = stats.get('comparison', {})
    
    # Header section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "🟢 Running" if config.get('enabled', False) else "🔴 Stopped"
        st.metric("Status", status)
    
    with col2:
        duration = calculate_duration(
            config.get('start_time', ''),
            config.get('end_time')
        )
        st.metric("Duration", duration)
    
    with col3:
        total_requests = comparison.get('total_requests', {}).get('total', 0) if comparison else 0
        st.metric("Total Requests", f"{total_requests:,}")
    
    # Test information
    st.markdown("---")
    st.subheader("Test Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.write(f"**Test ID:** `{stats.get('test_id', 'N/A')}`")
        st.write(f"**Model A (Control):** `{config.get('model_a', 'N/A')}`")
        st.write(f"**Model B (Variant):** `{config.get('model_b', 'N/A')}`")
    
    with info_col2:
        traffic_split = comparison.get('traffic_split', {}) if comparison else {}
        st.write(f"**Traffic Split:** {traffic_split.get('model_a', 'N/A')} / {traffic_split.get('model_b', 'N/A')}")
        st.write(f"**Start Time:** {config.get('start_time', 'N/A')}")
        if config.get('end_time'):
            st.write(f"**End Time:** {config.get('end_time', 'N/A')}")
        if config.get('description'):
            st.write(f"**Description:** {config.get('description', 'N/A')}")
    
    # Main comparison table
    st.markdown("---")
    st.subheader("Model Comparison")
    
    comparison_df = create_comparison_table(stats)
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Visualizations
    st.markdown("---")
    st.subheader("Visualizations")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        latency_fig = create_latency_comparison_chart(stats)
        if latency_fig:
            st.plotly_chart(latency_fig, use_container_width=True)
        
        error_fig = create_error_rate_chart(stats)
        if error_fig:
            st.plotly_chart(error_fig, use_container_width=True)
    
    with viz_col2:
        traffic_fig = create_traffic_distribution_chart(stats)
        if traffic_fig:
            st.plotly_chart(traffic_fig, use_container_width=True)
        
        # Improvement metrics
        if comparison:
            latency_comp = comparison.get('avg_latency_ms', {})
            if latency_comp.get('improvement_pct'):
                improvement = latency_comp.get('improvement_pct', 0)
                st.metric(
                    "Latency Improvement",
                    f"{improvement:.2f}%",
                    delta=f"{latency_comp.get('difference_ms', 0):.2f} ms"
                )
    
    # Statistical summary
    st.markdown("---")
    st.subheader("Statistical Summary")
    
    if comparison:
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            latency_diff = comparison.get('avg_latency_ms', {}).get('difference_ms', 0)
            st.metric(
                "Latency Difference",
                f"{latency_diff:.2f} ms",
                delta=f"{comparison.get('avg_latency_ms', {}).get('improvement_pct', 0):.2f}% improvement"
                if latency_diff < 0 else None
            )
        
        with summary_col2:
            error_a = comparison.get('error_rate', {}).get('model_a', 0)
            error_b = comparison.get('error_rate', {}).get('model_b', 0)
            error_diff = error_a - error_b
            st.metric(
                "Error Rate Difference",
                f"{error_diff:.2f}%",
                delta="Lower is better" if error_diff > 0 else "Higher is better"
            )
        
        with summary_col3:
            total = comparison.get('total_requests', {}).get('total', 0)
            st.metric("Total Test Requests", f"{total:,}")
    
    # Raw data expander
    with st.expander("📋 View Raw Data"):
        st.json(stats)


if __name__ == "__main__":
    main()

