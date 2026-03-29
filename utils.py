import pandas as pd
from datetime import datetime, timedelta
import streamlit as st


def format_currency(amount, symbol='₹'):
    """Format currency in Indian format"""
    if amount >= 10000000:  # 1 Crore
        return f"{symbol}{amount/10000000:.2f} Cr"
    elif amount >= 100000:  # 1 Lakh
        return f"{symbol}{amount/100000:.2f}L"
    else:
        return f"{symbol}{amount:,.2f}"


def format_percentage(value):
    """Format percentage"""
    return f"{value:.2f}%"


def get_date_range(days=365):
    """Get date range for historical data"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return start_date, end_date


def display_metric_card(title, value, delta=None):
    """Display a metric card in Streamlit"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.metric(label=title, value=value, delta=delta)


def create_download_button(data, filename, label="Download Report"):
    """Create a download button for reports"""
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime='text/plain'
    )
