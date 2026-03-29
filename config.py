import os
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()


def _get_secret(key, default=None):
    """Read from env vars first, then Streamlit secrets (for cloud deployment)."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


# API Keys
EMERGENT_LLM_KEY = _get_secret('EMERGENT_LLM_KEY')
SERPER_API_KEY   = _get_secret('SERPER_API_KEY')
OPENAI_MODEL     = _get_secret('OPENAI_MODEL', 'gpt-4o-mini')

# Default Indian Stocks (NSE)
DEFAULT_STOCKS = [
    'RELIANCE.NS',
    'TCS.NS',
    'INFY.NS',
    'HDFCBANK.NS',
    'WIPRO.NS',
    'ICICIBANK.NS',
    'BHARTIARTL.NS',
    'ITC.NS'
]

# Default Mutual Funds (AMFI Codes)
DEFAULT_MUTUAL_FUNDS = {
    'HDFC Mid Cap Fund': '118989',
    'SBI Bluechip Fund': '119598',
    'ICICI Prudential Bluechip Fund': '100308'
}

# Tax Slabs New Regime (2026)
TAX_SLABS_NEW_REGIME = [
    (300000, 0),
    (700000, 0.05),
    (1000000, 0.10),
    (1200000, 0.15),
    (1500000, 0.20),
    (float('inf'), 0.30)
]

# App Configuration
APP_TITLE     = "Sentinel AI - Investment Research Dashboard"
APP_SUBTITLE  = "Multi-Agent AI for the Indian Investor"
REPORT_HEADER = "Market Signal Analysis Report"
