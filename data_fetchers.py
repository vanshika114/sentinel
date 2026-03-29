import yfinance as yf
import pandas as pd
from mftool import Mftool
import requests
from datetime import datetime, timedelta
from config import SERPER_API_KEY
import streamlit as st


class StockDataFetcher:
    """Fetch stock data from NSE/BSE using yfinance"""

    def __init__(self):
        self.cache = {}

    @st.cache_data(ttl=300)
    def get_stock_info(_self, ticker):
        """Get current stock information"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            previous_close = info.get('previousClose', 0)
            change = current_price - previous_close
            change_pct = (change / previous_close * 100) if previous_close else 0

            return {
                'ticker': ticker,
                'name': info.get('longName', ticker),
                'current_price': current_price,
                'previous_close': previous_close,
                'change': change,
                'change_percent': change_pct,
                'market_cap': info.get('marketCap', 0),
                'volume': info.get('volume', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
            }
        except Exception as e:
            st.error(f"Error fetching stock data for {ticker}: {str(e)}")
            return None

    @st.cache_data(ttl=3600)
    def get_historical_data(_self, ticker, period='1y'):
        """Get historical stock data"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            return hist
        except Exception as e:
            st.error(f"Error fetching historical data for {ticker}: {str(e)}")
            return None

    @st.cache_data(ttl=3600)
    def get_financials(_self, ticker):
        """Get basic financials for a stock"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'revenue': info.get('totalRevenue', 0),
                'net_income': info.get('netIncomeToCommon', 0),
                'eps': info.get('trailingEps', 0),
                'book_value': info.get('bookValue', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
            }
        except Exception as e:
            st.error(f"Error fetching financials for {ticker}: {str(e)}")
            return None


class MutualFundDataFetcher:
    """Fetch mutual fund data using mftool"""

    def __init__(self):
        self.mf = Mftool()

    @st.cache_data(ttl=3600)
    def get_scheme_details(_self, scheme_code):
        """Get mutual fund scheme details"""
        try:
            data = _self.mf.get_scheme_quote(scheme_code)
            if data:
                return {
                    'scheme_code': scheme_code,
                    'scheme_name': data.get('scheme_name', 'N/A'),
                    'nav': float(data.get('nav', 0)),
                    'date': data.get('last_updated', 'N/A')
                }
            return None
        except Exception as e:
            st.error(f"Error fetching mutual fund data for {scheme_code}: {str(e)}")
            return None

    @st.cache_data(ttl=3600)
    def get_scheme_historical(_self, scheme_code):
        """Get historical NAV data"""
        try:
            data = _self.mf.get_scheme_historical_nav(scheme_code)
            if data and 'data' in data:
                df = pd.DataFrame(data['data'])
                df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
                df['nav'] = pd.to_numeric(df['nav'])
                return df.sort_values('date')
            return None
        except Exception as e:
            st.error(f"Error fetching historical data for {scheme_code}: {str(e)}")
            return None

    @st.cache_data(ttl=86400)
    def get_all_schemes(_self):
        """Get list of all available mutual fund schemes"""
        try:
            return _self.mf.get_scheme_codes()
        except Exception as e:
            st.error(f"Error fetching scheme list: {str(e)}")
            return {}


class NewsDataFetcher:
    """Fetch news and sentiment data using Serper API"""

    def __init__(self, api_key=None):
        self.api_key = api_key or SERPER_API_KEY
        self.base_url = "https://google.serper.dev/search"

    def search_news(self, query, num_results=10):
        """Search for news articles"""
        try:
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }

            payload = {
                'q': query,
                'num': num_results,
                'type': 'news'
            }

            response = requests.post(self.base_url, json=payload, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                news_results = []

                if 'news' in data:
                    for item in data['news'][:num_results]:
                        news_results.append({
                            'title': item.get('title', ''),
                            'link': item.get('link', ''),
                            'snippet': item.get('snippet', ''),
                            'source': item.get('source', ''),
                            'date': item.get('date', '')
                        })

                return news_results
            else:
                st.error(f"Serper API error: {response.status_code}")
                return []

        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
            return []

    def get_market_news(self, ticker=None):
        """Get market news for specific ticker or general market"""
        if ticker:
            company_name = ticker.replace('.NS', '').replace('.BO', '')
            query = f"{company_name} stock market India NSE news"
        else:
            query = "Indian stock market NSE BSE latest news"
        return self.search_news(query)
