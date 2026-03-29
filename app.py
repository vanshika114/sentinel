import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

from config import *
from data_fetchers import StockDataFetcher, MutualFundDataFetcher, NewsDataFetcher
from analysis_tools import (
    calculate_returns, calculate_cagr, calculate_sip_returns,
    calculate_tax_new_regime, calculate_joint_filing_benefit,
    calculate_sharpe_ratio, calculate_volatility, get_rsi, get_moving_averages
)
from agents import SentinelAIAgents
from utils import format_currency, format_percentage, create_download_button

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2.2rem; margin: 0; }
    .main-header p  { color: #a8b2d8; font-size: 1rem; margin: 0.5rem 0 0; }

    .metric-card {
        background: #1e2a3a;
        border: 1px solid #2d3f55;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .positive { color: #00d68f; }
    .negative { color: #ff4757; }

    .news-card {
        background: #1e2a3a;
        border-left: 4px solid #e94560;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
    }
    .news-card h4 { margin: 0 0 0.3rem; font-size: 0.95rem; }
    .news-card p  { margin: 0; font-size: 0.8rem; color: #a8b2d8; }

    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>📊 {APP_TITLE}</h1>
    <p>{APP_SUBTITLE}</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    selected_stocks = st.multiselect(
        "Select Stocks (NSE)",
        options=DEFAULT_STOCKS,
        default=DEFAULT_STOCKS[:4],
        help="Pick stocks to track"
    )
    custom_ticker = st.text_input("Add custom ticker (e.g. TATASTEEL.NS)", "")
    if custom_ticker and custom_ticker not in selected_stocks:
        selected_stocks.append(custom_ticker.upper().strip())

    st.markdown("---")
    chart_period = st.selectbox("Chart Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    st.markdown("---")
    st.info("💡 AI analysis uses CrewAI + GPT-4o-mini")
    st.markdown(f"*Last refreshed: {datetime.now().strftime('%d %b %Y %H:%M')}*")

# ── Instantiate Data Fetchers ──────────────────────────────────────────────────
stock_fetcher = StockDataFetcher()
mf_fetcher    = MutualFundDataFetcher()
news_fetcher  = NewsDataFetcher()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Stock Analysis",
    "🏦 Mutual Funds",
    "📰 Market News",
    "🧾 Tax Calculator",
    "🤖 AI Report Generator"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — STOCK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📈 Live Stock Analysis")

    if not selected_stocks:
        st.warning("Please select at least one stock from the sidebar.")
    else:
        # Live price cards
        cols = st.columns(min(len(selected_stocks), 4))
        stock_data_map = {}

        for i, ticker in enumerate(selected_stocks):
            data = stock_fetcher.get_stock_info(ticker)
            if data:
                stock_data_map[ticker] = data
                with cols[i % 4]:
                    color = "positive" if data['change_percent'] >= 0 else "negative"
                    arrow = "▲" if data['change_percent'] >= 0 else "▼"
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{ticker.replace('.NS','')}</strong><br>
                        <span style="font-size:1.4rem;">₹{data['current_price']:,.2f}</span><br>
                        <span class="{color}">{arrow} {abs(data['change_percent']):.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # Detailed view for selected stock
        detail_ticker = st.selectbox("View detailed chart for:", selected_stocks)

        if detail_ticker:
            hist = stock_fetcher.get_historical_data(detail_ticker, period=chart_period)

            if hist is not None and not hist.empty:
                col_a, col_b = st.columns([2, 1])

                with col_a:
                    mas = get_moving_averages(hist['Close'])
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=hist.index,
                        open=hist['Open'], high=hist['High'],
                        low=hist['Low'],   close=hist['Close'],
                        name="Price"
                    ))
                    fig.add_trace(go.Scatter(x=hist.index, y=mas['ma20'],
                                             name="MA20", line=dict(color="#f0a500", width=1)))
                    fig.add_trace(go.Scatter(x=hist.index, y=mas['ma50'],
                                             name="MA50", line=dict(color="#00d68f", width=1)))
                    fig.update_layout(
                        title=f"{detail_ticker} — Price Chart",
                        xaxis_rangeslider_visible=False,
                        template="plotly_dark",
                        height=450
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_b:
                    d = stock_data_map.get(detail_ticker, {})
                    if d:
                        st.markdown("**Key Metrics**")
                        st.metric("Current Price", f"₹{d['current_price']:,.2f}",
                                  f"{d['change_percent']:+.2f}%")
                        st.metric("Market Cap", format_currency(d['market_cap']))
                        st.metric("52W High", f"₹{d['52_week_high']:,.2f}")
                        st.metric("52W Low",  f"₹{d['52_week_low']:,.2f}")
                        st.metric("Volume",   f"{d['volume']:,}")
                        if d.get('pe_ratio'):
                            st.metric("P/E Ratio", f"{d['pe_ratio']:.2f}")

                # RSI chart
                rsi = get_rsi(hist['Close'])
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=hist.index, y=rsi, name="RSI", line=dict(color="#e94560")))
                fig_rsi.add_hline(y=70, line_dash="dot", line_color="red",   annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold (30)")
                fig_rsi.update_layout(title="RSI (14)", template="plotly_dark", height=200,
                                      yaxis=dict(range=[0, 100]))
                st.plotly_chart(fig_rsi, use_container_width=True)

                # Returns summary
                if len(hist) > 1:
                    ret_1m = calculate_returns(hist['Close'].iloc[-22], hist['Close'].iloc[-1]) if len(hist) >= 22 else 0
                    ret_6m = calculate_returns(hist['Close'].iloc[-126], hist['Close'].iloc[-1]) if len(hist) >= 126 else 0
                    ret_1y = calculate_returns(hist['Close'].iloc[0], hist['Close'].iloc[-1])
                    daily_returns = hist['Close'].pct_change().dropna() * 100

                    r1, r2, r3, r4, r5 = st.columns(5)
                    r1.metric("1-Month Return",    format_percentage(ret_1m),  delta=format_percentage(ret_1m))
                    r2.metric("6-Month Return",    format_percentage(ret_6m),  delta=format_percentage(ret_6m))
                    r3.metric("1-Year Return",     format_percentage(ret_1y),  delta=format_percentage(ret_1y))
                    r4.metric("Volatility (Ann.)", format_percentage(calculate_volatility(daily_returns)))
                    r5.metric("Sharpe Ratio",      str(calculate_sharpe_ratio(daily_returns)))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MUTUAL FUNDS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🏦 Mutual Fund Tracker")

    fund_options = list(DEFAULT_MUTUAL_FUNDS.keys())
    selected_funds = st.multiselect("Select Funds", fund_options, default=fund_options[:2])

    if selected_funds:
        fund_cols = st.columns(len(selected_funds))
        for i, fund_name in enumerate(selected_funds):
            code = DEFAULT_MUTUAL_FUNDS[fund_name]
            details = mf_fetcher.get_scheme_details(code)
            if details:
                with fund_cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{fund_name}</strong><br>
                        NAV: <span style="font-size:1.2rem;">₹{details['nav']:.2f}</span><br>
                        <small>Updated: {details['date']}</small>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # NAV historical chart
        detail_fund = st.selectbox("View NAV history for:", selected_funds)
        code = DEFAULT_MUTUAL_FUNDS[detail_fund]
        hist_nav = mf_fetcher.get_scheme_historical(code)

        if hist_nav is not None and not hist_nav.empty:
            fig_nav = px.line(hist_nav.tail(365), x='date', y='nav',
                              title=f"{detail_fund} — NAV History (1Y)",
                              template="plotly_dark")
            fig_nav.update_traces(line_color="#e94560")
            st.plotly_chart(fig_nav, use_container_width=True)

            # CAGR metrics
            if len(hist_nav) >= 252:
                nav_1y_ago = hist_nav.iloc[-252]['nav']
                nav_now    = hist_nav.iloc[-1]['nav']
                cagr_1y    = calculate_cagr(nav_1y_ago, nav_now, 1)
                st.metric("1-Year CAGR", format_percentage(cagr_1y))

    st.markdown("---")
    st.subheader("🧮 SIP Calculator")

    sip_col1, sip_col2, sip_col3 = st.columns(3)
    with sip_col1:
        sip_amount = st.number_input("Monthly SIP (₹)", min_value=500, value=5000, step=500)
    with sip_col2:
        sip_return = st.slider("Expected Annual Return (%)", 6.0, 20.0, 12.0, 0.5)
    with sip_col3:
        sip_years = st.slider("Investment Period (Years)", 1, 30, 10)

    sip_result = calculate_sip_returns(sip_amount, sip_return, sip_years)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Invested",  format_currency(sip_result['total_invested']))
    s2.metric("Maturity Value",  format_currency(sip_result['maturity_value']))
    s3.metric("Wealth Gained",   format_currency(sip_result['wealth_gained']))
    s4.metric("Absolute Return", format_percentage(sip_result['absolute_return']))

    # Growth chart
    months = list(range(1, sip_years * 12 + 1))
    invested_vals = [sip_amount * m for m in months]
    mr = sip_return / 100 / 12
    maturity_vals = [sip_amount * (((1 + mr) ** m - 1) / mr) * (1 + mr) for m in months]

    fig_sip = go.Figure()
    fig_sip.add_trace(go.Scatter(x=months, y=invested_vals, name="Amount Invested",
                                  fill='tozeroy', line=dict(color="#a8b2d8")))
    fig_sip.add_trace(go.Scatter(x=months, y=maturity_vals, name="Projected Value",
                                  fill='tonexty',  line=dict(color="#00d68f")))
    fig_sip.update_layout(title="SIP Growth Projection", template="plotly_dark",
                           xaxis_title="Month", yaxis_title="₹ Value")
    st.plotly_chart(fig_sip, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MARKET NEWS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📰 Market News & Sentiment")

    news_col1, news_col2 = st.columns([1, 3])
    with news_col1:
        news_ticker = st.selectbox("News for:", ["General Market"] + selected_stocks)
        if st.button("🔄 Refresh News"):
            st.cache_data.clear()

    ticker_for_news = None if news_ticker == "General Market" else news_ticker
    articles = news_fetcher.get_market_news(ticker_for_news)

    if articles:
        for article in articles:
            st.markdown(f"""
            <div class="news-card">
                <h4><a href="{article['link']}" target="_blank">{article['title']}</a></h4>
                <p>{article['snippet']}</p>
                <small>🗞 {article['source']} &nbsp;|&nbsp; 📅 {article['date']}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No news found. Check your Serper API key.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TAX CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🧾 Income Tax Calculator (New Regime 2026)")

    tax_col1, tax_col2 = st.columns(2)

    with tax_col1:
        st.markdown("#### Single Filer")
        gross_income = st.number_input("Gross Annual Income (₹)", min_value=0,
                                        value=1200000, step=50000)
        taxable, tax, eff_rate = calculate_tax_new_regime(gross_income)

        t1, t2, t3 = st.columns(3)
        t1.metric("Taxable Income",  format_currency(taxable))
        t2.metric("Tax Payable",     format_currency(tax))
        t3.metric("Effective Rate",  format_percentage(eff_rate))

        # Tax breakdown chart
        slabs = ["0–3L (0%)", "3–7L (5%)", "7–10L (10%)", "10–12L (15%)", "12–15L (20%)", "15L+ (30%)"]
        limits = [300000, 700000, 1000000, 1200000, 1500000, float('inf')]
        rates  = [0, 0.05, 0.10, 0.15, 0.20, 0.30]
        slab_tax = []
        prev = 0
        for lim, rate in zip(limits, rates):
            amt = min(taxable, lim) - prev
            slab_tax.append(max(0, amt * rate))
            prev = lim
            if taxable <= lim:
                break

        if any(s > 0 for s in slab_tax):
            fig_tax = px.pie(
                values=[s for s in slab_tax if s > 0],
                names=[slabs[i] for i, s in enumerate(slab_tax) if s > 0],
                title="Tax by Slab",
                template="plotly_dark"
            )
            st.plotly_chart(fig_tax, use_container_width=True)

    with tax_col2:
        st.markdown("#### Dual Income (Household)")
        income1 = st.number_input("Income — Person 1 (₹)", min_value=0, value=1500000, step=50000)
        income2 = st.number_input("Income — Person 2 (₹)", min_value=0, value=800000,  step=50000)

        result = calculate_joint_filing_benefit(income1, income2)

        j1, j2 = st.columns(2)
        j1.metric("Person 1 Tax",  format_currency(result['person1_tax']))
        j2.metric("Person 2 Tax",  format_currency(result['person2_tax']))

        j3, j4 = st.columns(2)
        j3.metric("Combined Tax",  format_currency(result['combined_tax']))
        j4.metric("Single-filer Tax", format_currency(result['single_filer_tax']))

        if result['tax_benefit'] > 0:
            st.success(f"✅ Tax savings via income splitting: {format_currency(result['tax_benefit'])}")
        else:
            st.info("No additional tax savings from income splitting in this scenario.")

    st.markdown("---")
    st.caption("📌 Includes 4% Health & Education Cess. Standard deduction ₹75,000 applied. "
               "Rebate u/s 87A applied for taxable income ≤ ₹7L.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AI REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🤖 AI Report Generator (Multi-Agent)")

    report_type = st.radio(
        "Report Type",
        ["Stock Analysis", "Mutual Fund Analysis", "Full Market Report"],
        horizontal=True
    )

    ai_agents = SentinelAIAgents()

    if report_type == "Stock Analysis":
        ai_ticker = st.selectbox("Select Stock:", selected_stocks)
        if st.button("🚀 Generate Stock Report", type="primary"):
            with st.spinner("🤖 Agents are analysing the stock…"):
                stock_data = stock_fetcher.get_stock_info(ai_ticker)
                report = ai_agents.analyze_stock(ai_ticker, stock_data)

            st.markdown("---")
            st.markdown(report)
            create_download_button(
                report,
                f"{ai_ticker}_report_{datetime.now().strftime('%Y%m%d')}.txt",
                "📥 Download Report"
            )

    elif report_type == "Mutual Fund Analysis":
        ai_fund = st.selectbox("Select Fund:", list(DEFAULT_MUTUAL_FUNDS.keys()))
        if st.button("🚀 Generate Fund Report", type="primary"):
            code = DEFAULT_MUTUAL_FUNDS[ai_fund]
            details = mf_fetcher.get_scheme_details(code)
            nav = details['nav'] if details else None

            with st.spinner("🤖 Agents are analysing the fund…"):
                report = ai_agents.analyze_mutual_fund(ai_fund, code, nav)

            st.markdown("---")
            st.markdown(report)
            create_download_button(
                report,
                f"MF_{ai_fund.replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                "📥 Download Report"
            )

    else:  # Full Market Report
        if st.button("🚀 Generate Market Report", type="primary"):
            with st.spinner("🤖 Agents are generating the market report…"):
                report = ai_agents.generate_market_report(selected_stocks)

            st.markdown("---")
            st.markdown(report)
            create_download_button(
                report,
                f"market_report_{datetime.now().strftime('%Y%m%d')}.txt",
                "📥 Download Report"
            )

    st.markdown("---")
    st.caption("⚡ Powered by CrewAI | GPT-4o-mini | Serper News Search")
