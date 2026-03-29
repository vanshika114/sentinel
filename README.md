# 📊 Sentinel AI — Investment Research Dashboard

> Multi-Agent AI platform for the Indian Investor  
> Built with CrewAI · GPT-4o-mini · Streamlit · yfinance

---

## Features

| Feature | Description |
|---|---|
| 📈 Live Stock Analysis | Real-time NSE/BSE data with candlestick charts, RSI, moving averages |
| 🏦 Mutual Fund Tracker | NAV history and SIP calculator for 10,000+ Indian funds |
| 📰 Market News | Real-time news via Serper.dev |
| 🧾 Tax Calculator | New Regime 2026 with cess, 87A rebate, and household income splitting |
| 🤖 AI Reports | 4-agent CrewAI pipeline for stock, fund, and market reports |

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/sentinel-ai.git
cd sentinel-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API keys
Edit `.env`:
```
EMERGENT_LLM_KEY=your_openai_or_emergent_key
SERPER_API_KEY=your_serper_key
OPENAI_MODEL=gpt-4o-mini
```
Get a free Serper key at https://serper.dev (2,500 searches/month free)

### 4. Run
```bash
streamlit run app.py --server.port=3000
```

Open **http://localhost:3000** in your browser.

---

## Project Structure

```
sentinel-ai/
├── .env                  # API keys (do NOT commit to git)
├── config.py             # App configuration & constants
├── utils.py              # Formatting helpers
├── data_fetchers.py      # Stock, MF, and news data
├── analysis_tools.py     # CAGR, XIRR, RSI, tax calculations
├── agents.py             # CrewAI multi-agent system
├── app.py                # Main Streamlit dashboard
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Deployment

### Streamlit Community Cloud (free)
1. Push repo to GitHub
2. Go to https://share.streamlit.io
3. Connect your repo, set `app.py` as the entry point
4. Add secrets (API keys) in the Streamlit Cloud dashboard under **Settings → Secrets**

### Railway / Render
Add a `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

---

## ⚠️ Important
- **Never commit your `.env` file.** Add `.env` to `.gitignore`.
- The app uses real-time market data; results are for informational purposes only.
- Not financial advice.

---

*Built with ❤️ for Indian Investors*
