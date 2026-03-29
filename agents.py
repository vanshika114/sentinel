import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from config import EMERGENT_LLM_KEY, OPENAI_MODEL, SERPER_API_KEY
import streamlit as st

# Point CrewAI to Emergent LLM gateway (OpenAI-compatible)
os.environ["OPENAI_API_KEY"] = EMERGENT_LLM_KEY or ""
os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"  # newer openai SDK uses OPENAI_BASE_URL
os.environ["SERPER_API_KEY"] = SERPER_API_KEY or ""


class SentinelAIAgents:
    """Multi-Agent AI system for investment research"""

    def __init__(self):
        self.model = OPENAI_MODEL
        self.search_tool = SerperDevTool()

    # ── Agent Definitions ──────────────────────────────────────────────────────

    def _market_scouter(self):
        return Agent(
            role="Market Scouter",
            goal="Gather real-time market data, price movements, and technical indicators for Indian stocks.",
            backstory=(
                "You are a seasoned equity analyst who specialises in NSE and BSE markets. "
                "You monitor price action, volume, 52-week highs/lows, and technical signals "
                "to identify stocks that deserve closer attention."
            ),
            tools=[self.search_tool],
            llm=self.model,
            verbose=False,
            allow_delegation=False,
        )

    def _sentiment_analyst(self):
        return Agent(
            role="Sentiment Analyst",
            goal="Analyse news sentiment, social signals, and macro events affecting Indian equities.",
            backstory=(
                "You are an expert in market sentiment and behavioural finance. "
                "You comb through financial news, RBI announcements, FII/DII flows, and "
                "corporate disclosures to gauge the mood of the market."
            ),
            tools=[self.search_tool],
            llm=self.model,
            verbose=False,
            allow_delegation=False,
        )

    def _risk_strategist(self):
        return Agent(
            role="Risk Strategist",
            goal="Assess portfolio risk, volatility, beta, and recommend position-sizing guidelines.",
            backstory=(
                "You are a quantitative risk manager with deep expertise in Indian capital markets. "
                "You evaluate downside scenarios, sector concentration risk, and macro tail risks "
                "to help investors protect their capital."
            ),
            tools=[self.search_tool],
            llm=self.model,
            verbose=False,
            allow_delegation=False,
        )

    def _portfolio_reporter(self):
        return Agent(
            role="Portfolio Reporter",
            goal="Synthesise insights from all agents into a clear, actionable investment report.",
            backstory=(
                "You are a senior investment strategist who translates complex data into plain-language "
                "recommendations. You craft concise reports that help retail investors make informed decisions "
                "in the Indian market context."
            ),
            tools=[],
            llm=self.model,
            verbose=False,
            allow_delegation=False,
        )

    # ── Public Methods ─────────────────────────────────────────────────────────

    def analyze_stock(self, ticker: str, stock_data: dict = None) -> str:
        """Run multi-agent analysis on a single stock"""

        ticker_clean = ticker.replace('.NS', '').replace('.BO', '')
        context = ""
        if stock_data:
            context = (
                f"Current Price: ₹{stock_data.get('current_price', 'N/A')}, "
                f"Change: {stock_data.get('change_percent', 0):.2f}%, "
                f"Market Cap: ₹{stock_data.get('market_cap', 0):,}, "
                f"P/E: {stock_data.get('pe_ratio', 'N/A')}, "
                f"52W High: ₹{stock_data.get('52_week_high', 'N/A')}, "
                f"52W Low: ₹{stock_data.get('52_week_low', 'N/A')}"
            )

        # Tasks
        scout_task = Task(
            description=(
                f"Research {ticker_clean} (NSE ticker: {ticker}). "
                f"Pre-fetched data: {context}. "
                "Find latest price, volume, technical outlook (RSI, moving averages), and "
                "any recent corporate actions or results."
            ),
            expected_output="A structured market summary with technical signals for the stock.",
            agent=self._market_scouter(),
        )

        sentiment_task = Task(
            description=(
                f"Analyse recent news and market sentiment for {ticker_clean}. "
                "Identify bullish or bearish catalysts, any analyst upgrades/downgrades, "
                "and macro factors (RBI policy, FII flows, sector trends) that could affect the stock."
            ),
            expected_output="A sentiment analysis with key positive and negative drivers.",
            agent=self._sentiment_analyst(),
        )

        risk_task = Task(
            description=(
                f"Assess the risk profile of {ticker_clean}. "
                "Comment on volatility, beta vs Nifty 50, sector-specific risks, "
                "and suggest appropriate position sizing for a retail investor."
            ),
            expected_output="A risk assessment with suggested stop-loss levels and position-size guidance.",
            agent=self._risk_strategist(),
        )

        report_task = Task(
            description=(
                f"Compile a final investment report for {ticker_clean} using the findings from all agents. "
                "Include: Executive Summary, Technical Outlook, Sentiment, Risk Assessment, "
                "and a clear BUY / HOLD / SELL recommendation with rationale."
            ),
            expected_output="A comprehensive, structured investment report in markdown format.",
            agent=self._portfolio_reporter(),
        )

        crew = Crew(
            agents=[self._market_scouter(), self._sentiment_analyst(),
                    self._risk_strategist(), self._portfolio_reporter()],
            tasks=[scout_task, sentiment_task, risk_task, report_task],
            process=Process.sequential,
            verbose=False,
        )

        try:
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            return f"⚠️ Analysis could not be completed: {str(e)}"

    def analyze_mutual_fund(self, fund_name: str, scheme_code: str, nav: float = None) -> str:
        """Run multi-agent analysis on a mutual fund"""

        context = f"Scheme Code: {scheme_code}"
        if nav:
            context += f", Current NAV: ₹{nav}"

        scout_task = Task(
            description=(
                f"Research the mutual fund '{fund_name}' ({context}). "
                "Find its category, AUM, expense ratio, fund manager, and recent NAV performance."
            ),
            expected_output="A summary of fund facts and recent performance.",
            agent=self._market_scouter(),
        )

        sentiment_task = Task(
            description=(
                f"Analyse market sentiment around '{fund_name}'. "
                "Look for recent inflow/outflow data, rating agency assessments, "
                "and any news about the AMC or fund management changes."
            ),
            expected_output="Sentiment analysis for the fund.",
            agent=self._sentiment_analyst(),
        )

        risk_task = Task(
            description=(
                f"Evaluate the risk metrics of '{fund_name}'. "
                "Comment on standard deviation, Sharpe ratio, portfolio concentration, "
                "and suitability for different investor risk profiles."
            ),
            expected_output="Risk metrics and suitability assessment.",
            agent=self._risk_strategist(),
        )

        report_task = Task(
            description=(
                f"Write an investment report for '{fund_name}'. "
                "Include: Fund Overview, Performance Analysis, Risk Profile, "
                "and a recommendation (Invest / Hold / Avoid) with reasoning."
            ),
            expected_output="Comprehensive mutual fund report in markdown format.",
            agent=self._portfolio_reporter(),
        )

        crew = Crew(
            agents=[self._market_scouter(), self._sentiment_analyst(),
                    self._risk_strategist(), self._portfolio_reporter()],
            tasks=[scout_task, sentiment_task, risk_task, report_task],
            process=Process.sequential,
            verbose=False,
        )

        try:
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            return f"⚠️ Fund analysis could not be completed: {str(e)}"

    def generate_market_report(self, tickers: list) -> str:
        """Generate a broad market overview report for multiple tickers"""

        tickers_str = ", ".join([t.replace('.NS', '') for t in tickers])

        scout_task = Task(
            description=(
                f"Provide a market overview for these Indian stocks: {tickers_str}. "
                "Summarise overall Nifty 50 / Sensex trend, sector rotation, and FII/DII activity today."
            ),
            expected_output="Market overview with index levels and sector trends.",
            agent=self._market_scouter(),
        )

        sentiment_task = Task(
            description=(
                "Identify the top 3 bullish and top 3 bearish macro/news factors "
                "affecting the Indian market right now. Consider RBI policy, global cues, and earnings season."
            ),
            expected_output="Top bullish and bearish market catalysts.",
            agent=self._sentiment_analyst(),
        )

        risk_task = Task(
            description=(
                "Outline the key systemic risks to the Indian market at this time — "
                "currency risk, inflation, geopolitical factors, and liquidity conditions."
            ),
            expected_output="Macro risk factors and their likely market impact.",
            agent=self._risk_strategist(),
        )

        report_task = Task(
            description=(
                f"Write a 'Market Signal Analysis Report' covering: {tickers_str}. "
                "Include: Market Snapshot, Key Themes, Risks, Opportunities, and "
                "a short-term market outlook (1-3 months)."
            ),
            expected_output="Full market report in markdown format.",
            agent=self._portfolio_reporter(),
        )

        crew = Crew(
            agents=[self._market_scouter(), self._sentiment_analyst(),
                    self._risk_strategist(), self._portfolio_reporter()],
            tasks=[scout_task, sentiment_task, risk_task, report_task],
            process=Process.sequential,
            verbose=False,
        )

        try:
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            return f"⚠️ Market report could not be generated: {str(e)}"
