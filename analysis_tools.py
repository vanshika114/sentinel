import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import newton
from config import TAX_SLABS_NEW_REGIME


def calculate_returns(start_value, end_value):
    """Calculate simple returns as percentage"""
    if start_value == 0:
        return 0
    return ((end_value - start_value) / start_value) * 100


def calculate_cagr(start_value, end_value, years):
    """Calculate Compound Annual Growth Rate"""
    if start_value == 0 or years == 0:
        return 0
    return (((end_value / start_value) ** (1 / years)) - 1) * 100


def calculate_xirr(cash_flows, dates):
    """
    Calculate XIRR (Extended Internal Rate of Return)
    cash_flows: list of cash flows (negative = investment, positive = returns)
    dates: list of corresponding dates
    """
    try:
        if len(cash_flows) != len(dates):
            return 0

        # Convert dates to float (days from first date)
        base_date = dates[0]
        days = [(d - base_date).days for d in dates]

        def npv(rate):
            return sum(cf / (1 + rate) ** (d / 365.0) for cf, d in zip(cash_flows, days))

        xirr = newton(npv, 0.1)
        return xirr * 100
    except Exception:
        return 0


def calculate_alpha(portfolio_return, benchmark_return, beta, risk_free_rate=6.5):
    """
    Calculate Jensen's Alpha
    All rates in percentage
    """
    expected_return = risk_free_rate + beta * (benchmark_return - risk_free_rate)
    return portfolio_return - expected_return


def calculate_sharpe_ratio(returns_series, risk_free_rate=6.5):
    """
    Calculate Sharpe Ratio from a series of returns
    returns_series: pandas Series of daily/monthly returns (as %)
    risk_free_rate: annual risk-free rate in %
    """
    if returns_series is None or len(returns_series) < 2:
        return 0

    # Annualise risk-free rate to match the period
    periods_per_year = 252  # trading days
    rf_per_period = risk_free_rate / 100 / periods_per_year

    excess_returns = returns_series / 100 - rf_per_period
    if excess_returns.std() == 0:
        return 0

    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)
    return round(sharpe, 2)


def calculate_volatility(returns_series, annualise=True):
    """
    Calculate volatility (standard deviation of returns)
    returns_series: pandas Series of daily returns (as %)
    """
    if returns_series is None or len(returns_series) < 2:
        return 0
    vol = returns_series.std()
    if annualise:
        vol = vol * np.sqrt(252)
    return round(vol, 2)


def calculate_tax_new_regime(gross_income):
    """
    Calculate income tax under New Regime (2026)
    Returns: (taxable_income, tax_payable, effective_rate)
    """
    # Standard deduction under new regime
    standard_deduction = 75000
    taxable_income = max(0, gross_income - standard_deduction)

    # Rebate under section 87A (income up to 7L after deductions → zero tax)
    if taxable_income <= 700000:
        return taxable_income, 0, 0.0

    tax = 0
    prev_limit = 0

    for limit, rate in TAX_SLABS_NEW_REGIME:
        if taxable_income <= prev_limit:
            break
        taxable_in_slab = min(taxable_income, limit) - prev_limit
        tax += taxable_in_slab * rate
        prev_limit = limit

    # Health & Education Cess @ 4%
    tax += tax * 0.04

    effective_rate = (tax / gross_income * 100) if gross_income > 0 else 0
    return taxable_income, round(tax, 2), round(effective_rate, 2)


def calculate_joint_filing_benefit(income1, income2):
    """
    Estimate tax benefit if both spouses file separately vs combined
    (Indian tax law — no joint filing; this shows the benefit of income splitting)
    """
    _, tax1, _ = calculate_tax_new_regime(income1)
    _, tax2, _ = calculate_tax_new_regime(income2)
    combined_tax = tax1 + tax2

    # If income were filed as one person
    _, single_tax, _ = calculate_tax_new_regime(income1 + income2)

    benefit = single_tax - combined_tax
    return {
        'person1_tax': tax1,
        'person2_tax': tax2,
        'combined_tax': combined_tax,
        'single_filer_tax': single_tax,
        'tax_benefit': benefit
    }


def calculate_sip_returns(monthly_investment, annual_return_rate, years):
    """
    Calculate SIP (Systematic Investment Plan) maturity value
    monthly_investment: monthly SIP amount in ₹
    annual_return_rate: expected annual return in %
    years: investment horizon in years
    """
    monthly_rate = annual_return_rate / 100 / 12
    months = years * 12
    total_invested = monthly_investment * months

    if monthly_rate == 0:
        maturity_value = total_invested
    else:
        maturity_value = monthly_investment * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)

    wealth_gained = maturity_value - total_invested
    absolute_return = calculate_returns(total_invested, maturity_value)

    return {
        'total_invested': round(total_invested, 2),
        'maturity_value': round(maturity_value, 2),
        'wealth_gained': round(wealth_gained, 2),
        'absolute_return': round(absolute_return, 2),
        'xirr': annual_return_rate  # SIP XIRR ≈ assumed rate
    }


def calculate_beta(stock_returns, benchmark_returns):
    """
    Calculate Beta of a stock vs benchmark
    Both inputs: pandas Series of daily returns
    """
    try:
        df = pd.DataFrame({'stock': stock_returns, 'benchmark': benchmark_returns}).dropna()
        if len(df) < 10:
            return 1.0
        covariance = df['stock'].cov(df['benchmark'])
        variance = df['benchmark'].var()
        return round(covariance / variance, 2) if variance != 0 else 1.0
    except Exception:
        return 1.0


def get_rsi(price_series, period=14):
    """
    Calculate Relative Strength Index (RSI)
    price_series: pandas Series of closing prices
    """
    delta = price_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_moving_averages(price_series):
    """Calculate 20-day and 50-day moving averages"""
    return {
        'ma20': price_series.rolling(20).mean(),
        'ma50': price_series.rolling(50).mean(),
        'ma200': price_series.rolling(200).mean(),
    }
