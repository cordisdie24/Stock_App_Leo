import math
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Advanced Stock Analyzer", layout="wide")
st.title("Advanced Stock Analysis Dashboard")

st.sidebar.header("Settings")

tickers_input = st.sidebar.text_input(
    "Enter stock tickers separated by commas",
    value="AAPL,MSFT,NVDA"
)

show_value_estimate = st.sidebar.checkbox("Value Estimate", value=True)
show_raw_data = st.sidebar.checkbox("Show raw data", value=False)


def clean_tickers(user_input: str) -> List[str]:
    tickers = [t.strip().upper() for t in user_input.split(",")]
    tickers = [t for t in tickers if t]
    return list(dict.fromkeys(tickers))


def safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def get_first_valid_value(df: pd.DataFrame, candidates: List[str]) -> Optional[float]:
    try:
        if df is None or df.empty:
            return None
        for candidate in candidates:
            if candidate in df.index:
                row = df.loc[candidate]
                if isinstance(row, pd.Series):
                    for value in row:
                        val = safe_float(value)
                        if val is not None:
                            return val
                else:
                    val = safe_float(row)
                    if val is not None:
                        return val
    except Exception:
        return None
    return None


@st.cache_data(ttl=3600, show_spinner="Fetching stock data...")
def load_price_data(tickers: List[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        tickers,
        period="1y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="ticker"
    )
    return df


@st.cache_data(ttl=3600, show_spinner="Fetching company data...")
def load_stock_package(ticker: str) -> dict:
    stock = yf.Ticker(ticker)

    package = {
        "info": {},
        "income_stmt": pd.DataFrame(),
        "balance_sheet": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
        "dividends": pd.Series(dtype=float),
    }

    try:
        info = stock.info
        if isinstance(info, dict):
            package["info"] = info
    except Exception:
        pass

    try:
        package["income_stmt"] = stock.income_stmt
    except Exception:
        pass

    try:
        package["balance_sheet"] = stock.balance_sheet
    except Exception:
        pass

    try:
        package["cashflow"] = stock.cashflow
    except Exception:
        pass

    try:
        package["dividends"] = stock.dividends
    except Exception:
        pass

    return package


def extract_close_series(price_data: pd.DataFrame, ticker: str) -> pd.Series:
    try:
        if isinstance(price_data.columns, pd.MultiIndex):
            if ticker in price_data.columns.get_level_values(0):
                s = price_data[ticker]["Close"].dropna()
                s.name = ticker
                return s
        if "Close" in price_data.columns:
            s = price_data["Close"].dropna()
            s.name = ticker
            return s
    except Exception:
        pass
    return pd.Series(dtype=float, name=ticker)


def compute_stock_metrics(close: pd.Series) -> dict:
    daily_returns = close.pct_change().dropna()
    total_return = (close.iloc[-1] / close.iloc[0]) - 1
    annual_volatility = daily_returns.std() * math.sqrt(252)

    ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

    running_max = close.cummax()
    drawdown = (close / running_max) - 1
    max_drawdown = drawdown.min()

    sharpe = None
    if len(daily_returns) > 1 and daily_returns.std() != 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * math.sqrt(252)

    return {
        "latest_close": float(close.iloc[-1]),
        "total_return": float(total_return),
        "annual_volatility": float(annual_volatility),
        "ma50": float(ma50) if ma50 is not None and pd.notna(ma50) else None,
        "ma200": float(ma200) if ma200 is not None and pd.notna(ma200) else None,
        "max_drawdown": float(max_drawdown),
        "sharpe": float(sharpe) if sharpe is not None and pd.notna(sharpe) else None,
    }


def compute_fallback_valuation_metrics(ticker: str, current_price: float) -> dict:
    package = load_stock_package(ticker)
    info = package["info"]
    income_stmt = package["income_stmt"]
    balance_sheet = package["balance_sheet"]
    cashflow = package["cashflow"]
    dividends = package["dividends"]

    market_cap = safe_float(info.get("marketCap"))
    shares_outstanding = safe_float(info.get("sharesOutstanding"))
    enterprise_value = safe_float(info.get("enterpriseValue"))
    target_price = safe_float(info.get("targetMeanPrice"))

    pe = safe_float(info.get("trailingPE"))
    pb = safe_float(info.get("priceToBook"))
    peg = safe_float(info.get("pegRatio"))
    ev_to_ebitda = safe_float(info.get("enterpriseToEbitda"))
    dividend_yield = safe_float(info.get("dividendYield"))

    net_income = get_first_valid_value(
        income_stmt,
        [
            "Net Income",
            "Net Income Common Stockholders",
            "Net Income Including Noncontrolling Interests",
        ],
    )

    ebitda = get_first_valid_value(
        income_stmt,
        [
            "EBITDA",
            "Normalized EBITDA",
        ],
    )

    total_equity = get_first_valid_value(
        balance_sheet,
        [
            "Stockholders Equity",
            "Total Equity Gross Minority Interest",
            "Common Stock Equity",
            "Total Stockholder Equity",
        ],
    )

    if market_cap is None and shares_outstanding is not None:
        market_cap = current_price * shares_outstanding

    if pe is None and market_cap is not None and net_income not in [None, 0]:
        pe = market_cap / net_income if net_income > 0 else None

    if pb is None and market_cap is not None and total_equity not in [None, 0]:
        pb = market_cap / total_equity if total_equity > 0 else None

    if ev_to_ebitda is None and enterprise_value is not None and ebitda not in [None, 0]:
        ev_to_ebitda = enterprise_value / ebitda if ebitda > 0 else None

    if dividend_yield is None:
        try:
            if dividends is not None and not dividends.empty:
                recent_dividends = dividends[dividends.index >= (pd.Timestamp.today() - pd.Timedelta(days=365))]
                annual_dividends = safe_float(recent_dividends.sum())
                if annual_dividends is not None and current_price > 0:
                    dividend_yield = annual_dividends / current_price
        except Exception:
            pass

    book_value_per_share = None
    eps = None
    implied_fair_value = None

    if shares_outstanding not in [None, 0]:
        if total_equity is not None and shares_outstanding > 0:
            book_value_per_share = total_equity / shares_outstanding
        if net_income is not None and shares_outstanding > 0:
            eps = net_income / shares_outstanding

    fair_value_components = []

    if eps is not None and eps > 0:
        fair_value_components.append(eps * 18)

    if book_value_per_share is not None and book_value_per_share > 0:
        fair_value_components.append(book_value_per_share * 3)

    if target_price is not None and target_price > 0:
        fair_value_components.append(target_price)

    if fair_value_components:
        implied_fair_value = sum(fair_value_components) / len(fair_value_components)

    return {
        "market_cap": market_cap,
        "shares_outstanding": shares_outstanding,
        "net_income": net_income,
        "ebitda": ebitda,
        "total_equity": total_equity,
        "pe": pe,
        "pb": pb,
        "peg": peg,
        "ev_to_ebitda": ev_to_ebitda,
        "dividend_yield": dividend_yield,
        "target_price": target_price,
        "eps": eps,
        "book_value_per_share": book_value_per_share,
        "implied_fair_value": implied_fair_value,
    }


def run_value_estimate(ticker: str, close: pd.Series) -> dict:
    current_price = float(close.iloc[-1])
    metrics = compute_fallback_valuation_metrics(ticker, current_price)

    pe = metrics["pe"]
    pb = metrics["pb"]
    peg = metrics["peg"]
    ev_to_ebitda = metrics["ev_to_ebitda"]
    dividend_yield = metrics["dividend_yield"]
    target_price = metrics["target_price"]
    implied_fair_value = metrics["implied_fair_value"]

    score = 0
    reasons = []

    ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

    if pe is not None:
        if pe < 18:
            score += 2
            reasons.append(f"P/E looks attractive at {pe:.2f}")
        elif pe < 25:
            score += 1
            reasons.append(f"P/E is reasonable at {pe:.2f}")
        elif pe > 35:
            score -= 2
            reasons.append(f"P/E looks expensive at {pe:.2f}")
        else:
            reasons.append(f"P/E is somewhat elevated at {pe:.2f}")

    if pb is not None:
        if pb < 3:
            score += 1
            reasons.append(f"Price to Book is moderate at {pb:.2f}")
        elif pb > 8:
            score -= 1
            reasons.append(f"Price to Book is high at {pb:.2f}")

    if peg is not None:
        if peg < 1:
            score += 2
            reasons.append(f"PEG looks attractive at {peg:.2f}")
        elif peg > 2:
            score -= 2
            reasons.append(f"PEG looks expensive at {peg:.2f}")

    if ev_to_ebitda is not None:
        if ev_to_ebitda < 10:
            score += 1
            reasons.append(f"EV/EBITDA looks attractive at {ev_to_ebitda:.2f}")
        elif ev_to_ebitda > 18:
            score -= 1
            reasons.append(f"EV/EBITDA looks expensive at {ev_to_ebitda:.2f}")

    if dividend_yield is not None:
        if dividend_yield > 0.03:
            score += 1
            reasons.append(f"Dividend yield is supportive at {dividend_yield:.2%}")

    if target_price is not None:
        target_upside = (target_price / current_price) - 1
        if target_upside > 0.15:
            score += 2
            reasons.append(f"Analyst target implies upside of {target_upside:.2%}")
        elif target_upside > 0.05:
            score += 1
            reasons.append(f"Analyst target implies moderate upside of {target_upside:.2%}")
        elif target_upside < -0.10:
            score -= 2
            reasons.append(f"Analyst target implies downside of {target_upside:.2%}")

    if implied_fair_value is not None:
        fair_value_gap = (implied_fair_value / current_price) - 1
        if fair_value_gap > 0.15:
            score += 2
            reasons.append(f"Estimated fair value implies upside of {fair_value_gap:.2%}")
        elif fair_value_gap > 0.05:
            score += 1
            reasons.append(f"Estimated fair value implies modest upside of {fair_value_gap:.2%}")
        elif fair_value_gap < -0.10:
            score -= 2
            reasons.append(f"Estimated fair value implies downside of {fair_value_gap:.2%}")

    if ma50 is not None:
        if current_price > ma50:
            score += 1
            reasons.append("Price is above the 50 day moving average")
        else:
            score -= 1
            reasons.append("Price is below the 50 day moving average")

    if ma200 is not None:
        if current_price > ma200:
            score += 2
            reasons.append("Price is above the 200 day moving average")
        else:
            score -= 2
            reasons.append("Price is below the 200 day moving average")

    if len(close) >= 126:
        six_month_return = (close.iloc[-1] / close.iloc[-126]) - 1
        if six_month_return > 0.15:
            score += 1
            reasons.append(f"Strong 6 month momentum at {six_month_return:.2%}")
        elif six_month_return < -0.15:
            score -= 1
            reasons.append(f"Weak 6 month momentum at {six_month_return:.2%}")

    daily_returns = close.pct_change().dropna()
    if not daily_returns.empty:
        annual_volatility = daily_returns.std() * math.sqrt(252)
        if annual_volatility > 0.50:
            score -= 1
            reasons.append(f"High annualized volatility at {annual_volatility:.2%}")

        running_max = close.cummax()
        drawdown = (close / running_max) - 1
        max_drawdown = drawdown.min()
        if max_drawdown < -0.35:
            score -= 1
            reasons.append(f"Large max drawdown at {max_drawdown:.2%}")

    if score >= 5:
        value_label = "Undervalued"
        action_label = "BUY"
    elif score >= 2:
        value_label = "Slightly Undervalued / Fair"
        action_label = "WATCH"
    elif score >= -1:
        value_label = "Fairly Valued"
        action_label = "HOLD / WATCH"
    elif score >= -4:
        value_label = "Slightly Overvalued"
        action_label = "CAUTION"
    else:
        value_label = "Overvalued"
        action_label = "AVOID"

    return {
        "score": score,
        "value_label": value_label,
        "action_label": action_label,
        "current_price": current_price,
        "pe": pe,
        "pb": pb,
        "peg": peg,
        "ev_to_ebitda": ev_to_ebitda,
        "dividend_yield": dividend_yield,
        "target_price": target_price,
        "eps": metrics["eps"],
        "book_value_per_share": metrics["book_value_per_share"],
        "implied_fair_value": implied_fair_value,
        "reasons": reasons,
    }


def normalize_prices(close_df: pd.DataFrame) -> pd.DataFrame:
    return close_df / close_df.iloc[0] * 100


tickers = clean_tickers(tickers_input)

if not tickers:
    st.info("Enter at least one ticker in the sidebar.")
    st.stop()

price_data = load_price_data(tickers)

close_table = pd.DataFrame()
for ticker in tickers:
    s = extract_close_series(price_data, ticker)
    if not s.empty:
        close_table[ticker] = s

close_table = close_table.dropna(how="all")

if close_table.empty:
    st.error("No valid stock data was found. Check your tickers and try again.")
    st.stop()

valid_tickers = close_table.columns.tolist()

st.header("1. Stock Signal")

selected_ticker = st.selectbox("Choose a stock for deeper analysis", valid_tickers, index=0)
selected_close = close_table[selected_ticker].dropna()

metrics = compute_stock_metrics(selected_close)

c1, c2, c3 = st.columns(3)
c1.metric("Latest Close", f"${metrics['latest_close']:,.2f}")
c2.metric("1 Year Return", f"{metrics['total_return']:.2%}")
c3.metric("Annualized Volatility", f"{metrics['annual_volatility']:.2%}")

c4, c5, c6 = st.columns(3)
c4.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
c5.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}" if metrics["sharpe"] is not None else "N/A")
c6.metric("50 Day MA", f"${metrics['ma50']:,.2f}" if metrics["ma50"] is not None else "N/A")

if show_value_estimate:
    st.divider()
    st.subheader("Value Estimate")

    valuation = run_value_estimate(selected_ticker, selected_close)

    v1, v2, v3 = st.columns(3)
    v1.metric("Valuation View", valuation["value_label"])
    v2.metric("Action Bias", valuation["action_label"])
    v3.metric("Valuation Score", valuation["score"])

    with st.expander("See valuation details"):
        st.write(f"Current Price: ${valuation['current_price']:,.2f}")
        st.write(f"Trailing P/E: {valuation['pe']:.2f}" if valuation["pe"] is not None else "Trailing P/E: N/A")
        st.write(f"Price to Book: {valuation['pb']:.2f}" if valuation["pb"] is not None else "Price to Book: N/A")
        st.write(f"PEG Ratio: {valuation['peg']:.2f}" if valuation["peg"] is not None else "PEG Ratio: N/A")
        st.write(f"EV/EBITDA: {valuation['ev_to_ebitda']:.2f}" if valuation["ev_to_ebitda"] is not None else "EV/EBITDA: N/A")
        st.write(
            f"Dividend Yield: {valuation['dividend_yield']:.2%}"
            if valuation["dividend_yield"] is not None else "Dividend Yield: N/A"
        )
        st.write(
            f"Analyst Mean Target: ${valuation['target_price']:,.2f}"
            if valuation["target_price"] is not None else "Analyst Mean Target: N/A"
        )
        st.write(
            f"EPS Estimate: ${valuation['eps']:,.2f}"
            if valuation["eps"] is not None else "EPS Estimate: N/A"
        )
        st.write(
            f"Book Value Per Share: ${valuation['book_value_per_share']:,.2f}"
            if valuation["book_value_per_share"] is not None else "Book Value Per Share: N/A"
        )
        st.write(
            f"Implied Fair Value: ${valuation['implied_fair_value']:,.2f}"
            if valuation["implied_fair_value"] is not None else "Implied Fair Value: N/A"
        )

        st.write("Reasons:")
        for reason in valuation["reasons"]:
            st.write(f"• {reason}")

fig_single = go.Figure()
fig_single.add_trace(
    go.Scatter(
        x=selected_close.index,
        y=selected_close.values,
        mode="lines",
        name=selected_ticker,
        line=dict(width=2)
    )
)

if metrics["ma50"] is not None:
    ma50_series = selected_close.rolling(50).mean()
    fig_single.add_trace(
        go.Scatter(
            x=ma50_series.index,
            y=ma50_series.values,
            mode="lines",
            name="MA 50",
            line=dict(width=1.5, dash="dot")
        )
    )

if metrics["ma200"] is not None:
    ma200_series = selected_close.rolling(200).mean()
    fig_single.add_trace(
        go.Scatter(
            x=ma200_series.index,
            y=ma200_series.values,
            mode="lines",
            name="MA 200",
            line=dict(width=1.5, dash="dash")
        )
    )

fig_single.update_layout(
    template="plotly_white",
    height=450,
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    title=f"{selected_ticker} Price and Trend Lines"
)
st.plotly_chart(fig_single, use_container_width=True)

st.divider()

st.header("2. Compare Stocks")

comparison_rows = []
for ticker in valid_tickers:
    close = close_table[ticker].dropna()
    m = compute_stock_metrics(close)
    comparison_rows.append({
        "Ticker": ticker,
        "Latest Close": round(m["latest_close"], 2),
        "1Y Return %": round(m["total_return"] * 100, 2),
        "Annual Volatility %": round(m["annual_volatility"] * 100, 2),
        "Max Drawdown %": round(m["max_drawdown"] * 100, 2),
        "Sharpe Ratio": round(m["sharpe"], 2) if m["sharpe"] is not None else None,
    })

comparison_df = pd.DataFrame(comparison_rows)
st.dataframe(comparison_df, use_container_width=True)

normalized = normalize_prices(close_table.dropna())

fig_compare = go.Figure()
for ticker in normalized.columns:
    fig_compare.add_trace(
        go.Scatter(
            x=normalized.index,
            y=normalized[ticker],
            mode="lines",
            name=ticker
        )
    )

fig_compare.update_layout(
    template="plotly_white",
    height=500,
    xaxis_title="Date",
    yaxis_title="Normalized Price (Base = 100)",
    title="Relative Performance Comparison"
)
st.plotly_chart(fig_compare, use_container_width=True)

st.divider()

st.header("3. Portfolio Builder")
st.write("Assign a weight to each stock. The weights should add up to 100%.")

weights = {}
weight_cols = st.columns(min(len(valid_tickers), 4) or 1)

for i, ticker in enumerate(valid_tickers):
    with weight_cols[i % len(weight_cols)]:
        weights[ticker] = st.number_input(
            f"{ticker} Weight %",
            min_value=0.0,
            max_value=100.0,
            value=round(100.0 / len(valid_tickers), 2),
            step=1.0,
            key=f"weight_{ticker}"
        )

total_weight = sum(weights.values())
st.write(f"Total Weight: {total_weight:.2f}%")

if abs(total_weight - 100.0) > 0.01:
    st.warning("Portfolio weights must add up to 100% to compute portfolio metrics.")
else:
    returns_df = close_table.pct_change().dropna()

    weight_vector = pd.Series({k: v / 100 for k, v in weights.items()})
    portfolio_daily_returns = returns_df[valid_tickers].mul(weight_vector, axis=1).sum(axis=1)

    portfolio_growth = (1 + portfolio_daily_returns).cumprod() * 100
    portfolio_total_return = portfolio_growth.iloc[-1] / portfolio_growth.iloc[0] - 1
    portfolio_volatility = portfolio_daily_returns.std() * math.sqrt(252)

    portfolio_running_max = portfolio_growth.cummax()
    portfolio_drawdown = (portfolio_growth / portfolio_running_max) - 1
    portfolio_max_drawdown = portfolio_drawdown.min()

    portfolio_sharpe = None
    if len(portfolio_daily_returns) > 1 and portfolio_daily_returns.std() != 0:
        portfolio_sharpe = (
            portfolio_daily_returns.mean() / portfolio_daily_returns.std()
        ) * math.sqrt(252)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Portfolio Return", f"{portfolio_total_return:.2%}")
    p2.metric("Portfolio Volatility", f"{portfolio_volatility:.2%}")
    p3.metric("Portfolio Max Drawdown", f"{portfolio_max_drawdown:.2%}")
    p4.metric("Portfolio Sharpe", f"{portfolio_sharpe:.2f}" if portfolio_sharpe is not None else "N/A")

    fig_port = go.Figure()
    fig_port.add_trace(
        go.Scatter(
            x=portfolio_growth.index,
            y=portfolio_growth.values,
            mode="lines",
            name="Portfolio"
        )
    )

    fig_port.update_layout(
        template="plotly_white",
        height=450,
        xaxis_title="Date",
        yaxis_title="Growth of $100",
        title="Portfolio Growth Over the Past 12 Months"
    )
    st.plotly_chart(fig_port, use_container_width=True)

if show_raw_data:
    st.divider()
    st.subheader("Raw Price Data")
    st.dataframe(close_table.tail(30), use_container_width=True)