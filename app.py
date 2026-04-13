import math
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Stock Analyzer", layout="wide")

st.title("Stock Analysis Dashboard")
st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").strip().upper()

default_start = date.today() - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=date.today())

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

ma_window = st.sidebar.slider(
    "Moving Average Window (days)",
    min_value=5,
    max_value=200,
    value=50,
    step=5,
)

risk_free_rate = (
    st.sidebar.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=4.5,
        step=0.1,
    )
    / 100
)

vol_window = st.sidebar.slider(
    "Rolling Volatility Window (days)",
    min_value=10,
    max_value=120,
    value=30,
    step=5,
)


@st.cache_data(ttl=3600, show_spinner="Fetching stock data...")
def load_data(symbol: str, start: date, end: date) -> pd.DataFrame:
    """
    Download daily stock data from Yahoo Finance for a given date range.
    """
    if not symbol:
        return pd.DataFrame()

    df = yf.download(
        symbol,
        start=start,
        end=end + timedelta(days=1),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def validate_data(df: pd.DataFrame) -> bool:
    required_cols = {"Close", "Volume"}
    return not df.empty and required_cols.issubset(df.columns)


if not ticker:
    st.info("Enter a stock ticker in the sidebar to get started.")
    st.stop()

try:
    df = load_data(ticker, start_date, end_date)
except Exception as e:
    st.error("Failed to download stock data.")
    st.exception(e)
    st.stop()

if not validate_data(df):
    st.error(f"No data found for {ticker}. Try another ticker symbol or date range.")
    st.stop()

df = df.copy().dropna(subset=["Close"])
df["Daily Return"] = df["Close"].pct_change()
df["Cumulative Return"] = (1 + df["Daily Return"].fillna(0)).cumprod() - 1
df[f"{ma_window}-Day MA"] = df["Close"].rolling(window=ma_window).mean()
df["Rolling Volatility"] = df["Daily Return"].rolling(vol_window).std() * math.sqrt(252)

latest_close = float(df["Close"].iloc[-1])
total_return = float(df["Cumulative Return"].iloc[-1])
avg_daily_ret = float(df["Daily Return"].mean())
volatility = float(df["Daily Return"].std())
ann_volatility = volatility * math.sqrt(252)
ann_return = avg_daily_ret * 252
sharpe = (ann_return - risk_free_rate) / ann_volatility if ann_volatility and not math.isnan(ann_volatility) else float("nan")
skewness = float(df["Daily Return"].skew())
kurtosis = float(df["Daily Return"].kurtosis())
max_close = float(df["Close"].max())
min_close = float(df["Close"].min())

st.subheader(f"{ticker} Key Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Close", f"${latest_close:,.2f}")
col2.metric("Total Return", f"{total_return:.2%}")
col3.metric("Annualized Return", f"{ann_return:.2%}")
col4.metric("Sharpe Ratio", "N/A" if math.isnan(sharpe) else f"{sharpe:.2f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Annualized Volatility", f"{ann_volatility:.2%}")
col6.metric("Skewness", f"{skewness:.2f}")
col7.metric("Excess Kurtosis", f"{kurtosis:.2f}")
col8.metric("Avg Daily Return", f"{avg_daily_ret:.4%}")

col9, col10, _, _ = st.columns(4)
col9.metric("Period High", f"${max_close:,.2f}")
col10.metric("Period Low", f"${min_close:,.2f}")

st.divider()

st.subheader("Price and Moving Average")

fig_price = go.Figure()
fig_price.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Close Price",
        line=dict(width=2),
    )
)
fig_price.add_trace(
    go.Scatter(
        x=df.index,
        y=df[f"{ma_window}-Day MA"],
        mode="lines",
        name=f"{ma_window}-Day MA",
        line=dict(width=2, dash="dash"),
    )
)
fig_price.update_layout(
    template="plotly_white",
    height=450,
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    margin=dict(l=20, r=20, t=40, b=20),
)
st.plotly_chart(fig_price, use_container_width=True)

if ma_window > len(df):
    st.warning(
        f"The selected {ma_window}-day window is longer than the available data "
        f"({len(df)} trading days). The moving average line may not appear."
    )

st.subheader("Cumulative Return Over Time")

fig_cum = go.Figure()
fig_cum.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Cumulative Return"],
        mode="lines",
        name="Cumulative Return",
        fill="tozeroy",
    )
)
fig_cum.update_layout(
    template="plotly_white",
    height=400,
    xaxis_title="Date",
    yaxis_title="Cumulative Return",
    yaxis_tickformat=".0%",
    margin=dict(l=20, r=20, t=40, b=20),
)
st.plotly_chart(fig_cum, use_container_width=True)

st.subheader("Daily Trading Volume")

fig_vol = go.Figure()
fig_vol.add_trace(
    go.Bar(
        x=df.index,
        y=df["Volume"],
        name="Volume",
        opacity=0.7,
    )
)
fig_vol.update_layout(
    template="plotly_white",
    height=350,
    xaxis_title="Date",
    yaxis_title="Shares Traded",
    margin=dict(l=20, r=20, t=40, b=20),
)
st.plotly_chart(fig_vol, use_container_width=True)

st.subheader("Distribution of Daily Returns")

fig_hist = go.Figure()
fig_hist.add_trace(
    go.Histogram(
        x=df["Daily Return"].dropna(),
        nbinsx=60,
        opacity=0.75,
        name="Daily Returns",
    )
)
fig_hist.update_layout(
    template="plotly_white",
    height=350,
    xaxis_title="Daily Return",
    yaxis_title="Frequency",
    margin=dict(l=20, r=20, t=40, b=20),
)
st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("Rolling Annualized Volatility")

fig_roll_vol = go.Figure()
fig_roll_vol.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Rolling Volatility"],
        mode="lines",
        name=f"{vol_window}-Day Rolling Vol",
        line=dict(width=1.5),
    )
)
fig_roll_vol.update_layout(
    template="plotly_white",
    height=400,
    xaxis_title="Date",
    yaxis_title="Annualized Volatility",
    yaxis_tickformat=".0%",
    margin=dict(l=20, r=20, t=40, b=20),
)
st.plotly_chart(fig_roll_vol, use_container_width=True)

with st.expander("Show raw data"):
    st.dataframe(df.tail(50), use_container_width=True)