import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Stock Analyzer", layout="wide")

st.title("Stock Analysis Dashboard")
st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").strip().upper()


@st.cache_data(ttl=3600, show_spinner="Fetching stock data...")
def load_data(symbol: str) -> pd.DataFrame:
    """
    Download 1 year of daily stock data from Yahoo Finance.
    Uses period instead of start/end because it is usually more reliable
    on cloud deployments.
    """
    if not symbol:
        return pd.DataFrame()

    df = yf.download(
        symbol,
        period="1y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def validate_data(df: pd.DataFrame) -> bool:
    """Check that the DataFrame is usable for the dashboard."""
    required_cols = {"Close"}
    return not df.empty and required_cols.issubset(df.columns)


if not ticker:
    st.info("Enter a stock ticker in the sidebar to get started.")
    st.stop()

try:
    df = load_data(ticker)
except Exception as e:
    st.error("Failed to download stock data.")
    st.exception(e)
    st.stop()

if not validate_data(df):
    st.error(f"No data found for {ticker}. Try another ticker symbol.")
    st.stop()

df = df.copy()
df["Daily Return"] = df["Close"].pct_change()

latest_close = float(df["Close"].iloc[-1])
first_close = float(df["Close"].iloc[0])
total_return = (latest_close / first_close) - 1
daily_volatility = float(df["Daily Return"].std())
annual_volatility = daily_volatility * math.sqrt(252)
max_close = float(df["Close"].max())
min_close = float(df["Close"].min())

st.subheader(f"{ticker} Key Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Latest Close", f"${latest_close:,.2f}")
col2.metric("1 Year Return", f"{total_return:.2%}")
col3.metric("Annualized Volatility", f"{annual_volatility:.2%}")

col4, col5 = st.columns(2)
col4.metric("12 Month High", f"${max_close:,.2f}")
col5.metric("12 Month Low", f"${min_close:,.2f}")

st.divider()
st.subheader(f"{ticker} Closing Price Over the Past 12 Months")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Close Price",
        line=dict(width=2),
    )
)

fig.update_layout(
    template="plotly_white",
    height=500,
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    margin=dict(l=20, r=20, t=40, b=20),
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("Show raw data"):
    st.dataframe(df.tail(20), use_container_width=True)
    