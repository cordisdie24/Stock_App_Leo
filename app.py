import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy import stats

st.set_page_config(page_title="Stock Analyzer", layout="wide")

TRADING_DAYS = 252
BENCHMARK_TICKER = "^GSPC"
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA"]
FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "NFLX", "AMD",
    "INTC", "MU", "ADBE", "CRM", "ORCL", "QCOM", "TXN", "UBER", "SHOP", "CRWD",
    "PANW", "NOW", "PLTR", "ASML", "ARM", "SNOW", "ANET", "MELI", "AMAT", "LRCX",
    "KLAC", "MRVL", "CDNS", "SNPS", "TTD", "DDOG", "ZS", "MDB", "NET", "TEAM",
    "JPM", "GS", "MS", "BAC", "WFC", "BLK", "KKR", "BX", "V", "MA",
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "OXY", "HAL", "FANG",
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "ISRG", "ABT", "PFE", "TMO", "DHR",
    "COST", "WMT", "HD", "LOW", "MCD", "SBUX", "NKE", "DIS", "CMG", "BKNG",
    "CAT", "DE", "GE", "RTX", "BA", "HON", "ETN", "LMT", "NOC", "UNP",
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLI", "XLV", "XLY",
]
UNIVERSE_FILE_CANDIDATES = ["stock_universe.csv", "ticker_universe.csv", "us_stocks_universe.csv"]


@st.cache_data(ttl=3600, show_spinner="Fetching stock data...")
def load_data(symbols: tuple[str, ...], start: date, end: date) -> dict[str, pd.DataFrame]:
    """Download daily stock data from Yahoo Finance for one or more symbols."""
    data_map: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        if not symbol:
            continue

        try:
            df = yf.download(
                symbol,
                start=start,
                end=end + timedelta(days=1),
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            df = pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if not df.empty:
            df = df.sort_index()

        data_map[symbol] = df

    return data_map


@st.cache_data(ttl=3600, show_spinner=False)
def load_single_data(symbol: str, start: date, end: date) -> pd.DataFrame:
    if not symbol:
        return pd.DataFrame()

    try:
        df = yf.download(
            symbol,
            start=start,
            end=end + timedelta(days=1),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        df = pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if not df.empty:
        df = df.sort_index()

    return df


def validate_data(df: pd.DataFrame) -> bool:
    required_cols = {"Close", "Volume"}
    return not df.empty and required_cols.issubset(df.columns)


def enrich_single_asset(df: pd.DataFrame, ma_window: int, vol_window: int) -> pd.DataFrame:
    out = df.copy().dropna(subset=["Close"])
    out["Daily Return"] = out["Close"].pct_change()
    out["Cumulative Return"] = (1 + out["Daily Return"].fillna(0)).cumprod() - 1
    out[f"{ma_window}-Day MA"] = out["Close"].rolling(window=ma_window).mean()
    out["Rolling Volatility"] = out["Daily Return"].rolling(window=vol_window).std() * math.sqrt(TRADING_DAYS)
    return out


def compute_summary_stats(returns: pd.Series, risk_free_rate: float) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {
            "Total Return": float("nan"),
            "Avg Daily Return": float("nan"),
            "Annualized Return": float("nan"),
            "Annualized Volatility": float("nan"),
            "Sharpe Ratio": float("nan"),
            "Skewness": float("nan"),
            "Excess Kurtosis": float("nan"),
            "Jarque-Bera Stat": float("nan"),
            "Jarque-Bera p-value": float("nan"),
            "Observations": 0,
        }

    cumulative = (1 + returns).cumprod() - 1
    avg_daily_ret = float(returns.mean())
    volatility = float(returns.std())
    ann_volatility = volatility * math.sqrt(TRADING_DAYS) if not math.isnan(volatility) else float("nan")
    ann_return = avg_daily_ret * TRADING_DAYS if not math.isnan(avg_daily_ret) else float("nan")
    sharpe = (
        (ann_return - risk_free_rate) / ann_volatility
        if ann_volatility and not math.isnan(ann_volatility)
        else float("nan")
    )

    jb_stat, jb_pvalue = stats.jarque_bera(returns)

    return {
        "Total Return": float(cumulative.iloc[-1]),
        "Avg Daily Return": avg_daily_ret,
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_volatility,
        "Sharpe Ratio": sharpe,
        "Skewness": float(returns.skew()),
        "Excess Kurtosis": float(returns.kurtosis()),
        "Jarque-Bera Stat": float(jb_stat),
        "Jarque-Bera p-value": float(jb_pvalue),
        "Observations": int(returns.shape[0]),
    }


def format_metric(value: float, kind: str = "number") -> str:
    if pd.isna(value):
        return "N/A"
    if kind == "currency":
        return f"${value:,.2f}"
    if kind == "percent":
        return f"{value:.2%}"
    if kind == "small_percent":
        return f"{value:.4%}"
    if kind == "integer":
        return f"{int(value):,}"
    return f"{value:.2f}"


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else float("nan")


def compute_recommendation_features(df: pd.DataFrame, risk_free_rate: float) -> dict[str, float]:
    enriched = enrich_single_asset(df, ma_window=50, vol_window=30)
    returns = enriched["Daily Return"].dropna()
    close = enriched["Close"].dropna()

    if returns.empty or close.empty or len(close) < 30:
        return {}

    total_return = float(enriched["Cumulative Return"].iloc[-1])
    volatility = float(returns.std()) * math.sqrt(TRADING_DAYS)
    avg_daily = float(returns.mean())
    ann_return = avg_daily * TRADING_DAYS
    sharpe = (ann_return - risk_free_rate) / volatility if volatility and not math.isnan(volatility) else float("nan")
    momentum_20 = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 20 else float("nan")
    momentum_60 = float(close.iloc[-1] / close.iloc[-61] - 1) if len(close) > 60 else float("nan")
    ma_20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else float("nan")
    ma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else float("nan")
    rsi_14 = compute_rsi(close, 14)

    trend_score = 0.0
    if not math.isnan(ma_20):
        trend_score += 1.0 if close.iloc[-1] > ma_20 else -1.0
    if not math.isnan(ma_50):
        trend_score += 1.0 if close.iloc[-1] > ma_50 else -1.0

    rsi_score = 0.0
    if not math.isnan(rsi_14):
        if 45 <= rsi_14 <= 65:
            rsi_score = 1.0
        elif rsi_14 > 75:
            rsi_score = -1.0
        elif rsi_14 < 25:
            rsi_score = 0.5

    profitability_score = (
        0.30 * total_return
        + 0.25 * (0 if math.isnan(sharpe) else sharpe)
        + 0.20 * (0 if math.isnan(momentum_20) else momentum_20)
        + 0.15 * (0 if math.isnan(momentum_60) else momentum_60)
        + 0.10 * trend_score
        + 0.05 * rsi_score
        - 0.10 * (0 if math.isnan(volatility) else volatility)
    )

    return {
        "Latest Close": float(close.iloc[-1]),
        "Total Return": total_return,
        "Annualized Return": ann_return,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Momentum 20D": momentum_20,
        "Momentum 60D": momentum_60,
        "RSI 14": rsi_14,
        "Trend Score": trend_score,
        "Recommendation Score": profitability_score,
    }


@st.cache_data(ttl=3600, show_spinner="Scanning candidate universe...")
def build_program_suggestions(
    candidate_tickers: tuple[str, ...],
    start: date,
    end: date,
    risk_free_rate: float,
    max_scan: int,
) -> pd.DataFrame:
    selected = tuple(candidate_tickers[:max_scan])
    raw_map = load_data(selected, start, end)
    rows = []

    for ticker, df in raw_map.items():
        if not validate_data(df):
            continue
        features = compute_recommendation_features(df, risk_free_rate)
        if not features:
            continue
        rows.append({"Ticker": ticker, **features})

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Recommendation Score", "Latest Close"])
    return out.sort_values("Recommendation Score", ascending=False).reset_index(drop=True)


def compute_two_asset_portfolio(
    returns_df: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    weight_a: float,
) -> dict[str, object]:
    pair_df = returns_df[[asset_a, asset_b]].dropna().copy()
    weight_b = 1.0 - weight_a

    if pair_df.empty:
        return {
            "pair_returns": pd.Series(dtype=float),
            "covariance": float("nan"),
            "corr": float("nan"),
            "var_a": float("nan"),
            "var_b": float("nan"),
            "portfolio_variance": float("nan"),
            "portfolio_volatility": float("nan"),
            "weight_b": weight_b,
        }

    pair_df["Portfolio Return"] = weight_a * pair_df[asset_a] + weight_b * pair_df[asset_b]

    var_a = float(pair_df[asset_a].var())
    var_b = float(pair_df[asset_b].var())
    covariance = float(pair_df[[asset_a, asset_b]].cov().iloc[0, 1])
    corr = float(pair_df[[asset_a, asset_b]].corr().iloc[0, 1])

    portfolio_variance = (weight_a ** 2) * var_a + (weight_b ** 2) * var_b + 2 * weight_a * weight_b * covariance
    portfolio_volatility = math.sqrt(portfolio_variance) if portfolio_variance >= 0 else float("nan")

    return {
        "pair_returns": pair_df["Portfolio Return"],
        "covariance": covariance,
        "corr": corr,
        "var_a": var_a,
        "var_b": var_b,
        "portfolio_variance": portfolio_variance,
        "portfolio_volatility": portfolio_volatility,
        "weight_b": weight_b,
    }


def get_candidate_universe(target_size: int) -> tuple[list[str], str]:
    for file_name in UNIVERSE_FILE_CANDIDATES:
        path = Path(file_name)
        if path.exists():
            try:
                df = pd.read_csv(path)
                first_col = df.columns[0]
                tickers = [str(x).strip().upper() for x in df[first_col].dropna().tolist() if str(x).strip()]
                tickers = list(dict.fromkeys(tickers))
                if tickers:
                    return tickers[:target_size], f"Loaded {min(len(tickers), target_size):,} symbols from {file_name}."
            except Exception:
                pass

    fallback = list(dict.fromkeys(FALLBACK_UNIVERSE))
    return fallback[:min(target_size, len(fallback))], (
        "No local universe file was found, so the app is using a built-in fallback universe. "
        "For a true 10,000-symbol scan, add a CSV like stock_universe.csv with one ticker per row."
    )


st.title("Stock Analysis Dashboard")
st.caption("Analyze individual stocks, relative performance, and portfolio behavior in a cleaner dashboard layout.")
st.sidebar.header("Controls")

raw_tickers = st.sidebar.text_input("Tickers (comma-separated)", value=", ".join(DEFAULT_TICKERS))
user_tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
user_tickers = list(dict.fromkeys(user_tickers))

include_benchmark = st.sidebar.checkbox("Include S&P 500 Benchmark (^GSPC)", value=True)
show_program_suggestions = st.sidebar.checkbox("Show program-picked Up/Down suggestions", value=True)
requested_pool_size = st.sidebar.number_input("Requested candidate pool size", min_value=100, max_value=10000, value=10000, step=100)
max_live_scan = st.sidebar.slider("Live scan limit per run", min_value=50, max_value=500, value=200, step=25)

portfolio_tickers = list(user_tickers)
fetch_tickers = list(user_tickers)
if include_benchmark and BENCHMARK_TICKER not in fetch_tickers:
    fetch_tickers.append(BENCHMARK_TICKER)

default_start = date.today() - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=date.today())

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

ma_window = st.sidebar.slider("Moving Average Window (days)", min_value=5, max_value=200, value=50, step=5)

risk_free_rate = (
    st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=4.5, step=0.1) / 100
)

vol_window = st.sidebar.slider("Rolling Volatility Window (days)", min_value=10, max_value=120, value=30, step=5)
corr_window = st.sidebar.slider("Rolling Correlation Window (days)", min_value=10, max_value=120, value=30, step=5)
qq_sample_size = st.sidebar.slider("Q-Q Plot Sample Size", min_value=50, max_value=500, value=250, step=25)

if not portfolio_tickers:
    st.info("Enter at least one stock ticker in the sidebar to get started.")
    st.stop()

try:
    raw_data = load_data(tuple(fetch_tickers), start_date, end_date)
except Exception as e:
    st.error("Failed to download stock data.")
    st.exception(e)
    st.stop()

asset_data: dict[str, pd.DataFrame] = {}
invalid_tickers: list[str] = []

for ticker in portfolio_tickers:
    df = raw_data.get(ticker, pd.DataFrame())
    if validate_data(df):
        asset_data[ticker] = enrich_single_asset(df, ma_window, vol_window)
    else:
        invalid_tickers.append(ticker)

if invalid_tickers:
    st.warning(f"No data found for: {', '.join(invalid_tickers)}")

if not asset_data:
    st.error("No valid stock data was returned. Try different tickers or a wider date range.")
    st.stop()

benchmark_df = pd.DataFrame()
if include_benchmark:
    benchmark_raw = raw_data.get(BENCHMARK_TICKER, pd.DataFrame())
    if validate_data(benchmark_raw):
        benchmark_df = enrich_single_asset(benchmark_raw, ma_window, vol_window)
    else:
        st.warning("S&P 500 benchmark data could not be loaded for this period.")

primary_ticker = next(iter(asset_data.keys()))
primary_df = asset_data[primary_ticker]

return_df = pd.concat([df["Daily Return"].rename(ticker) for ticker, df in asset_data.items()], axis=1).dropna(how="all")
cumulative_df = (1 + return_df.fillna(0)).cumprod() - 1

portfolio_returns = return_df.copy().dropna(how="all")
portfolio_returns["Equal Weight Portfolio"] = portfolio_returns.mean(axis=1, skipna=True)
portfolio_cumulative = (1 + portfolio_returns["Equal Weight Portfolio"].fillna(0)).cumprod() - 1
portfolio_vol = portfolio_returns["Equal Weight Portfolio"].rolling(window=vol_window).std() * math.sqrt(TRADING_DAYS)

summary_rows = []
for ticker, df in asset_data.items():
    stats_map = compute_summary_stats(df["Daily Return"], risk_free_rate)
    stats_map.update(
        {
            "Ticker": ticker,
            "Latest Close": float(df["Close"].iloc[-1]),
            "Period High": float(df["Close"].max()),
            "Period Low": float(df["Close"].min()),
        }
    )
    summary_rows.append(stats_map)

portfolio_stats = compute_summary_stats(portfolio_returns["Equal Weight Portfolio"], risk_free_rate)
portfolio_stats.update({"Ticker": "Equal Weight Portfolio", "Latest Close": float("nan"), "Period High": float("nan"), "Period Low": float("nan")})
summary_rows.append(portfolio_stats)

if not benchmark_df.empty:
    benchmark_stats = compute_summary_stats(benchmark_df["Daily Return"], risk_free_rate)
    benchmark_stats.update(
        {
            "Ticker": "^GSPC",
            "Latest Close": float(benchmark_df["Close"].iloc[-1]),
            "Period High": float(benchmark_df["Close"].max()),
            "Period Low": float(benchmark_df["Close"].min()),
        }
    )
    summary_rows.append(benchmark_stats)

summary_df = pd.DataFrame(summary_rows).set_index("Ticker")

latest_close = float(primary_df["Close"].iloc[-1])
primary_stats = compute_summary_stats(primary_df["Daily Return"], risk_free_rate)
max_close = float(primary_df["Close"].max())
min_close = float(primary_df["Close"].min())

st.subheader(f"{primary_ticker} — Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Close", format_metric(latest_close, "currency"))
col2.metric("Total Return", format_metric(primary_stats["Total Return"], "percent"))
col3.metric("Annualized Return", format_metric(primary_stats["Annualized Return"], "percent"))
col4.metric("Sharpe Ratio", format_metric(primary_stats["Sharpe Ratio"]))

col5, col6, col7, col8 = st.columns(4)
col5.metric("Annualized Volatility (sigma)", format_metric(primary_stats["Annualized Volatility"], "percent"))
col6.metric("Skewness", format_metric(primary_stats["Skewness"]))
col7.metric("Excess Kurtosis", format_metric(primary_stats["Excess Kurtosis"]))
col8.metric("Avg Daily Return", format_metric(primary_stats["Avg Daily Return"], "small_percent"))

col9, col10, col11, col12 = st.columns(4)
col9.metric("Period High", format_metric(max_close, "currency"))
col10.metric("Period Low", format_metric(min_close, "currency"))
col11.metric("Jarque-Bera Stat", format_metric(primary_stats["Jarque-Bera Stat"]))
col12.metric("Jarque-Bera p-value", format_metric(primary_stats["Jarque-Bera p-value"]))

st.divider()

if show_program_suggestions:
    with st.expander("Program-Picked Suggestions", expanded=False):
        candidate_universe, universe_message = get_candidate_universe(int(requested_pool_size))
        st.caption(universe_message)
        st.caption(
            f"Requested pool size: {int(requested_pool_size):,}. Live scan limit this run: {max_live_scan:,}. "
            "This keeps the app responsive and avoids Yahoo rate limits."
        )

        suggestion_df = build_program_suggestions(tuple(candidate_universe), start_date, end_date, risk_free_rate, max_live_scan)

        if suggestion_df.empty:
            st.warning("No program-generated suggestions were available for the current scan settings.")
        else:
            up_col, down_col = st.columns(2)
            with up_col:
                st.markdown("#### Top 10 highest recommendation scores")
                top_up = suggestion_df.head(10).copy()
                top_up_display = top_up[["Ticker", "Recommendation Score", "Sharpe Ratio", "Momentum 20D", "Annualized Return"]].copy()
                top_up_display["Recommendation Score"] = top_up_display["Recommendation Score"].map(format_metric)
                top_up_display["Sharpe Ratio"] = top_up_display["Sharpe Ratio"].map(format_metric)
                top_up_display["Momentum 20D"] = top_up_display["Momentum 20D"].map(lambda x: format_metric(x, "percent"))
                top_up_display["Annualized Return"] = top_up_display["Annualized Return"].map(lambda x: format_metric(x, "percent"))
                st.dataframe(top_up_display, use_container_width=True, height=390)

            with down_col:
                st.markdown("#### Top 10 lowest recommendation scores")
                top_down = suggestion_df.tail(10).sort_values("Recommendation Score", ascending=True).copy()
                top_down_display = top_down[["Ticker", "Recommendation Score", "Sharpe Ratio", "Momentum 20D", "Annualized Return"]].copy()
                top_down_display["Recommendation Score"] = top_down_display["Recommendation Score"].map(format_metric)
                top_down_display["Sharpe Ratio"] = top_down_display["Sharpe Ratio"].map(format_metric)
                top_down_display["Momentum 20D"] = top_down_display["Momentum 20D"].map(lambda x: format_metric(x, "percent"))
                top_down_display["Annualized Return"] = top_down_display["Annualized Return"].map(lambda x: format_metric(x, "percent"))
                st.dataframe(top_down_display, use_container_width=True, height=390)

st.divider()
st.markdown("## Market View")
st.subheader("Price & Moving Average")
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=primary_df.index, y=primary_df["Close"], mode="lines", name=f"{primary_ticker} Close", line=dict(width=2)))
fig_price.add_trace(go.Scatter(x=primary_df.index, y=primary_df[f"{ma_window}-Day MA"], mode="lines", name=f"{ma_window}-Day MA", line=dict(width=2, dash="dash")))
fig_price.update_layout(template="plotly_white", height=450, xaxis_title="Date", yaxis_title="Price (USD)", margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig_price, use_container_width=True)

if ma_window > len(primary_df):
    st.warning(f"The selected {ma_window}-day window is longer than the available data ({len(primary_df)} trading days). The moving average line may not appear.")

st.divider()
st.markdown("## Performance and Risk")
perf_left, perf_right = st.columns(2)

with perf_left:
    st.subheader("Daily Trading Volume")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=primary_df.index, y=primary_df["Volume"], name="Volume", opacity=0.7))
    fig_vol.update_layout(template="plotly_white", height=350, xaxis_title="Date", yaxis_title="Shares Traded", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_vol, use_container_width=True)

with perf_right:
    st.subheader("Distribution of Daily Returns")
    returns_clean = primary_df["Daily Return"].dropna()
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=returns_clean, nbinsx=60, opacity=0.75, name="Daily Returns", histnorm="probability density"))
    if not returns_clean.empty:
        x_range = np.linspace(float(returns_clean.min()), float(returns_clean.max()), 200)
        mu = float(returns_clean.mean())
        sigma = float(returns_clean.std())
        if sigma > 0:
            fig_hist.add_trace(go.Scatter(x=x_range, y=stats.norm.pdf(x_range, mu, sigma), mode="lines", name="Normal Distribution", line=dict(width=2)))
    fig_hist.update_layout(template="plotly_white", height=350, xaxis_title="Daily Return", yaxis_title="Density", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(f"Jarque-Bera test for {primary_ticker}: statistic = {primary_stats['Jarque-Bera Stat']:.2f}, p-value = {primary_stats['Jarque-Bera p-value']:.4f}")

trend_left, trend_right = st.columns(2)

with trend_left:
    st.subheader("Cumulative Return Over Time")
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=primary_df.index, y=primary_df["Cumulative Return"], mode="lines", name="Cumulative Return", fill="tozeroy"))
    fig_cum.update_layout(template="plotly_white", height=400, xaxis_title="Date", yaxis_title="Cumulative Return", yaxis_tickformat=".0%", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_cum, use_container_width=True)

with trend_right:
    st.subheader("Rolling Annualized Volatility")
    fig_roll_vol = go.Figure()
    fig_roll_vol.add_trace(go.Scatter(x=primary_df.index, y=primary_df["Rolling Volatility"], mode="lines", name=f"{vol_window}-Day Rolling Vol", line=dict(width=1.5)))
    fig_roll_vol.add_trace(go.Scatter(x=portfolio_vol.index, y=portfolio_vol, mode="lines", name="Portfolio Rolling Vol", line=dict(width=2, dash="dash")))
    fig_roll_vol.update_layout(template="plotly_white", height=400, xaxis_title="Date", yaxis_title="Annualized Volatility", yaxis_tickformat=".0%", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_roll_vol, use_container_width=True)

st.divider()
st.markdown("## Relative Performance")
st.subheader("Multi-Stock Cumulative Return Comparison")
fig_compare = go.Figure()
for ticker in cumulative_df.columns:
    fig_compare.add_trace(go.Scatter(x=cumulative_df.index, y=cumulative_df[ticker], mode="lines", name=ticker))
if not benchmark_df.empty:
    fig_compare.add_trace(go.Scatter(x=benchmark_df.index, y=benchmark_df["Cumulative Return"], mode="lines", name="^GSPC Benchmark", line=dict(dash="dot")))
fig_compare.add_trace(go.Scatter(x=portfolio_cumulative.index, y=portfolio_cumulative, mode="lines", name="Equal Weight Portfolio", line=dict(width=3)))
fig_compare.update_layout(template="plotly_white", height=450, xaxis_title="Date", yaxis_title="Cumulative Return", yaxis_tickformat=".0%", margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig_compare, use_container_width=True)

st.subheader("Summary Statistics Table")
summary_display = summary_df.copy()
for col in ["Total Return", "Avg Daily Return", "Annualized Return", "Annualized Volatility"]:
    summary_display[col] = summary_display[col].map(lambda x: format_metric(x, "small_percent") if col == "Avg Daily Return" else format_metric(x, "percent"))
summary_display["Jarque-Bera p-value"] = summary_display["Jarque-Bera p-value"].map(format_metric)
summary_display["Sharpe Ratio"] = summary_display["Sharpe Ratio"].map(format_metric)
summary_display["Skewness"] = summary_display["Skewness"].map(format_metric)
summary_display["Excess Kurtosis"] = summary_display["Excess Kurtosis"].map(format_metric)
summary_display["Jarque-Bera Stat"] = summary_display["Jarque-Bera Stat"].map(format_metric)
summary_display["Observations"] = summary_display["Observations"].map(lambda x: format_metric(x, "integer"))
summary_display["Latest Close"] = summary_display["Latest Close"].map(lambda x: format_metric(x, "currency"))
summary_display["Period High"] = summary_display["Period High"].map(lambda x: format_metric(x, "currency"))
summary_display["Period Low"] = summary_display["Period Low"].map(lambda x: format_metric(x, "currency"))
st.dataframe(summary_display, use_container_width=True, height=330)

st.divider()
st.markdown("## Portfolio Analysis")
st.subheader("Equal-Weight Portfolio Explorer")
port_col1, port_col2 = st.columns(2)
with port_col1:
    st.metric("Portfolio Total Return", format_metric(portfolio_stats["Total Return"], "percent"))
    st.metric("Portfolio Annualized Return", format_metric(portfolio_stats["Annualized Return"], "percent"))
    st.metric("Portfolio Sharpe Ratio", format_metric(portfolio_stats["Sharpe Ratio"]))
with port_col2:
    st.metric("Portfolio Annualized Volatility", format_metric(portfolio_stats["Annualized Volatility"], "percent"))
    st.metric("Portfolio Skewness", format_metric(portfolio_stats["Skewness"]))
    st.metric("Portfolio Jarque-Bera p-value", format_metric(portfolio_stats["Jarque-Bera p-value"]))

weights_df = pd.DataFrame({"Ticker": list(asset_data.keys()), "Weight": [1 / len(asset_data)] * len(asset_data)})
weights_df["Weight"] = weights_df["Weight"].map(lambda x: f"{x:.2%}")
st.dataframe(weights_df, use_container_width=True, height=220)

fig_portfolio = go.Figure()
fig_portfolio.add_trace(go.Scatter(x=portfolio_cumulative.index, y=portfolio_cumulative, mode="lines", name="Equal Weight Portfolio", line=dict(width=3)))
if not benchmark_df.empty:
    fig_portfolio.add_trace(go.Scatter(x=benchmark_df.index, y=benchmark_df["Cumulative Return"], mode="lines", name="^GSPC Benchmark", line=dict(dash="dot")))
fig_portfolio.update_layout(template="plotly_white", height=400, xaxis_title="Date", yaxis_title="Cumulative Return", yaxis_tickformat=".0%", margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig_portfolio, use_container_width=True)

st.subheader("Two-Asset Portfolio Explorer")
valid_two_asset = len(return_df.columns) >= 2
if not valid_two_asset:
    st.info("Add at least two valid tickers to use the Two-Asset Portfolio Explorer.")
else:
    two_col1, two_col2, two_col3 = st.columns(3)
    asset_options = list(return_df.columns)
    asset_a = two_col1.selectbox("Asset 1", asset_options, index=0)
    asset_b = two_col2.selectbox("Asset 2", asset_options, index=1 if len(asset_options) > 1 else 0)
    weight_a_pct = two_col3.slider("Weight in Asset 1 (%)", min_value=0, max_value=100, value=50, step=5)

    if asset_a == asset_b:
        st.error("Asset 1 and Asset 2 must be different.")
    else:
        weight_a = weight_a_pct / 100
        two_asset = compute_two_asset_portfolio(return_df, asset_a, asset_b, weight_a)
        weight_b = two_asset["weight_b"]

        if two_asset["pair_returns"].empty:
            st.error("Not enough overlapping return data to build the two-asset portfolio.")
        else:
            pair_stats = compute_summary_stats(two_asset["pair_returns"], risk_free_rate)
            pair_cumulative = (1 + two_asset["pair_returns"].fillna(0)).cumprod() - 1

            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Weight in Asset 1", f"{weight_a:.0%}")
            t2.metric("Weight in Asset 2", f"{weight_b:.0%}")
            t3.metric("Portfolio Variance", f"{two_asset['portfolio_variance']:.8f}")
            t4.metric("Portfolio Volatility", f"{two_asset['portfolio_volatility']:.4%}")

            t5, t6, t7, t8 = st.columns(4)
            t5.metric("Covariance", f"{two_asset['covariance']:.8f}")
            t6.metric("Correlation", f"{two_asset['corr']:.2f}")
            t7.metric("Two-Asset Total Return", format_metric(pair_stats["Total Return"], "percent"))
            t8.metric("Two-Asset Sharpe", format_metric(pair_stats["Sharpe Ratio"]))

            st.markdown(
                f"**Portfolio variance formula:**  \n"
                f"σ²ₚ = w₁²σ₁² + w₂²σ₂² + 2w₁w₂Cov(R₁,R₂)  \n"
                f"= ({weight_a:.2f}² × {two_asset['var_a']:.8f}) + ({weight_b:.2f}² × {two_asset['var_b']:.8f}) + 2 × {weight_a:.2f} × {weight_b:.2f} × {two_asset['covariance']:.8f}  \n"
                f"= **{two_asset['portfolio_variance']:.8f}**"
            )

            fig_two_asset = go.Figure()
            fig_two_asset.add_trace(go.Scatter(x=pair_cumulative.index, y=pair_cumulative, mode="lines", name="Two-Asset Portfolio", line=dict(width=3)))
            fig_two_asset.add_trace(go.Scatter(x=cumulative_df.index, y=cumulative_df[asset_a], mode="lines", name=asset_a, line=dict(dash="dash")))
            fig_two_asset.add_trace(go.Scatter(x=cumulative_df.index, y=cumulative_df[asset_b], mode="lines", name=asset_b, line=dict(dash="dot")))
            fig_two_asset.update_layout(template="plotly_white", height=420, xaxis_title="Date", yaxis_title="Cumulative Return", yaxis_tickformat=".0%", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_two_asset, use_container_width=True)

st.divider()
st.markdown("## Statistical Diagnostics")
diag_left, diag_right = st.columns(2)

with diag_left:
    st.subheader("Q-Q Plot")
    qq_returns = returns_clean.tail(min(len(returns_clean), qq_sample_size))
    fig_qq = go.Figure()
    if len(qq_returns) > 1:
        osm, osr = stats.probplot(qq_returns, dist="norm", fit=False)
        fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Observed Returns"))
        slope, intercept, _ = stats.probplot(qq_returns, dist="norm", fit=True)[1]
        x_line = np.linspace(min(osm), max(osm), 100)
        y_line = slope * x_line + intercept
        fig_qq.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Reference Line"))
    fig_qq.update_layout(template="plotly_white", height=400, xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_qq, use_container_width=True)

with diag_right:
    if len(return_df.columns) >= 2:
        st.subheader("Correlation Heatmap")
        corr_matrix = return_df.dropna(how="all").corr()
        fig_heat = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index, zmin=-1, zmax=1, text=np.round(corr_matrix.values, 2), texttemplate="%{text}"))
        fig_heat.update_layout(template="plotly_white", height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

st.subheader("Rolling Correlation")
    corr_options = list(return_df.columns)
    corr_a = st.selectbox("First ticker", corr_options, index=0, key="corr_a")
    corr_b = st.selectbox("Second ticker", corr_options, index=1 if len(corr_options) > 1 else 0, key="corr_b")

    if corr_a == corr_b:
        st.info("Choose two different tickers to view rolling correlation.")
    else:
        rolling_corr = return_df[corr_a].rolling(window=corr_window).corr(return_df[corr_b])
        fig_roll_corr = go.Figure()
        fig_roll_corr.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, mode="lines", name=f"{corr_a} vs {corr_b}", line=dict(width=2)))
        fig_roll_corr.update_layout(template="plotly_white", height=400, xaxis_title="Date", yaxis_title="Rolling Correlation", yaxis=dict(range=[-1, 1]), margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_roll_corr, use_container_width=True)

st.divider()
st.markdown("## Data Access")
st.subheader("Raw Data")
with st.expander(f"View Raw Data for {primary_ticker}"):
    st.dataframe(primary_df.tail(60), use_container_width=True)

csv_data = primary_df.to_csv().encode("utf-8")
st.download_button(label=f"Download {primary_ticker} data as CSV", data=csv_data, file_name=f"{primary_ticker.lower()}_stock_data.csv", mime="text/csv")
