import math
from datetime import date, timedelta

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


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero-box {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: white;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        opacity: 0.85;
    }
    .signal-box {
        padding: 1rem 1.2rem;
        border-radius: 18px;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.10);
    }
    .signal-buy {
        background: linear-gradient(135deg, rgba(16,185,129,0.22), rgba(5,150,105,0.10));
    }
    .signal-hold {
        background: linear-gradient(135deg, rgba(245,158,11,0.22), rgba(217,119,6,0.10));
    }
    .signal-sell {
        background: linear-gradient(135deg, rgba(239,68,68,0.22), rgba(185,28,28,0.10));
    }
    .signal-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.8;
    }
    .signal-value {
        font-size: 1.9rem;
        font-weight: 800;
        margin-top: 0.15rem;
        margin-bottom: 0.25rem;
    }
    .signal-reason {
        font-size: 0.95rem;
        opacity: 0.92;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 0.3rem;
    }
    .small-note {
        color: #94a3b8;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner="Fetching stock data...")
def load_data(symbols: tuple[str, ...], start: date, end: date) -> dict[str, pd.DataFrame]:
    data_map: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        if not symbol:
            continue

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

        if not df.empty:
            df = df.sort_index()

        data_map[symbol] = df

    return data_map


def validate_data(df: pd.DataFrame) -> bool:
    required_cols = {"Close", "Volume"}
    return not df.empty and required_cols.issubset(df.columns)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def enrich_single_asset(df: pd.DataFrame, ma_window: int, vol_window: int) -> pd.DataFrame:
    out = df.copy().dropna(subset=["Close"])
    out["Daily Return"] = out["Close"].pct_change()
    out["Cumulative Return"] = (1 + out["Daily Return"].fillna(0)).cumprod() - 1
    out[f"{ma_window}-Day MA"] = out["Close"].rolling(window=ma_window).mean()
    out["20-Day MA"] = out["Close"].rolling(window=20).mean()
    out["50-Day MA"] = out["Close"].rolling(window=50).mean()
    out["Rolling Volatility"] = out["Daily Return"].rolling(window=vol_window).std() * math.sqrt(TRADING_DAYS)
    out["RSI"] = compute_rsi(out["Close"], period=14)
    out["Momentum"] = out["Close"] - out["Close"].shift(20)
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


def generate_recommendation(df: pd.DataFrame, stats_map: dict[str, float]) -> dict[str, object]:
    if df.empty or len(df) < 50:
        return {
            "Recommendation": "Hold",
            "Score": 0,
            "Reasons": ["Not enough data to generate a stronger signal."],
            "RSI": float("nan"),
            "Momentum": float("nan"),
        }

    latest_close = float(df["Close"].iloc[-1])
    ma20 = float(df["20-Day MA"].iloc[-1]) if not pd.isna(df["20-Day MA"].iloc[-1]) else float("nan")
    ma50 = float(df["50-Day MA"].iloc[-1]) if not pd.isna(df["50-Day MA"].iloc[-1]) else float("nan")
    rsi = float(df["RSI"].iloc[-1]) if not pd.isna(df["RSI"].iloc[-1]) else float("nan")
    momentum = float(df["Momentum"].iloc[-1]) if not pd.isna(df["Momentum"].iloc[-1]) else float("nan")
    annual_return = float(stats_map.get("Annualized Return", float("nan")))
    annual_vol = float(stats_map.get("Annualized Volatility", float("nan")))
    sharpe = float(stats_map.get("Sharpe Ratio", float("nan")))

    score = 0
    reasons = []

    if not pd.isna(annual_return):
        if annual_return > 0:
            score += 1
            reasons.append("Annualized return is positive.")
        elif annual_return < 0:
            score -= 1
            reasons.append("Annualized return is negative.")

    if not pd.isna(annual_vol):
        if annual_vol < 0.25:
            score += 1
            reasons.append("Volatility is relatively controlled.")
        elif annual_vol > 0.40:
            score -= 1
            reasons.append("Volatility is high.")

    if not pd.isna(sharpe):
        if sharpe > 1:
            score += 1
            reasons.append("Sharpe ratio is strong.")
        elif sharpe < 0:
            score -= 1
            reasons.append("Sharpe ratio is negative.")

    if not pd.isna(ma20) and not pd.isna(ma50):
        if latest_close > ma20 and ma20 > ma50:
            score += 1
            reasons.append("Price is above the short and medium trend lines.")
        elif latest_close < ma20 and ma20 < ma50:
            score -= 1
            reasons.append("Price is below the short and medium trend lines.")

    if not pd.isna(rsi):
        if rsi < 30:
            score += 1
            reasons.append("RSI suggests the stock may be oversold.")
        elif rsi > 70:
            score -= 1
            reasons.append("RSI suggests the stock may be overbought.")

    if not pd.isna(momentum):
        if momentum > 0:
            score += 1
            reasons.append("Momentum is positive.")
        elif momentum < 0:
            score -= 1
            reasons.append("Momentum is negative.")

    if score >= 3:
        recommendation = "Buy"
    elif score <= -3:
        recommendation = "Sell"
    else:
        recommendation = "Hold"

    return {
        "Recommendation": recommendation,
        "Score": score,
        "Reasons": reasons,
        "RSI": rsi,
        "Momentum": momentum,
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


def signal_class(signal: str) -> str:
    if signal == "Buy":
        return "signal-box signal-buy"
    if signal == "Sell":
        return "signal-box signal-sell"
    return "signal-box signal-hold"


st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">Stock Analysis Dashboard</div>
        <div class="hero-subtitle">Yahoo Finance powered dashboard with performance analytics, technical indicators, portfolio views, and a rule-based recommendation engine.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Controls")

raw_tickers = st.sidebar.text_input("Tickers (comma-separated)", value=", ".join(DEFAULT_TICKERS))
user_tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
user_tickers = list(dict.fromkeys(user_tickers))

include_benchmark = st.sidebar.checkbox("Include S&P 500 Benchmark (^GSPC)", value=True)
show_up = st.sidebar.toggle("Up", value=True)
show_down = st.sidebar.toggle("Down", value=True)

default_start = date.today() - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=date.today())

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

ma_window = st.sidebar.slider("Moving Average Window (days)", 5, 200, 50, 5)

risk_free_rate = (
    st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=4.5, step=0.1) / 100
)

vol_window = st.sidebar.slider("Rolling Volatility Window (days)", 10, 120, 30, 5)
corr_window = st.sidebar.slider("Rolling Correlation Window (days)", 10, 120, 30, 5)
qq_sample_size = st.sidebar.slider("Q-Q Plot Sample Size", 50, 500, 250, 25)

portfolio_tickers = list(user_tickers)
fetch_tickers = list(user_tickers)
if include_benchmark and BENCHMARK_TICKER not in fetch_tickers:
    fetch_tickers.append(BENCHMARK_TICKER)

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

price_df = pd.concat([df["Close"].rename(ticker) for ticker, df in asset_data.items()], axis=1).dropna(how="all")
return_df = pd.concat([df["Daily Return"].rename(ticker) for ticker, df in asset_data.items()], axis=1).dropna(how="all")
cumulative_df = (1 + return_df.fillna(0)).cumprod() - 1

portfolio_returns = return_df.copy().dropna(how="all")
portfolio_returns["Equal Weight Portfolio"] = portfolio_returns.mean(axis=1, skipna=True)
portfolio_cumulative = (1 + portfolio_returns["Equal Weight Portfolio"].fillna(0)).cumprod() - 1
portfolio_vol = portfolio_returns["Equal Weight Portfolio"].rolling(window=vol_window).std() * math.sqrt(TRADING_DAYS)

summary_rows = []
for ticker, df in asset_data.items():
    stats_map = compute_summary_stats(df["Daily Return"], risk_free_rate)
    rec_map = generate_recommendation(df, stats_map)
    stats_map.update(
        {
            "Ticker": ticker,
            "Latest Close": float(df["Close"].iloc[-1]),
            "Period High": float(df["Close"].max()),
            "Period Low": float(df["Close"].min()),
            "Recommendation": rec_map["Recommendation"],
            "Score": rec_map["Score"],
            "Latest RSI": rec_map["RSI"],
            "Latest Momentum": rec_map["Momentum"],
        }
    )
    summary_rows.append(stats_map)

portfolio_stats = compute_summary_stats(portfolio_returns["Equal Weight Portfolio"], risk_free_rate)
portfolio_stats.update(
    {
        "Ticker": "Equal Weight Portfolio",
        "Latest Close": float("nan"),
        "Period High": float("nan"),
        "Period Low": float("nan"),
        "Recommendation": "N/A",
        "Score": float("nan"),
        "Latest RSI": float("nan"),
        "Latest Momentum": float("nan"),
    }
)
summary_rows.append(portfolio_stats)

if not benchmark_df.empty:
    benchmark_stats = compute_summary_stats(benchmark_df["Daily Return"], risk_free_rate)
    benchmark_rec = generate_recommendation(benchmark_df, benchmark_stats)
    benchmark_stats.update(
        {
            "Ticker": "^GSPC",
            "Latest Close": float(benchmark_df["Close"].iloc[-1]),
            "Period High": float(benchmark_df["Close"].max()),
            "Period Low": float(benchmark_df["Close"].min()),
            "Recommendation": benchmark_rec["Recommendation"],
            "Score": benchmark_rec["Score"],
            "Latest RSI": benchmark_rec["RSI"],
            "Latest Momentum": benchmark_rec["Momentum"],
        }
    )
    summary_rows.append(benchmark_stats)

summary_df = pd.DataFrame(summary_rows).set_index("Ticker")
ranking_df = summary_df.copy()
ranking_df = ranking_df.loc[~ranking_df.index.isin(["Equal Weight Portfolio", "^GSPC"])].copy()

primary_stats = compute_summary_stats(primary_df["Daily Return"], risk_free_rate)
primary_recommendation = generate_recommendation(primary_df, primary_stats)

latest_close = float(primary_df["Close"].iloc[-1])
max_close = float(primary_df["Close"].max())
min_close = float(primary_df["Close"].min())

st.markdown(
    f"""
    <div class="{signal_class(primary_recommendation["Recommendation"])}">
        <div class="signal-label">{primary_ticker} Recommendation</div>
        <div class="signal-value">{primary_recommendation["Recommendation"]}</div>
        <div class="signal-reason">Score: {primary_recommendation["Score"]} | RSI: {format_metric(primary_recommendation["RSI"])} | Momentum: {format_metric(primary_recommendation["Momentum"])}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Latest Close", format_metric(latest_close, "currency"))
metric_col2.metric("Total Return", format_metric(primary_stats["Total Return"], "percent"))
metric_col3.metric("Annualized Return", format_metric(primary_stats["Annualized Return"], "percent"))
metric_col4.metric("Sharpe Ratio", format_metric(primary_stats["Sharpe Ratio"]))

metric_col5, metric_col6, metric_col7, metric_col8 = st.columns(4)
metric_col5.metric("Annualized Volatility", format_metric(primary_stats["Annualized Volatility"], "percent"))
metric_col6.metric("Period High", format_metric(max_close, "currency"))
metric_col7.metric("Period Low", format_metric(min_close, "currency"))
metric_col8.metric("Avg Daily Return", format_metric(primary_stats["Avg Daily Return"], "small_percent"))

with st.expander("Why this recommendation was generated", expanded=True):
    for reason in primary_recommendation["Reasons"]:
        st.write(reason)

st.markdown('<div class="section-title">Price and Trend</div>', unsafe_allow_html=True)

price_col, movers_col = st.columns([2.2, 1.2])

with price_col:
    fig_price = go.Figure()
    fig_price.add_trace(
        go.Scatter(
            x=primary_df.index,
            y=primary_df["Close"],
            mode="lines",
            name=f"{primary_ticker} Close",
            line=dict(width=2),
        )
    )
    fig_price.add_trace(
        go.Scatter(
            x=primary_df.index,
            y=primary_df[f"{ma_window}-Day MA"],
            mode="lines",
            name=f"{ma_window}-Day MA",
            line=dict(width=2, dash="dash"),
        )
    )
    fig_price.add_trace(
        go.Scatter(
            x=primary_df.index,
            y=primary_df["20-Day MA"],
            mode="lines",
            name="20-Day MA",
            line=dict(width=1.5, dash="dot"),
        )
    )
    fig_price.add_trace(
        go.Scatter(
            x=primary_df.index,
            y=primary_df["50-Day MA"],
            mode="lines",
            name="50-Day MA",
            line=dict(width=1.5),
        )
    )
    fig_price.update_layout(
        template="plotly_white",
        height=430,
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_price, use_container_width=True)

with movers_col:
    st.markdown("#### Top Movers")

    if ranking_df.empty:
        st.info("No valid stock tickers available for ranking.")
    else:
        if not show_up and not show_down:
            st.info("Turn on Up or Down in the sidebar to display top movers.")
        if show_up:
            winners_df = ranking_df.sort_values("Total Return", ascending=False).head(10)
            display_winners = winners_df[["Total Return", "Recommendation"]].copy()
            display_winners["Total Return"] = display_winners["Total Return"].map(lambda x: format_metric(x, "percent"))
            st.markdown("**Top 10 Up**")
            st.dataframe(display_winners, use_container_width=True, height=280)
        if show_down:
            losers_df = ranking_df.sort_values("Total Return", ascending=True).head(10)
            display_losers = losers_df[["Total Return", "Recommendation"]].copy()
            display_losers["Total Return"] = display_losers["Total Return"].map(lambda x: format_metric(x, "percent"))
            st.markdown("**Top 10 Down**")
            st.dataframe(display_losers, use_container_width=True, height=280)

if ma_window > len(primary_df):
    st.warning(
        f"The selected {ma_window}-day window is longer than the available data ({len(primary_df)} trading days). "
        f"The moving average line may not appear."
    )

st.markdown('<div class="section-title">Technical Indicators</div>', unsafe_allow_html=True)

tech_col1, tech_col2 = st.columns(2)

with tech_col1:
    fig_rsi = go.Figure()
    fig_rsi.add_trace(
        go.Scatter(
            x=primary_df.index,
            y=primary_df["RSI"],
            mode="lines",
            name="RSI",
            line=dict(width=2),
        )
    )
    fig_rsi.add_hline(y=70, line_dash="dash")
    fig_rsi.add_hline(y=30, line_dash="dash")
    fig_rsi.update_layout(
        template="plotly_white",
        height=320,
        xaxis_title="Date",
        yaxis_title="RSI",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

with tech_col2:
    fig_mom = go.Figure()
    fig_mom.add_trace(
        go.Scatter(
            x=primary_df.index,
            y=primary_df["Momentum"],
            mode="lines",
            name="Momentum",
            line=dict(width=2),
        )
    )
    fig_mom.add_hline(y=0, line_dash="dash")
    fig_mom.update_layout(
        template="plotly_white",
        height=320,
        xaxis_title="Date",
        yaxis_title="Momentum",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_mom, use_container_width=True)

st.markdown('<div class="section-title">Performance and Risk</div>', unsafe_allow_html=True)

perf_col1, perf_col2 = st.columns(2)

with perf_col1:
    fig_cum = go.Figure()
    fig_cum.add_trace(
        go.Scatter(
            x=primary_df.index,
            y=primary_df["Cumulative Return"],
            mode="lines",
            name="Cumulative Return",
            fill="tozeroy",
        )
    )
    fig_cum.update_layout(
        template="plotly_white",
        height=360,
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        yaxis_tickformat=".0%",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_cum, use_container_width=True)

with perf_col2:
    fig_roll_vol = go.Figure()
    fig_roll_vol.add_trace(
        go.Scatter(
            x=primary_df.index,
            y=primary_df["Rolling Volatility"],
            mode="lines",
            name=f"{vol_window}-Day Rolling Vol",
            line=dict(width=1.5),
        )
    )
    fig_roll_vol.add_trace(
        go.Scatter(
            x=portfolio_vol.index,
            y=portfolio_vol,
            mode="lines",
            name="Portfolio Rolling Vol",
            line=dict(width=2, dash="dash"),
        )
    )
    fig_roll_vol.update_layout(
        template="plotly_white",
        height=360,
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        yaxis_tickformat=".0%",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_roll_vol, use_container_width=True)

dist_col1, dist_col2 = st.columns(2)

with dist_col1:
    fig_vol = go.Figure()
    fig_vol.add_trace(
        go.Bar(
            x=primary_df.index,
            y=primary_df["Volume"],
            name="Volume",
            opacity=0.7,
        )
    )
    fig_vol.update_layout(
        template="plotly_white",
        height=320,
        xaxis_title="Date",
        yaxis_title="Shares Traded",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_vol, use_container_width=True)

with dist_col2:
    returns_clean = primary_df["Daily Return"].dropna()
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=returns_clean,
            nbinsx=60,
            opacity=0.75,
            name="Daily Returns",
            histnorm="probability density",
        )
    )
    if not returns_clean.empty:
        x_range = np.linspace(float(returns_clean.min()), float(returns_clean.max()), 200)
        mu = float(returns_clean.mean())
        sigma = float(returns_clean.std())
        if sigma > 0:
            fig_hist.add_trace(
                go.Scatter(
                    x=x_range,
                    y=stats.norm.pdf(x_range, mu, sigma),
                    mode="lines",
                    name="Normal Distribution",
                    line=dict(width=2),
                )
            )
    fig_hist.update_layout(
        template="plotly_white",
        height=320,
        xaxis_title="Daily Return",
        yaxis_title="Density",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.caption(
    f"Jarque-Bera test for {primary_ticker}: statistic = {primary_stats['Jarque-Bera Stat']:.2f}, "
    f"p-value = {primary_stats['Jarque-Bera p-value']:.4f}"
)

st.markdown('<div class="section-title">Multi-Stock Comparison</div>', unsafe_allow_html=True)

fig_compare = go.Figure()
for ticker in cumulative_df.columns:
    fig_compare.add_trace(
        go.Scatter(
            x=cumulative_df.index,
            y=cumulative_df[ticker],
            mode="lines",
            name=ticker,
        )
    )
if not benchmark_df.empty:
    fig_compare.add_trace(
        go.Scatter(
            x=benchmark_df.index,
            y=benchmark_df["Cumulative Return"],
            mode="lines",
            name="^GSPC Benchmark",
            line=dict(dash="dot"),
        )
    )
fig_compare.add_trace(
    go.Scatter(
        x=portfolio_cumulative.index,
        y=portfolio_cumulative,
        mode="lines",
        name="Equal Weight Portfolio",
        line=dict(width=3),
    )
)
fig_compare.update_layout(
    template="plotly_white",
    height=430,
    xaxis_title="Date",
    yaxis_title="Cumulative Return",
    yaxis_tickformat=".0%",
    margin=dict(l=20, r=20, t=30, b=20),
)
st.plotly_chart(fig_compare, use_container_width=True)

st.markdown('<div class="section-title">Summary Table</div>', unsafe_allow_html=True)

summary_display = summary_df.copy()
for col in ["Total Return", "Avg Daily Return", "Annualized Return", "Annualized Volatility"]:
    summary_display[col] = summary_display[col].map(
        lambda x: format_metric(x, "small_percent") if col == "Avg Daily Return" else format_metric(x, "percent")
    )
summary_display["Jarque-Bera p-value"] = summary_display["Jarque-Bera p-value"].map(format_metric)
summary_display["Sharpe Ratio"] = summary_display["Sharpe Ratio"].map(format_metric)
summary_display["Skewness"] = summary_display["Skewness"].map(format_metric)
summary_display["Excess Kurtosis"] = summary_display["Excess Kurtosis"].map(format_metric)
summary_display["Jarque-Bera Stat"] = summary_display["Jarque-Bera Stat"].map(format_metric)
summary_display["Observations"] = summary_display["Observations"].map(lambda x: format_metric(x, "integer"))
summary_display["Latest Close"] = summary_display["Latest Close"].map(lambda x: format_metric(x, "currency"))
summary_display["Period High"] = summary_display["Period High"].map(lambda x: format_metric(x, "currency"))
summary_display["Period Low"] = summary_display["Period Low"].map(lambda x: format_metric(x, "currency"))
summary_display["Score"] = summary_display["Score"].map(format_metric)
summary_display["Latest RSI"] = summary_display["Latest RSI"].map(format_metric)
summary_display["Latest Momentum"] = summary_display["Latest Momentum"].map(format_metric)
st.dataframe(summary_display, use_container_width=True)

st.markdown('<div class="section-title">Portfolio Explorer</div>', unsafe_allow_html=True)

port_col1, port_col2 = st.columns(2)
with port_col1:
    st.metric("Portfolio Total Return", format_metric(portfolio_stats["Total Return"], "percent"))
    st.metric("Portfolio Annualized Return", format_metric(portfolio_stats["Annualized Return"], "percent"))
    st.metric("Portfolio Sharpe Ratio", format_metric(portfolio_stats["Sharpe Ratio"]))
with port_col2:
    st.metric("Portfolio Annualized Volatility", format_metric(portfolio_stats["Annualized Volatility"], "percent"))
    st.metric("Portfolio Skewness", format_metric(portfolio_stats["Skewness"]))
    st.metric("Portfolio Jarque-Bera p-value", format_metric(portfolio_stats["Jarque-Bera p-value"]))

weights_df = pd.DataFrame(
    {
        "Ticker": list(asset_data.keys()),
        "Weight": [1 / len(asset_data)] * len(asset_data),
    }
)
weights_df["Weight"] = weights_df["Weight"].map(lambda x: f"{x:.2%}")
st.dataframe(weights_df, use_container_width=True)

fig_portfolio = go.Figure()
fig_portfolio.add_trace(
    go.Scatter(
        x=portfolio_cumulative.index,
        y=portfolio_cumulative,
        mode="lines",
        name="Equal Weight Portfolio",
        line=dict(width=3),
    )
)
if not benchmark_df.empty:
    fig_portfolio.add_trace(
        go.Scatter(
            x=benchmark_df.index,
            y=benchmark_df["Cumulative Return"],
            mode="lines",
            name="^GSPC Benchmark",
            line=dict(dash="dot"),
        )
    )
fig_portfolio.update_layout(
    template="plotly_white",
    height=400,
    xaxis_title="Date",
    yaxis_title="Cumulative Return",
    yaxis_tickformat=".0%",
    margin=dict(l=20, r=20, t=30, b=20),
)
st.plotly_chart(fig_portfolio, use_container_width=True)

st.markdown('<div class="section-title">Two-Asset Portfolio Explorer</div>', unsafe_allow_html=True)

valid_two_asset = len(return_df.columns) >= 2
if not valid_two_asset:
    st.info("Add at least two valid tickers to use the Two-Asset Portfolio Explorer.")
else:
    two_col1, two_col2, two_col3 = st.columns(3)
    asset_options = list(return_df.columns)
    asset_a = two_col1.selectbox("Asset 1", asset_options, index=0)
    default_b_index = 1 if len(asset_options) > 1 else 0
    asset_b = two_col2.selectbox("Asset 2", asset_options, index=default_b_index)
    weight_a_pct = two_col3.slider("Weight in Asset 1 (%)", 0, 100, 50, 5)

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
                f"= ({weight_a:.2f}² × {two_asset['var_a']:.8f}) + ({weight_b:.2f}² × {two_asset['var_b']:.8f}) "
                f"+ 2 × {weight_a:.2f} × {weight_b:.2f} × {two_asset['covariance']:.8f}  \n"
                f"= **{two_asset['portfolio_variance']:.8f}**"
            )

            fig_two_asset = go.Figure()
            fig_two_asset.add_trace(
                go.Scatter(
                    x=pair_cumulative.index,
                    y=pair_cumulative,
                    mode="lines",
                    name="Two-Asset Portfolio",
                    line=dict(width=3),
                )
            )
            fig_two_asset.add_trace(
                go.Scatter(
                    x=cumulative_df.index,
                    y=cumulative_df[asset_a],
                    mode="lines",
                    name=asset_a,
                    line=dict(dash="dash"),
                )
            )
            fig_two_asset.add_trace(
                go.Scatter(
                    x=cumulative_df.index,
                    y=cumulative_df[asset_b],
                    mode="lines",
                    name=asset_b,
                    line=dict(dash="dot"),
                )
            )
            fig_two_asset.update_layout(
                template="plotly_white",
                height=420,
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                yaxis_tickformat=".0%",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_two_asset, use_container_width=True)

st.markdown('<div class="section-title">Additional Diagnostics</div>', unsafe_allow_html=True)

diag_col1, diag_col2 = st.columns(2)

with diag_col1:
    qq_returns = primary_df["Daily Return"].dropna().tail(min(len(primary_df["Daily Return"].dropna()), qq_sample_size))
    fig_qq = go.Figure()
    if len(qq_returns) > 1:
        osm, osr = stats.probplot(qq_returns, dist="norm", fit=False)
        fig_qq.add_trace(
            go.Scatter(
                x=osm,
                y=osr,
                mode="markers",
                name="Observed Returns",
            )
        )
        slope, intercept, _ = stats.probplot(qq_returns, dist="norm", fit=True)[1]
        x_line = np.linspace(min(osm), max(osm), 100)
        y_line = slope * x_line + intercept
        fig_qq.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Reference Line",
            )
        )
    fig_qq.update_layout(
        template="plotly_white",
        height=360,
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_qq, use_container_width=True)

with diag_col2:
    if len(return_df.columns) >= 2:
        corr_matrix = return_df.dropna(how="all").corr()
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
            )
        )
        fig_heat.update_layout(
            template="plotly_white",
            height=360,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

if len(return_df.columns) >= 2:
    st.markdown("#### Rolling Correlation")
    corr_options = list(return_df.columns)
    corr_a = st.selectbox("First ticker", corr_options, index=0, key="corr_a")
    corr_b = st.selectbox("Second ticker", corr_options, index=1 if len(corr_options) > 1 else 0, key="corr_b")

    if corr_a == corr_b:
        st.info("Choose two different tickers to view rolling correlation.")
    else:
        rolling_corr = return_df[corr_a].rolling(window=corr_window).corr(return_df[corr_b])
        fig_roll_corr = go.Figure()
        fig_roll_corr.add_trace(
            go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr,
                mode="lines",
                name=f"{corr_a} vs {corr_b}",
                line=dict(width=2),
            )
        )
        fig_roll_corr.update_layout(
            template="plotly_white",
            height=380,
            xaxis_title="Date",
            yaxis_title="Rolling Correlation",
            yaxis=dict(range=[-1, 1]),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_roll_corr, use_container_width=True)

st.markdown('<div class="section-title">Raw Data</div>', unsafe_allow_html=True)
with st.expander(f"View Raw Data for {primary_ticker}"):
    st.dataframe(primary_df.tail(60), use_container_width=True)

csv_data = primary_df.to_csv().encode("utf-8")
st.download_button(
    label=f"Download {primary_ticker} data as CSV",
    data=csv_data,
    file_name=f"{primary_ticker.lower()}_stock_data.csv",
    mime="text/csv",
)