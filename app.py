import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy import stats

# ─────────────────────────────────────────────
#  PAGE CONFIG & GLOBAL STYLE
# ─────────────────────────────────────────────
st.set_page_config(page_title="Equitek · Stock Analyzer", layout="wide", page_icon="📈")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp {
        background: #0d0f14;
        color: #e8eaf0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #12151c !important;
        border-right: 1px solid #1e2330;
    }
    [data-testid="stSidebar"] * { color: #c8cad4 !important; }
    [data-testid="stSidebar"] h2 {
        font-family: 'DM Serif Display', serif;
        color: #f0c060 !important;
        font-size: 1.3rem;
        letter-spacing: .02em;
        margin-bottom: .5rem;
    }
    [data-testid="stSidebar"] .stSlider > div > div > div { background: #f0c060 !important; }
    [data-testid="stSidebar"] label { font-size: .82rem !important; color: #8b8fa8 !important; }

    /* ── Main headings ── */
    h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2.6rem !important;
        color: #f0c060;
        letter-spacing: -.01em;
    }
    h2 {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem !important;
        color: #c8cad4;
        border-bottom: 1px solid #1e2330;
        padding-bottom: .4rem;
        margin-top: 1.4rem !important;
    }
    h3 {
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem !important;
        font-weight: 600;
        color: #a0a4bc;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    /* ── Metric tiles ── */
    [data-testid="stMetric"] {
        background: #12151c;
        border: 1px solid #1e2330;
        border-radius: 10px;
        padding: 14px 18px !important;
    }
    [data-testid="stMetricLabel"] { font-size: .72rem !important; color: #6b6f87 !important; text-transform: uppercase; letter-spacing: .07em; }
    [data-testid="stMetricValue"] { font-family: 'DM Mono', monospace; font-size: 1.35rem !important; color: #f0c060 !important; }

    /* ── DataFrames ── */
    [data-testid="stDataFrame"] { border: 1px solid #1e2330; border-radius: 10px; overflow: hidden; }

    /* ── Buttons ── */
    .stDownloadButton button {
        background: #f0c060 !important;
        color: #0d0f14 !important;
        font-family: 'DM Mono', monospace;
        font-size: .82rem;
        border: none;
        border-radius: 6px;
        padding: 8px 18px;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        border: 1px solid #1e2330 !important;
        border-radius: 10px;
        background: #12151c;
    }
    [data-testid="stExpander"] summary { color: #c8cad4 !important; font-weight: 600; }

    /* ── Info / Warning / Error boxes ── */
    [data-testid="stAlert"] { border-radius: 8px; font-size: .88rem; }

    /* ── Section label pill ── */
    .pill {
        display: inline-block;
        background: #1e2330;
        color: #f0c060;
        font-family: 'DM Mono', monospace;
        font-size: .72rem;
        letter-spacing: .1em;
        padding: 3px 12px;
        border-radius: 20px;
        margin-bottom: 4px;
    }

    /* ── Divider ── */
    hr { border-color: #1e2330 !important; margin: 2rem 0 !important; }

    /* ── Suggestion badge colors ── */
    .badge-up   { background:#0e3a2a; color:#34c77a; border:1px solid #1a5e40; padding:2px 10px; border-radius:20px; font-size:.75rem; font-family:'DM Mono',monospace; }
    .badge-down { background:#3a0e0e; color:#e05c5c; border:1px solid #5e1a1a; padding:2px 10px; border-radius:20px; font-size:.75rem; font-family:'DM Mono',monospace; }

    /* ── Chart container ── */
    .chart-card {
        background: #12151c;
        border: 1px solid #1e2330;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }

    /* scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0d0f14; }
    ::-webkit-scrollbar-thumb { background: #2a2e3d; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
TRADING_DAYS = 252
BENCHMARK_TICKER = "^GSPC"
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA"]
FALLBACK_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","NFLX","AMD",
    "INTC","MU","ADBE","CRM","ORCL","QCOM","TXN","UBER","SHOP","CRWD",
    "PANW","NOW","PLTR","ASML","ARM","SNOW","ANET","MELI","AMAT","LRCX",
    "KLAC","MRVL","CDNS","SNPS","TTD","DDOG","ZS","MDB","NET","TEAM",
    "JPM","GS","MS","BAC","WFC","BLK","KKR","BX","V","MA",
    "XOM","CVX","COP","SLB","EOG","MPC","PSX","OXY","HAL","FANG",
    "LLY","UNH","JNJ","ABBV","MRK","ISRG","ABT","PFE","TMO","DHR",
    "COST","WMT","HD","LOW","MCD","SBUX","NKE","DIS","CMG","BKNG",
    "CAT","DE","GE","RTX","BA","HON","ETN","LMT","NOC","UNP",
    "SPY","QQQ","IWM","DIA","XLF","XLK","XLE","XLI","XLV","XLY",
]
UNIVERSE_FILE_CANDIDATES = [
    "stock_universe.csv","ticker_universe.csv","us_stocks_universe.csv",
]

# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Fetching stock data…")
def load_data(symbols: tuple[str, ...], start: date, end: date) -> dict[str, pd.DataFrame]:
    data_map: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        if not symbol:
            continue
        try:
            df = yf.download(
                symbol, start=start, end=end + timedelta(days=1),
                interval="1d", auto_adjust=False, progress=False, threads=False,
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
            symbol, start=start, end=end + timedelta(days=1),
            interval="1d", auto_adjust=False, progress=False, threads=False,
        )
    except Exception:
        df = pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if not df.empty:
        df = df.sort_index()
    return df


# ─────────────────────────────────────────────
#  VALIDATION & ENRICHMENT
# ─────────────────────────────────────────────
def validate_data(df: pd.DataFrame) -> bool:
    return not df.empty and {"Close", "Volume"}.issubset(df.columns)


def enrich_single_asset(df: pd.DataFrame, ma_window: int, vol_window: int) -> pd.DataFrame:
    out = df.copy().dropna(subset=["Close"])
    out["Daily Return"]     = out["Close"].pct_change()
    out["Cumulative Return"] = (1 + out["Daily Return"].fillna(0)).cumprod() - 1
    out[f"{ma_window}-Day MA"] = out["Close"].rolling(window=ma_window).mean()
    out["Rolling Volatility"] = (
        out["Daily Return"].rolling(window=vol_window).std() * math.sqrt(TRADING_DAYS)
    )
    return out


# ─────────────────────────────────────────────
#  STATISTICS
# ─────────────────────────────────────────────
def compute_summary_stats(returns: pd.Series, risk_free_rate: float) -> dict[str, float]:
    returns = returns.dropna()
    nan_result = {
        "Total Return": float("nan"), "Avg Daily Return": float("nan"),
        "Annualized Return": float("nan"), "Annualized Volatility": float("nan"),
        "Sharpe Ratio": float("nan"), "Skewness": float("nan"),
        "Excess Kurtosis": float("nan"), "Jarque-Bera Stat": float("nan"),
        "Jarque-Bera p-value": float("nan"), "Observations": 0,
    }
    if returns.empty:
        return nan_result

    cumulative  = (1 + returns).cumprod() - 1
    avg_daily   = float(returns.mean())
    volatility  = float(returns.std())
    ann_vol     = volatility * math.sqrt(TRADING_DAYS) if not math.isnan(volatility) else float("nan")
    ann_ret     = avg_daily * TRADING_DAYS if not math.isnan(avg_daily) else float("nan")
    sharpe      = (ann_ret - risk_free_rate) / ann_vol if ann_vol and not math.isnan(ann_vol) else float("nan")
    jb_stat, jb_p = stats.jarque_bera(returns)

    return {
        "Total Return": float(cumulative.iloc[-1]),
        "Avg Daily Return": avg_daily,
        "Annualized Return": ann_ret,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Skewness": float(returns.skew()),
        "Excess Kurtosis": float(returns.kurtosis()),
        "Jarque-Bera Stat": float(jb_stat),
        "Jarque-Bera p-value": float(jb_p),
        "Observations": int(returns.shape[0]),
    }


def format_metric(value: float, kind: str = "number") -> str:
    if pd.isna(value):
        return "N/A"
    if kind == "currency":      return f"${value:,.2f}"
    if kind == "percent":       return f"{value:.2%}"
    if kind == "small_percent": return f"{value:.4%}"
    if kind == "integer":       return f"{int(value):,}"
    return f"{value:.2f}"


# ─────────────────────────────────────────────
#  RSI
# ─────────────────────────────────────────────
def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi      = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else float("nan")


# ─────────────────────────────────────────────
#  RECOMMENDATION ENGINE  (program-only, never
#  seeded from the user's ticker list)
# ─────────────────────────────────────────────
def compute_recommendation_features(df: pd.DataFrame, risk_free_rate: float) -> dict[str, float]:
    enriched = enrich_single_asset(df, ma_window=50, vol_window=30)
    returns  = enriched["Daily Return"].dropna()
    close    = enriched["Close"].dropna()

    if returns.empty or close.empty or len(close) < 30:
        return {}

    total_return = float(enriched["Cumulative Return"].iloc[-1])
    volatility   = float(returns.std()) * math.sqrt(TRADING_DAYS)
    avg_daily    = float(returns.mean())
    ann_return   = avg_daily * TRADING_DAYS
    sharpe       = (ann_return - risk_free_rate) / volatility if volatility and not math.isnan(volatility) else float("nan")

    momentum_20 = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 20 else float("nan")
    momentum_60 = float(close.iloc[-1] / close.iloc[-61] - 1) if len(close) > 60 else float("nan")
    ma_20       = float(close.rolling(20).mean().iloc[-1])     if len(close) >= 20 else float("nan")
    ma_50       = float(close.rolling(50).mean().iloc[-1])     if len(close) >= 50 else float("nan")
    rsi_14      = compute_rsi(close, 14)

    # Trend score: price vs moving averages
    trend_score = 0.0
    if not math.isnan(ma_20):
        trend_score += 1.0 if close.iloc[-1] > ma_20 else -1.0
    if not math.isnan(ma_50):
        trend_score += 1.0 if close.iloc[-1] > ma_50 else -1.0

    # RSI score
    rsi_score = 0.0
    if not math.isnan(rsi_14):
        if 45 <= rsi_14 <= 65:   rsi_score =  1.0
        elif rsi_14 > 75:        rsi_score = -1.0
        elif rsi_14 < 25:        rsi_score =  0.5

    profitability_score = (
          0.30 * total_return
        + 0.25 * (0 if math.isnan(sharpe)       else sharpe)
        + 0.20 * (0 if math.isnan(momentum_20)  else momentum_20)
        + 0.15 * (0 if math.isnan(momentum_60)  else momentum_60)
        + 0.10 * trend_score
        + 0.05 * rsi_score
        - 0.10 * (0 if math.isnan(volatility)   else volatility)
    )

    return {
        "Latest Close":       float(close.iloc[-1]),
        "Total Return":       total_return,
        "Annualized Return":  ann_return,
        "Annualized Vol":     volatility,
        "Sharpe Ratio":       sharpe,
        "Momentum 20D":       momentum_20,
        "Momentum 60D":       momentum_60,
        "RSI 14":             rsi_14,
        "Trend Score":        trend_score,
        "Score":              profitability_score,
    }


@st.cache_data(ttl=3600, show_spinner="Scanning candidate universe…")
def build_program_suggestions(
    candidate_tickers: tuple[str, ...],
    start: date,
    end: date,
    risk_free_rate: float,
    max_scan: int,
) -> pd.DataFrame:
    """Scan the *program's own* universe — completely independent of the user's tickers."""
    selected  = tuple(candidate_tickers[:max_scan])
    raw_map   = load_data(selected, start, end)
    rows      = []

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
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Score", "Latest Close"])
    return out.sort_values("Score", ascending=False).reset_index(drop=True)


def get_candidate_universe(target_size: int) -> tuple[list[str], str]:
    for file_name in UNIVERSE_FILE_CANDIDATES:
        path = Path(file_name)
        if path.exists():
            try:
                df        = pd.read_csv(path)
                first_col = df.columns[0]
                tickers   = [str(x).strip().upper() for x in df[first_col].dropna().tolist() if str(x).strip()]
                tickers   = list(dict.fromkeys(tickers))
                if tickers:
                    return tickers[:target_size], f"Loaded {min(len(tickers), target_size):,} symbols from **{file_name}**."
            except Exception:
                pass

    fallback = list(dict.fromkeys(FALLBACK_UNIVERSE))
    return fallback[: min(target_size, len(fallback))], (
        "⚠️ No local universe file found — using built-in fallback universe. "
        "For a true 10,000-symbol scan, add a CSV like `stock_universe.csv` with one ticker per row."
    )


def compute_two_asset_portfolio(
    returns_df: pd.DataFrame, asset_a: str, asset_b: str, weight_a: float,
) -> dict[str, object]:
    pair_df  = returns_df[[asset_a, asset_b]].dropna().copy()
    weight_b = 1.0 - weight_a

    if pair_df.empty:
        return {
            "pair_returns": pd.Series(dtype=float), "covariance": float("nan"),
            "corr": float("nan"), "var_a": float("nan"), "var_b": float("nan"),
            "portfolio_variance": float("nan"), "portfolio_volatility": float("nan"),
            "weight_b": weight_b,
        }

    pair_df["Portfolio Return"] = weight_a * pair_df[asset_a] + weight_b * pair_df[asset_b]
    var_a    = float(pair_df[asset_a].var())
    var_b    = float(pair_df[asset_b].var())
    covariance = float(pair_df[[asset_a, asset_b]].cov().iloc[0, 1])
    corr       = float(pair_df[[asset_a, asset_b]].corr().iloc[0, 1])
    pvar       = (weight_a**2)*var_a + (weight_b**2)*var_b + 2*weight_a*weight_b*covariance
    pvol       = math.sqrt(pvar) if pvar >= 0 else float("nan")

    return {
        "pair_returns": pair_df["Portfolio Return"],
        "covariance": covariance, "corr": corr,
        "var_a": var_a, "var_b": var_b,
        "portfolio_variance": pvar, "portfolio_volatility": pvol,
        "weight_b": weight_b,
    }


# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.divider()

    raw_tickers = st.text_input("Tickers (comma-separated)", value=", ".join(DEFAULT_TICKERS))
    user_tickers = list(dict.fromkeys([t.strip().upper() for t in raw_tickers.split(",") if t.strip()]))

    include_benchmark        = st.checkbox("Include S&P 500 (^GSPC)", value=True)
    show_program_suggestions = st.checkbox("Show AI-Picked Suggestions", value=True)

    st.divider()
    requested_pool_size = st.number_input(
        "Candidate pool size", min_value=100, max_value=10000, value=10000, step=100
    )
    max_live_scan = st.slider("Live scan limit", min_value=50, max_value=500, value=200, step=25)

    st.divider()
    default_start = date.today() - timedelta(days=365)
    start_date = st.date_input("Start Date", value=default_start)
    end_date   = st.date_input("End Date",   value=date.today())

    st.divider()
    ma_window  = st.slider("Moving Average Window",      min_value=5,  max_value=200, value=50,  step=5)
    vol_window = st.slider("Rolling Volatility Window",  min_value=10, max_value=120, value=30,  step=5)
    corr_window = st.slider("Rolling Correlation Window",min_value=10, max_value=120, value=30,  step=5)
    qq_sample_size = st.slider("Q-Q Plot Sample Size",   min_value=50, max_value=500, value=250, step=25)
    risk_free_rate = st.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=4.5, step=0.1) / 100

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

if not user_tickers:
    st.info("Enter at least one stock ticker in the sidebar to get started.")
    st.stop()

# ─────────────────────────────────────────────
#  DATA FETCH
# ─────────────────────────────────────────────
fetch_tickers = list(user_tickers)
if include_benchmark and BENCHMARK_TICKER not in fetch_tickers:
    fetch_tickers.append(BENCHMARK_TICKER)

try:
    raw_data = load_data(tuple(fetch_tickers), start_date, end_date)
except Exception as e:
    st.error("Failed to download stock data.")
    st.exception(e)
    st.stop()

asset_data:    dict[str, pd.DataFrame] = {}
invalid_tickers: list[str]            = []

for ticker in user_tickers:
    df = raw_data.get(ticker, pd.DataFrame())
    if validate_data(df):
        asset_data[ticker] = enrich_single_asset(df, ma_window, vol_window)
    else:
        invalid_tickers.append(ticker)

if invalid_tickers:
    st.warning(f"No data found for: {', '.join(invalid_tickers)}")

if not asset_data:
    st.error("No valid stock data returned. Try different tickers or a wider date range.")
    st.stop()

benchmark_df = pd.DataFrame()
if include_benchmark:
    braw = raw_data.get(BENCHMARK_TICKER, pd.DataFrame())
    if validate_data(braw):
        benchmark_df = enrich_single_asset(braw, ma_window, vol_window)
    else:
        st.warning("S&P 500 benchmark data could not be loaded.")

primary_ticker = next(iter(asset_data.keys()))
primary_df     = asset_data[primary_ticker]

return_df      = pd.concat([df["Daily Return"].rename(t) for t, df in asset_data.items()], axis=1).dropna(how="all")
cumulative_df  = (1 + return_df.fillna(0)).cumprod() - 1

portfolio_returns                    = return_df.copy().dropna(how="all")
portfolio_returns["Equal Weight"]    = portfolio_returns.mean(axis=1, skipna=True)
portfolio_cumulative                 = (1 + portfolio_returns["Equal Weight"].fillna(0)).cumprod() - 1
portfolio_vol                        = portfolio_returns["Equal Weight"].rolling(window=vol_window).std() * math.sqrt(TRADING_DAYS)

# Summary stats
summary_rows = []
for ticker, df in asset_data.items():
    sm = compute_summary_stats(df["Daily Return"], risk_free_rate)
    sm.update({"Ticker": ticker, "Latest Close": float(df["Close"].iloc[-1]),
                "Period High": float(df["Close"].max()), "Period Low": float(df["Close"].min())})
    summary_rows.append(sm)

portfolio_stats = compute_summary_stats(portfolio_returns["Equal Weight"], risk_free_rate)
portfolio_stats.update({"Ticker": "Equal Weight Portfolio",
                        "Latest Close": float("nan"), "Period High": float("nan"), "Period Low": float("nan")})
summary_rows.append(portfolio_stats)

if not benchmark_df.empty:
    bs = compute_summary_stats(benchmark_df["Daily Return"], risk_free_rate)
    bs.update({"Ticker": "^GSPC", "Latest Close": float(benchmark_df["Close"].iloc[-1]),
               "Period High": float(benchmark_df["Close"].max()), "Period Low": float(benchmark_df["Close"].min())})
    summary_rows.append(bs)

summary_df  = pd.DataFrame(summary_rows).set_index("Ticker")
primary_stats = compute_summary_stats(primary_df["Daily Return"], risk_free_rate)

# ═══════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════
st.markdown(
    f"""
    <div style="display:flex;align-items:baseline;gap:1rem;margin-bottom:.2rem">
      <h1 style="margin:0">Equitek</h1>
      <span style="font-family:'DM Mono',monospace;font-size:.85rem;color:#6b6f87">
        Stock Analysis Dashboard
      </span>
    </div>
    <p style="color:#6b6f87;font-size:.85rem;margin-top:0">
      Analyzing&nbsp;
      {'&nbsp;·&nbsp;'.join(f'<span style="color:#f0c060;font-weight:600">{t}</span>' for t in asset_data)}
      &nbsp;·&nbsp;{start_date.strftime('%b %d %Y')} → {end_date.strftime('%b %d %Y')}
    </p>
    """,
    unsafe_allow_html=True,
)
st.divider()


# ═══════════════════════════════════════════════════════════
#  SECTION 0 — AI-PICKED SUGGESTIONS  (program's own universe)
# ═══════════════════════════════════════════════════════════
if show_program_suggestions:
    st.markdown('<span class="pill">AI SCANNER</span>', unsafe_allow_html=True)
    st.markdown("## 🤖 Program-Picked Suggestions")
    st.caption(
        "Stocks below are selected **entirely by the program** from its own candidate universe — "
        "completely independent of the tickers you entered above."
    )

    candidate_universe, universe_message = get_candidate_universe(int(requested_pool_size))
    st.caption(universe_message)
    st.caption(
        f"Pool: **{int(requested_pool_size):,}** symbols requested · "
        f"Live scan limit this run: **{max_live_scan:,}** · scores cached for 1 h."
    )

    suggestion_df = build_program_suggestions(
        tuple(candidate_universe), start_date, end_date, risk_free_rate, max_live_scan,
    )

    # Score methodology callout
    with st.expander("📐 Scoring Methodology", expanded=False):
        st.markdown(
            """
| Factor | Weight | Notes |
|---|---|---|
| Total Return | 30 % | Cumulative period return |
| Sharpe Ratio | 25 % | Risk-adjusted excess return |
| 20-Day Momentum | 20 % | Short-term price acceleration |
| 60-Day Momentum | 15 % | Medium-term trend strength |
| Trend Score | 10 % | Price vs 20-day & 50-day MA |
| RSI Score | 5 % | 45–65 healthy · >75 penalty · <25 small bonus |
| Volatility Penalty | −10 % | Annualized volatility drag |

**Interpretation:** Buy → strong returns + good momentum + price above MAs + solid Sharpe.  
Sell → weak returns + bad momentum + price below MAs + poor risk-adjusted profile.
            """
        )

    if suggestion_df.empty:
        st.warning("No suggestions available for the current scan settings.")
    else:
        up_col, down_col = st.columns(2, gap="large")

        def style_suggestion_table(df_in: pd.DataFrame) -> pd.DataFrame:
            display = df_in[["Ticker", "Score", "Sharpe Ratio", "Momentum 20D",
                              "Momentum 60D", "Annualized Return", "RSI 14", "Latest Close"]].copy()
            display["Score"]            = display["Score"].map(format_metric)
            display["Sharpe Ratio"]     = display["Sharpe Ratio"].map(format_metric)
            display["Momentum 20D"]     = display["Momentum 20D"].map(lambda x: format_metric(x, "percent"))
            display["Momentum 60D"]     = display["Momentum 60D"].map(lambda x: format_metric(x, "percent"))
            display["Annualized Return"]= display["Annualized Return"].map(lambda x: format_metric(x, "percent"))
            display["RSI 14"]           = display["RSI 14"].map(format_metric)
            display["Latest Close"]     = display["Latest Close"].map(lambda x: format_metric(x, "currency"))
            return display

        with up_col:
            st.markdown(
                '<span class="badge-up">▲ TOP 10 BUY CANDIDATES</span>',
                unsafe_allow_html=True,
            )
            top_up = suggestion_df.head(10).copy()
            st.dataframe(style_suggestion_table(top_up), use_container_width=True, height=395)

        with down_col:
            st.markdown(
                '<span class="badge-down">▼ TOP 10 SELL / AVOID</span>',
                unsafe_allow_html=True,
            )
            top_down = suggestion_df.tail(10).sort_values("Score", ascending=True).copy()
            st.dataframe(style_suggestion_table(top_down), use_container_width=True, height=395)

    st.divider()


# ═══════════════════════════════════════════════════════════
#  SECTION 1 — KEY METRICS
# ═══════════════════════════════════════════════════════════
st.markdown('<span class="pill">OVERVIEW</span>', unsafe_allow_html=True)
st.markdown(f"## 📊 {primary_ticker} — Key Metrics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Close",        format_metric(float(primary_df["Close"].iloc[-1]), "currency"))
c2.metric("Total Return",        format_metric(primary_stats["Total Return"],       "percent"))
c3.metric("Annualized Return",   format_metric(primary_stats["Annualized Return"],  "percent"))
c4.metric("Sharpe Ratio",        format_metric(primary_stats["Sharpe Ratio"]))

c5, c6, c7, c8 = st.columns(4)
c5.metric("Annualized Volatility",format_metric(primary_stats["Annualized Volatility"], "percent"))
c6.metric("Skewness",             format_metric(primary_stats["Skewness"]))
c7.metric("Excess Kurtosis",      format_metric(primary_stats["Excess Kurtosis"]))
c8.metric("Avg Daily Return",     format_metric(primary_stats["Avg Daily Return"], "small_percent"))

c9, c10, c11, c12 = st.columns(4)
c9.metric("Period High",          format_metric(float(primary_df["Close"].max()), "currency"))
c10.metric("Period Low",          format_metric(float(primary_df["Close"].min()), "currency"))
c11.metric("Jarque-Bera Stat",    format_metric(primary_stats["Jarque-Bera Stat"]))
c12.metric("Jarque-Bera p-value", format_metric(primary_stats["Jarque-Bera p-value"]))

st.divider()


# ═══════════════════════════════════════════════════════════
#  SECTION 2 — MARKET VIEW
# ═══════════════════════════════════════════════════════════
st.markdown('<span class="pill">MARKET VIEW</span>', unsafe_allow_html=True)
st.markdown("## 📈 Price & Technical")

CHART_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#12151c",
    plot_bgcolor="#0d0f14",
    font_color="#c8cad4",
    xaxis=dict(gridcolor="#1e2330", showgrid=True),
    yaxis=dict(gridcolor="#1e2330", showgrid=True),
    margin=dict(l=20, r=20, t=40, b=20),
)

# Price + MA
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(
    x=primary_df.index, y=primary_df["Close"],
    mode="lines", name=f"{primary_ticker} Close",
    line=dict(width=2, color="#f0c060"),
))
fig_price.add_trace(go.Scatter(
    x=primary_df.index, y=primary_df[f"{ma_window}-Day MA"],
    mode="lines", name=f"{ma_window}-Day MA",
    line=dict(width=1.5, dash="dash", color="#5b9cf6"),
))
fig_price.update_layout(height=450, xaxis_title="Date", yaxis_title="Price (USD)", **CHART_THEME)
st.plotly_chart(fig_price, use_container_width=True)

if ma_window > len(primary_df):
    st.warning(f"Selected {ma_window}-day window exceeds available data ({len(primary_df)} days).")

st.divider()


# ═══════════════════════════════════════════════════════════
#  SECTION 3 — PERFORMANCE & RISK
# ═══════════════════════════════════════════════════════════
st.markdown('<span class="pill">PERFORMANCE & RISK</span>', unsafe_allow_html=True)
st.markdown("## 📉 Performance Analysis")

perf_l, perf_r = st.columns(2, gap="large")

with perf_l:
    st.subheader("Daily Trading Volume")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=primary_df.index, y=primary_df["Volume"],
        name="Volume", opacity=0.75, marker_color="#5b9cf6",
    ))
    fig_vol.update_layout(height=350, xaxis_title="Date", yaxis_title="Shares Traded", **CHART_THEME)
    st.plotly_chart(fig_vol, use_container_width=True)

with perf_r:
    st.subheader("Distribution of Daily Returns")
    returns_clean = primary_df["Daily Return"].dropna()
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=returns_clean, nbinsx=60, opacity=0.75,
        name="Daily Returns", histnorm="probability density",
        marker_color="#f0c060",
    ))
    if not returns_clean.empty:
        x_range = np.linspace(float(returns_clean.min()), float(returns_clean.max()), 200)
        mu, sigma = float(returns_clean.mean()), float(returns_clean.std())
        if sigma > 0:
            fig_hist.add_trace(go.Scatter(
                x=x_range, y=stats.norm.pdf(x_range, mu, sigma),
                mode="lines", name="Normal Fit", line=dict(width=2, color="#e05c5c"),
            ))
    fig_hist.update_layout(height=350, xaxis_title="Daily Return", yaxis_title="Density", **CHART_THEME)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(
        f"Jarque-Bera: stat = {primary_stats['Jarque-Bera Stat']:.2f}, "
        f"p-value = {primary_stats['Jarque-Bera p-value']:.4f}"
    )

trend_l, trend_r = st.columns(2, gap="large")

with trend_l:
    st.subheader("Cumulative Return Over Time")
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=primary_df.index, y=primary_df["Cumulative Return"],
        mode="lines", name="Cumulative Return", fill="tozeroy",
        line=dict(color="#34c77a", width=2),
    ))
    fig_cum.update_layout(height=400, xaxis_title="Date", yaxis_title="Cumulative Return",
                          yaxis_tickformat=".0%", **CHART_THEME)
    st.plotly_chart(fig_cum, use_container_width=True)

with trend_r:
    st.subheader("Rolling Annualized Volatility")
    fig_rvol = go.Figure()
    fig_rvol.add_trace(go.Scatter(
        x=primary_df.index, y=primary_df["Rolling Volatility"],
        mode="lines", name=f"{vol_window}-Day Rolling Vol",
        line=dict(width=1.5, color="#f0c060"),
    ))
    fig_rvol.add_trace(go.Scatter(
        x=portfolio_vol.index, y=portfolio_vol,
        mode="lines", name="Portfolio Rolling Vol",
        line=dict(width=2, dash="dash", color="#5b9cf6"),
    ))
    fig_rvol.update_layout(height=400, xaxis_title="Date", yaxis_title="Annualized Volatility",
                           yaxis_tickformat=".0%", **CHART_THEME)
    st.plotly_chart(fig_rvol, use_container_width=True)

st.divider()


# ═══════════════════════════════════════════════════════════
#  SECTION 4 — RELATIVE PERFORMANCE
# ═══════════════════════════════════════════════════════════
st.markdown('<span class="pill">RELATIVE PERFORMANCE</span>', unsafe_allow_html=True)
st.markdown("## 🔀 Multi-Stock Comparison")

COLORS = ["#f0c060", "#5b9cf6", "#34c77a", "#e05c5c", "#a78bfa", "#fb923c", "#22d3ee"]
fig_compare = go.Figure()
for i, ticker in enumerate(cumulative_df.columns):
    fig_compare.add_trace(go.Scatter(
        x=cumulative_df.index, y=cumulative_df[ticker],
        mode="lines", name=ticker,
        line=dict(width=2, color=COLORS[i % len(COLORS)]),
    ))
if not benchmark_df.empty:
    fig_compare.add_trace(go.Scatter(
        x=benchmark_df.index, y=benchmark_df["Cumulative Return"],
        mode="lines", name="^GSPC", line=dict(dash="dot", color="#6b6f87", width=1.5),
    ))
fig_compare.add_trace(go.Scatter(
    x=portfolio_cumulative.index, y=portfolio_cumulative,
    mode="lines", name="Equal Weight Portfolio",
    line=dict(width=3, color="#ffffff"),
))
fig_compare.update_layout(height=450, xaxis_title="Date", yaxis_title="Cumulative Return",
                          yaxis_tickformat=".0%", **CHART_THEME)
st.plotly_chart(fig_compare, use_container_width=True)

st.subheader("Summary Statistics")
summary_display = summary_df.copy()
for col in ["Total Return", "Avg Daily Return", "Annualized Return", "Annualized Volatility"]:
    summary_display[col] = summary_display[col].map(
        lambda x, c=col: format_metric(x, "small_percent") if c == "Avg Daily Return" else format_metric(x, "percent")
    )
for col in ["Sharpe Ratio", "Skewness", "Excess Kurtosis", "Jarque-Bera Stat"]:
    summary_display[col] = summary_display[col].map(format_metric)
summary_display["Jarque-Bera p-value"] = summary_display["Jarque-Bera p-value"].map(format_metric)
summary_display["Observations"]        = summary_display["Observations"].map(lambda x: format_metric(x, "integer"))
for col in ["Latest Close", "Period High", "Period Low"]:
    summary_display[col] = summary_display[col].map(lambda x: format_metric(x, "currency"))
st.dataframe(summary_display, use_container_width=True, height=330)

st.divider()


# ═══════════════════════════════════════════════════════════
#  SECTION 5 — PORTFOLIO ANALYSIS
# ═══════════════════════════════════════════════════════════
st.markdown('<span class="pill">PORTFOLIO</span>', unsafe_allow_html=True)
st.markdown("## 💼 Portfolio Analysis")

st.subheader("Equal-Weight Portfolio")
p1, p2, p3 = st.columns(3)
p1.metric("Total Return",        format_metric(portfolio_stats["Total Return"],        "percent"))
p1.metric("Annualized Return",   format_metric(portfolio_stats["Annualized Return"],   "percent"))
p2.metric("Sharpe Ratio",        format_metric(portfolio_stats["Sharpe Ratio"]))
p2.metric("Annualized Volatility",format_metric(portfolio_stats["Annualized Volatility"],"percent"))
p3.metric("Skewness",            format_metric(portfolio_stats["Skewness"]))
p3.metric("Jarque-Bera p-value", format_metric(portfolio_stats["Jarque-Bera p-value"]))

weights_df = pd.DataFrame({
    "Ticker": list(asset_data.keys()),
    "Weight": [f"{1/len(asset_data):.2%}"] * len(asset_data),
})
st.dataframe(weights_df, use_container_width=True, height=180)

fig_port = go.Figure()
fig_port.add_trace(go.Scatter(
    x=portfolio_cumulative.index, y=portfolio_cumulative,
    mode="lines", name="Equal Weight Portfolio",
    line=dict(width=3, color="#34c77a"),
))
if not benchmark_df.empty:
    fig_port.add_trace(go.Scatter(
        x=benchmark_df.index, y=benchmark_df["Cumulative Return"],
        mode="lines", name="^GSPC", line=dict(dash="dot", color="#6b6f87"),
    ))
fig_port.update_layout(height=400, xaxis_title="Date", yaxis_title="Cumulative Return",
                       yaxis_tickformat=".0%", **CHART_THEME)
st.plotly_chart(fig_port, use_container_width=True)

# Two-Asset Explorer
st.subheader("Two-Asset Explorer")
if len(return_df.columns) < 2:
    st.info("Add at least two valid tickers to use the Two-Asset Explorer.")
else:
    ta1, ta2, ta3 = st.columns(3)
    opts    = list(return_df.columns)
    asset_a = ta1.selectbox("Asset 1", opts, index=0)
    asset_b = ta2.selectbox("Asset 2", opts, index=1 if len(opts) > 1 else 0)
    w_a_pct = ta3.slider("Weight in Asset 1 (%)", 0, 100, 50, 5)

    if asset_a == asset_b:
        st.error("Asset 1 and Asset 2 must be different.")
    else:
        weight_a    = w_a_pct / 100
        two_asset   = compute_two_asset_portfolio(return_df, asset_a, asset_b, weight_a)
        weight_b    = two_asset["weight_b"]

        if two_asset["pair_returns"].empty:
            st.error("Not enough overlapping return data.")
        else:
            pair_stats      = compute_summary_stats(two_asset["pair_returns"], risk_free_rate)
            pair_cumulative = (1 + two_asset["pair_returns"].fillna(0)).cumprod() - 1

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Weight A",           f"{weight_a:.0%}")
            m2.metric("Weight B",           f"{weight_b:.0%}")
            m3.metric("Portfolio Variance", f"{two_asset['portfolio_variance']:.8f}")
            m4.metric("Portfolio Volatility",f"{two_asset['portfolio_volatility']:.4%}")

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Covariance",        f"{two_asset['covariance']:.8f}")
            m6.metric("Correlation",       f"{two_asset['corr']:.2f}")
            m7.metric("Total Return",      format_metric(pair_stats["Total Return"], "percent"))
            m8.metric("Sharpe Ratio",      format_metric(pair_stats["Sharpe Ratio"]))

            st.markdown(
                f"**σ²ₚ = w₁²σ₁² + w₂²σ₂² + 2w₁w₂Cov(R₁,R₂)**  \n"
                f"= ({weight_a:.2f}² × {two_asset['var_a']:.8f}) + ({weight_b:.2f}² × {two_asset['var_b']:.8f}) "
                f"+ 2 × {weight_a:.2f} × {weight_b:.2f} × {two_asset['covariance']:.8f}  \n"
                f"= **{two_asset['portfolio_variance']:.8f}**"
            )

            fig_ta = go.Figure()
            fig_ta.add_trace(go.Scatter(
                x=pair_cumulative.index, y=pair_cumulative,
                mode="lines", name="Two-Asset Portfolio",
                line=dict(width=3, color="#f0c060"),
            ))
            fig_ta.add_trace(go.Scatter(
                x=cumulative_df.index, y=cumulative_df[asset_a],
                mode="lines", name=asset_a, line=dict(dash="dash", color="#5b9cf6"),
            ))
            fig_ta.add_trace(go.Scatter(
                x=cumulative_df.index, y=cumulative_df[asset_b],
                mode="lines", name=asset_b, line=dict(dash="dot", color="#34c77a"),
            ))
            fig_ta.update_layout(height=420, xaxis_title="Date", yaxis_title="Cumulative Return",
                                 yaxis_tickformat=".0%", **CHART_THEME)
            st.plotly_chart(fig_ta, use_container_width=True)

st.divider()


# ═══════════════════════════════════════════════════════════
#  SECTION 6 — STATISTICAL DIAGNOSTICS
# ═══════════════════════════════════════════════════════════
st.markdown('<span class="pill">DIAGNOSTICS</span>', unsafe_allow_html=True)
st.markdown("## 🔬 Statistical Diagnostics")

diag_l, diag_r = st.columns(2, gap="large")

with diag_l:
    st.subheader("Q-Q Plot")
    qq_returns = returns_clean.tail(min(len(returns_clean), qq_sample_size))
    fig_qq     = go.Figure()
    if len(qq_returns) > 1:
        osm, osr     = stats.probplot(qq_returns, dist="norm", fit=False)
        slope, intercept, _ = stats.probplot(qq_returns, dist="norm", fit=True)[1]
        fig_qq.add_trace(go.Scatter(
            x=osm, y=osr, mode="markers", name="Observed",
            marker=dict(color="#f0c060", size=4, opacity=0.7),
        ))
        x_line = np.linspace(min(osm), max(osm), 100)
        fig_qq.add_trace(go.Scatter(
            x=x_line, y=slope * x_line + intercept,
            mode="lines", name="Reference Line",
            line=dict(width=2, color="#e05c5c"),
        ))
    fig_qq.update_layout(height=400, xaxis_title="Theoretical Quantiles",
                         yaxis_title="Sample Quantiles", **CHART_THEME)
    st.plotly_chart(fig_qq, use_container_width=True)

with diag_r:
    if len(return_df.columns) >= 2:
        st.subheader("Correlation Heatmap")
        corr_matrix = return_df.dropna(how="all").corr()
        fig_heat    = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            colorscale="RdBu",
        ))
        fig_heat.update_layout(height=400, **CHART_THEME)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Add at least two valid tickers to view the correlation heatmap.")

# Rolling Correlation
st.subheader("Rolling Correlation")
if len(return_df.columns) < 2:
    st.info("Add at least two valid tickers to view rolling correlation.")
else:
    rc_opts = list(return_df.columns)
    rc_a    = st.selectbox("First ticker",  rc_opts, index=0,                                    key="corr_a")
    rc_b    = st.selectbox("Second ticker", rc_opts, index=1 if len(rc_opts) > 1 else 0,         key="corr_b")

    if rc_a == rc_b:
        st.info("Choose two different tickers.")
    else:
        rolling_corr = return_df[rc_a].rolling(window=corr_window).corr(return_df[rc_b])
        fig_rc       = go.Figure()
        fig_rc.add_trace(go.Scatter(
            x=rolling_corr.index, y=rolling_corr,
            mode="lines", name=f"{rc_a} vs {rc_b}",
            line=dict(width=2, color="#a78bfa"),
        ))
        fig_rc.add_hline(y=0, line_dash="dot", line_color="#6b6f87")
        fig_rc.update_layout(height=400, xaxis_title="Date", yaxis_title="Rolling Correlation",
                             yaxis=dict(range=[-1, 1], gridcolor="#1e2330"),
                             **{k: v for k, v in CHART_THEME.items() if k != "yaxis"})
        st.plotly_chart(fig_rc, use_container_width=True)

st.divider()


# ═══════════════════════════════════════════════════════════
#  SECTION 7 — DATA ACCESS
# ═══════════════════════════════════════════════════════════
st.markdown('<span class="pill">DATA</span>', unsafe_allow_html=True)
st.markdown("## 🗃️ Raw Data & Export")

with st.expander(f"View last 60 rows — {primary_ticker}"):
    st.dataframe(primary_df.tail(60), use_container_width=True)

st.download_button(
    label=f"⬇ Download {primary_ticker} data as CSV",
    data=primary_df.to_csv().encode("utf-8"),
    file_name=f"{primary_ticker.lower()}_stock_data.csv",
    mime="text/csv",
)