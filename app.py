import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy import stats

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="TERMINAL · Stock Analyzer", layout="wide", page_icon="▸")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

/* ═══════════════ RESET ═══════════════ */
[data-testid="stHeader"]      { display:none !important; }
[data-testid="stToolbar"]     { display:none !important; }
[data-testid="stDecoration"]  { display:none !important; }
.block-container { padding:2.5rem 3rem 4rem 3rem !important; max-width:1500px; }

/* ═══════════════ BASE ═══════════════ */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background: #080c10 !important;
    color: #c8d8e8 !important;
}
.stApp { background: #080c10 !important; }

/* ═══════════════ SIDEBAR ═══════════════ */
[data-testid="stSidebar"] {
    background: #0c1018 !important;
    border-right: 1px solid #1a2535 !important;
}
[data-testid="stSidebar"] * { color: #8a9bb0 !important; }
[data-testid="stSidebar"] h2 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .85rem !important; font-weight: 700 !important;
    color: #4fc3f7 !important; letter-spacing: .2em !important;
    text-transform: uppercase !important; margin-bottom: 1rem !important;
}
[data-testid="stSidebar"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .72rem !important; color: #4a5a6a !important;
    text-transform: uppercase !important; letter-spacing: .1em !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1a2535 !important; margin: 1rem 0 !important;
}
[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stNumberInput > div > div > input {
    background: #080c10 !important; color: #c8d8e8 !important;
    border: 1px solid #1a2535 !important; border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: .8rem !important;
    caret-color: #4fc3f7 !important;
}
[data-testid="stSidebar"] .stTextInput > div > div > input:focus,
[data-testid="stSidebar"] .stNumberInput > div > div > input:focus {
    border-color: #4fc3f7 !important;
    box-shadow: 0 0 0 2px rgba(79,195,247,.15) !important;
    outline: none !important;
}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
[data-testid="stSidebar"] .stDateInput > div > div > input {
    background: #080c10 !important; color: #c8d8e8 !important;
    border: 1px solid #1a2535 !important; border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: .8rem !important;
}
[data-testid="stSidebar"] .stNumberInput button {
    background: #1a2535 !important; color: #c8d8e8 !important;
    border: 1px solid #253545 !important;
}
[data-testid="stSidebar"] .stCheckbox span { color: #8a9bb0 !important; }

/* ═══════════════ MAIN INPUTS ═══════════════ */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #0c1018 !important; color: #c8d8e8 !important;
    border: 1px solid #1a2535 !important; border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: .85rem !important;
    caret-color: #4fc3f7 !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #4fc3f7 !important;
    box-shadow: 0 0 0 2px rgba(79,195,247,.12) !important;
    outline: none !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background: #0c1018 !important; color: #c8d8e8 !important;
    border: 1px solid #1a2535 !important; border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: .85rem !important;
}
[data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="menu"] {
    background: #0c1018 !important; border: 1px solid #1a2535 !important;
    box-shadow: 0 8px 32px rgba(0,0,0,.6) !important;
}
[data-baseweb="option"] {
    background: #0c1018 !important; color: #c8d8e8 !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: .82rem !important;
}
[data-baseweb="option"]:hover { background: #1a2535 !important; color: #4fc3f7 !important; }
.stNumberInput button {
    background: #1a2535 !important; color: #c8d8e8 !important;
    border: 1px solid #253545 !important;
}
.stDateInput > div > div > input {
    background: #0c1018 !important; color: #c8d8e8 !important;
    border: 1px solid #1a2535 !important; border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: .85rem !important;
}

/* ═══════════════ SLIDERS ═══════════════ */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #4fc3f7 !important; border-color: #4fc3f7 !important;
    box-shadow: 0 0 8px rgba(79,195,247,.4) !important;
}
[data-testid="stSlider"] div[class*="StyledSliderTrack"] { background: #1a2535 !important; }

/* ═══════════════ HEADINGS ═══════════════ */
h1 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 700 !important; font-size: 2.2rem !important;
    color: #e8f4ff !important; letter-spacing: .04em !important;
}
h2 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important; font-size: 1rem !important;
    color: #4fc3f7 !important; letter-spacing: .15em !important;
    text-transform: uppercase !important; margin-top: 2.5rem !important;
    margin-bottom: .8rem !important;
}
h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .78rem !important; font-weight: 500 !important;
    color: #4a5a6a !important; text-transform: uppercase !important;
    letter-spacing: .12em !important;
}

/* ═══════════════ METRIC TILES ═══════════════ */
[data-testid="stMetric"] {
    background: #0c1018 !important;
    border: 1px solid #1a2535 !important;
    border-radius: 4px !important;
    padding: 14px 18px !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .65rem !important; font-weight: 500 !important;
    color: #4a5a6a !important; text-transform: uppercase !important;
    letter-spacing: .12em !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.25rem !important; font-weight: 700 !important;
    color: #e8f4ff !important;
}

/* ═══════════════ DATAFRAMES ═══════════════ */
[data-testid="stDataFrame"] {
    border: 1px solid #1a2535 !important;
    border-radius: 4px !important;
    overflow: hidden !important;
}

/* ═══════════════ BUTTONS ═══════════════ */
.stDownloadButton button {
    background: transparent !important; color: #4fc3f7 !important;
    border: 1px solid #4fc3f7 !important; border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .78rem !important; letter-spacing: .1em !important;
    padding: 8px 20px !important; text-transform: uppercase !important;
    transition: all .2s !important;
}
.stDownloadButton button:hover {
    background: #4fc3f7 !important; color: #080c10 !important;
}

/* ═══════════════ EXPANDER ═══════════════ */
[data-testid="stExpander"] {
    border: 1px solid #1a2535 !important;
    border-radius: 4px !important; background: #0c1018 !important;
}
[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .82rem !important; font-weight: 600 !important;
    color: #8a9bb0 !important; letter-spacing: .05em !important;
}

/* ═══════════════ ALERTS ═══════════════ */
[data-testid="stAlert"] {
    border-radius: 4px !important; font-size: .85rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ═══════════════ DIVIDER ═══════════════ */
hr { border-color: #1a2535 !important; margin: 2.5rem 0 !important; }

/* ═══════════════ CAPTIONS / SMALL TEXT ═══════════════ */
[data-testid="stCaptionContainer"] p,
.stCaption { color: #4a5a6a !important; font-size: .78rem !important;
             font-family: 'IBM Plex Mono', monospace !important; }
p { color: #8a9bb0 !important; }
li { color: #8a9bb0 !important; }

/* ═══════════════ TABLE STYLES ═══════════════ */
thead tr th {
    background: #0c1018 !important; color: #4fc3f7 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .7rem !important; text-transform: uppercase !important;
    letter-spacing: .1em !important; border-bottom: 1px solid #1a2535 !important;
}
tbody tr td {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .82rem !important; color: #c8d8e8 !important;
    border-bottom: 1px solid #0f1825 !important;
}
tbody tr:hover td { background: #0f1825 !important; }

/* ═══════════════ SCROLLBAR ═══════════════ */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #080c10; }
::-webkit-scrollbar-thumb { background: #1a2535; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════
TRADING_DAYS     = 252
BENCHMARK_TICKER = "^GSPC"
DEFAULT_TICKERS  = ["AAPL", "MSFT", "NVDA"]
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
UNIVERSE_FILE_CANDIDATES = ["stock_universe.csv","ticker_universe.csv","us_stocks_universe.csv"]

# Chart palette — high contrast on dark
PAL = ["#4fc3f7","#f9a825","#ef5350","#66bb6a","#ab47bc","#26c6da","#ff7043","#8d6e63"]

# ══════════════════════════════════════════════════════════════════
#  CHART THEME
# ══════════════════════════════════════════════════════════════════
CT = dict(
    template="plotly_dark",
    paper_bgcolor="#0c1018", plot_bgcolor="#080c10",
    font=dict(family="IBM Plex Mono, monospace", color="#8a9bb0", size=11),
    xaxis=dict(gridcolor="#0f1825", linecolor="#1a2535", showgrid=True,
               tickfont=dict(family="IBM Plex Mono, monospace", color="#4a5a6a", size=10)),
    yaxis=dict(gridcolor="#0f1825", linecolor="#1a2535", showgrid=True,
               tickfont=dict(family="IBM Plex Mono, monospace", color="#4a5a6a", size=10)),
    margin=dict(l=12, r=12, t=32, b=12),
    legend=dict(bgcolor="rgba(0,0,0,0)",
                font=dict(family="IBM Plex Mono, monospace", color="#8a9bb0", size=10),
                bordercolor="#1a2535", borderwidth=1),
)

# ══════════════════════════════════════════════════════════════════
#  DATA HELPERS
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner="▸ fetching market data…")
def load_data(symbols: tuple, start: date, end: date) -> dict:
    out = {}
    for sym in symbols:
        if not sym: continue
        try:
            df = yf.download(sym, start=start, end=end + timedelta(days=1),
                             interval="1d", auto_adjust=False, progress=False, threads=False)
        except Exception:
            df = pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            df = df.sort_index()
        out[sym] = df
    return out


def valid(df: pd.DataFrame) -> bool:
    return not df.empty and {"Close","Volume"}.issubset(df.columns)


def enrich(df: pd.DataFrame, maw: int, volw: int) -> pd.DataFrame:
    o = df.copy().dropna(subset=["Close"])
    o["Ret"]    = o["Close"].pct_change()
    o["CumRet"] = (1 + o["Ret"].fillna(0)).cumprod() - 1
    o[f"MA{maw}"]  = o["Close"].rolling(maw).mean()
    o["RolVol"] = o["Ret"].rolling(volw).std() * math.sqrt(TRADING_DAYS)
    return o


def sstats(r: pd.Series, rfr: float) -> dict:
    r = r.dropna()
    nan = float("nan")
    if r.empty:
        return dict(tot=nan,mu=nan,ar=nan,av=nan,sh=nan,sk=nan,ku=nan,jbs=nan,jbp=nan,n=0)
    cum = (1+r).cumprod()-1
    mu  = float(r.mean()); sg = float(r.std())
    av  = sg*math.sqrt(TRADING_DAYS); ar = mu*TRADING_DAYS
    sh  = (ar-rfr)/av if av else nan
    jbs,jbp = stats.jarque_bera(r)
    return dict(tot=float(cum.iloc[-1]),mu=mu,ar=ar,av=av,sh=sh,
                sk=float(r.skew()),ku=float(r.kurtosis()),
                jbs=float(jbs),jbp=float(jbp),n=int(r.shape[0]))


def f(v, k="n") -> str:
    if pd.isna(v): return "—"
    if k=="$":    return f"${v:,.2f}"
    if k=="%":    return f"{v:.2%}"
    if k=="%4":   return f"{v:.4%}"
    if k=="i":    return f"{int(v):,}"
    if k=="8f":   return f"{v:.8f}"
    return f"{v:.4f}" if abs(v)<0.01 and v!=0 else f"{v:.2f}"


def rsi(close: pd.Series, p=14) -> float:
    d = close.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    rs = g / l.replace(0, np.nan)
    v  = 100 - 100/(1+rs)
    return float(v.iloc[-1]) if not v.empty else float("nan")


# ══════════════════════════════════════════════════════════════════
#  RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════
def rec_features(df: pd.DataFrame, rfr: float) -> dict:
    e  = enrich(df, 50, 30)
    r  = e["Ret"].dropna(); c = e["Close"].dropna()
    if r.empty or len(c) < 30: return {}
    tot = float(e["CumRet"].iloc[-1])
    vol = float(r.std())*math.sqrt(TRADING_DAYS)
    ar  = float(r.mean())*TRADING_DAYS
    sh  = (ar-rfr)/vol if vol else float("nan")
    m20 = float(c.iloc[-1]/c.iloc[-21]-1) if len(c)>20 else float("nan")
    m60 = float(c.iloc[-1]/c.iloc[-61]-1) if len(c)>60 else float("nan")
    ma20= float(c.rolling(20).mean().iloc[-1]) if len(c)>=20 else float("nan")
    ma50= float(c.rolling(50).mean().iloc[-1]) if len(c)>=50 else float("nan")
    r14 = rsi(c,14); last = float(c.iloc[-1])
    tr  = 0.0
    if not math.isnan(ma20): tr += 1. if last>ma20 else -1.
    if not math.isnan(ma50): tr += 1. if last>ma50 else -1.
    rs  = 0.0
    if not math.isnan(r14):
        if 45<=r14<=65: rs=1.
        elif r14>75:    rs=-1.
        elif r14<25:    rs=.5
    score = (.30*tot + .25*(0 if math.isnan(sh) else sh)
             + .20*(0 if math.isnan(m20) else m20)
             + .15*(0 if math.isnan(m60) else m60)
             + .10*tr + .05*rs
             - .10*(0 if math.isnan(vol) else vol))
    return dict(px=last,tot=tot,ar=ar,vol=vol,sh=sh,m20=m20,m60=m60,r14=r14,tr=tr,score=score)


def signal(score: float):
    """Returns (label, fg_hex, bg_hex, border_hex, arrow)"""
    if math.isnan(score):
        return "N/A",  "#4a5a6a", "#0c1018",  "#1a2535", "·"
    if score >  0.15:
        return "BUY",  "#00e676", "#001a0a",  "#00c853", "▲"
    if score < -0.05:
        return "SELL", "#ff5252", "#1a0000",  "#d50000", "▼"
    return         "HOLD", "#ffd740", "#1a1400",  "#ffab00", "◆"


@st.cache_data(ttl=3600, show_spinner="▸ scanning universe…")
def build_scan(cands: tuple, start: date, end: date, rfr: float, mx: int) -> pd.DataFrame:
    raw  = load_data(cands[:mx], start, end)
    rows = []
    for t,df in raw.items():
        if not valid(df): continue
        ft = rec_features(df, rfr)
        if ft: rows.append({"Ticker":t,**ft})
    if not rows: return pd.DataFrame()
    out = pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan).dropna(subset=["score","px"])
    return out.sort_values("score",ascending=False).reset_index(drop=True)


def get_universe(n: int):
    for fn in UNIVERSE_FILE_CANDIDATES:
        p = Path(fn)
        if p.exists():
            try:
                df = pd.read_csv(p)
                tks = list(dict.fromkeys(
                    [str(x).strip().upper() for x in df[df.columns[0]].dropna() if str(x).strip()]))
                if tks:
                    return tuple(tks[:n]), f"loaded {min(len(tks),n):,} tickers from `{fn}`"
            except Exception: pass
    fb = list(dict.fromkeys(FALLBACK_UNIVERSE))
    return tuple(fb[:min(n,len(fb))]), "no universe file found — using built-in fallback"


def weighted_port(ret_df: pd.DataFrame, wts: dict, rfr: float):
    tks = [t for t in wts if t in ret_df.columns]
    w   = np.array([wts[t] for t in tks]); w = w/w.sum()
    sub = ret_df[tks].dropna(how="all")
    pr  = pd.Series(sub.fillna(0).values @ w, index=sub.index)
    cum = (1+pr.fillna(0)).cumprod()-1
    return pr, cum, sstats(pr,rfr), dict(zip(tks,w))


def two_asset(ret_df, a, b, wa):
    pair = ret_df[[a,b]].dropna().copy(); wb=1.-wa
    if pair.empty:
        return dict(pr=pd.Series(dtype=float),cov=float("nan"),cor=float("nan"),
                    va=float("nan"),vb=float("nan"),pv=float("nan"),pvol=float("nan"),wb=wb)
    pair["P"] = wa*pair[a]+wb*pair[b]
    va=float(pair[a].var()); vb=float(pair[b].var())
    cov=float(pair[[a,b]].cov().iloc[0,1]); cor=float(pair[[a,b]].corr().iloc[0,1])
    pv=wa**2*va+wb**2*vb+2*wa*wb*cov
    return dict(pr=pair["P"],cov=cov,cor=cor,va=va,vb=vb,
                pv=pv,pvol=math.sqrt(pv) if pv>=0 else float("nan"),wb=wb)


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ▸ TERMINAL")
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:.65rem;"
        "color:#253545;letter-spacing:.2em;margin-top:-12px'>STOCK ANALYZER v2</p>",
        unsafe_allow_html=True)
    st.divider()
    raw_input    = st.text_input("TICKERS", value=", ".join(DEFAULT_TICKERS),
                                 placeholder="AAPL, MSFT, TSLA …")
    user_tickers = list(dict.fromkeys([t.strip().upper() for t in raw_input.split(",") if t.strip()]))
    inc_bench    = st.checkbox("Include S&P 500", value=True)
    show_scan    = st.checkbox("Show AI Scanner", value=True)
    st.divider()
    pool_size  = st.number_input("UNIVERSE POOL SIZE", 100, 10000, 10000, 100)
    scan_limit = st.slider("SCAN LIMIT", 50, 500, 200, 25)
    st.divider()
    d0 = date.today()-timedelta(days=365)
    start_date = st.date_input("START DATE", value=d0)
    end_date   = st.date_input("END DATE",   value=date.today())
    st.divider()
    maw  = st.slider("MA WINDOW",       5,  200,  50,  5)
    volw = st.slider("VOL WINDOW",      10, 120,  30,  5)
    corw = st.slider("CORR WINDOW",     10, 120,  30,  5)
    qqn  = st.slider("Q-Q SAMPLE",      50, 500, 250, 25)
    rfr  = st.number_input("RISK-FREE RATE %", 0., 20., 4.5, .1) / 100

if start_date >= end_date:
    st.sidebar.error("start must precede end"); st.stop()
if not user_tickers:
    st.info("▸ enter tickers in the sidebar"); st.stop()

# ══════════════════════════════════════════════════════════════════
#  FETCH
# ══════════════════════════════════════════════════════════════════
fetch_list = list(user_tickers)
if inc_bench and BENCHMARK_TICKER not in fetch_list:
    fetch_list.append(BENCHMARK_TICKER)

try:
    raw = load_data(tuple(fetch_list), start_date, end_date)
except Exception as e:
    st.error("data fetch failed"); st.exception(e); st.stop()

assets: dict = {}
bad: list = []
for t in user_tickers:
    df = raw.get(t, pd.DataFrame())
    if valid(df): assets[t] = enrich(df, maw, volw)
    else: bad.append(t)

if bad: st.warning(f"no data: {', '.join(bad)}")
if not assets: st.error("no valid data returned"); st.stop()

bench = pd.DataFrame()
if inc_bench:
    br = raw.get(BENCHMARK_TICKER, pd.DataFrame())
    if valid(br): bench = enrich(br, maw, volw)

pt  = next(iter(assets)); pdf = assets[pt]
N   = len(assets)
ret = pd.concat([df["Ret"].rename(t) for t,df in assets.items()], axis=1).dropna(how="all")
cum = (1+ret.fillna(0)).cumprod()-1

eq_w              = {t:1/N for t in assets}
_, eq_cum, eq_st, _ = weighted_port(ret, eq_w, rfr)
rc  = pdf["Ret"].dropna()

# compute signals for every user ticker
sigs = {}
for t,df in assets.items():
    ft    = rec_features(df, rfr)
    sc    = ft.get("score", float("nan")) if ft else float("nan")
    lb,fg,bg,bd,ar = signal(sc)
    sigs[t] = dict(score=sc, label=lb, fg=fg, bg=bg, bd=bd, arrow=ar, ft=ft)


# ══════════════════════════════════════════════════════════════════
#  HEADER BAR
# ══════════════════════════════════════════════════════════════════
ticker_pills = "  ".join(
    f'<span style="font-family:IBM Plex Mono,monospace;font-size:.85rem;'
    f'font-weight:700;color:#4fc3f7;background:#0c1018;border:1px solid #1a2535;'
    f'padding:2px 10px;border-radius:2px">{t}</span>'
    for t in assets
)
st.markdown(f"""
<div style="border-bottom:1px solid #1a2535;padding-bottom:16px;margin-bottom:4px">
  <div style="display:flex;align-items:baseline;gap:16px;flex-wrap:wrap">
    <span style="font-family:IBM Plex Mono,monospace;font-size:1.6rem;font-weight:700;
                 color:#e8f4ff;letter-spacing:.06em">▸ TERMINAL</span>
    <span style="font-family:IBM Plex Mono,monospace;font-size:.72rem;color:#253545;
                 letter-spacing:.2em">STOCK ANALYZER</span>
  </div>
  <div style="margin-top:10px;display:flex;align-items:center;gap:8px;flex-wrap:wrap">
    {ticker_pills}
    <span style="font-family:IBM Plex Mono,monospace;font-size:.72rem;color:#253545">
      {start_date:%Y-%m-%d} → {end_date:%Y-%m-%d}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  SECTION 1 ── SIGNAL DASHBOARD (user tickers)
# ══════════════════════════════════════════════════════════════════
st.markdown("## 01 / SIGNAL DASHBOARD")
st.markdown(
    "<p style='font-family:IBM Plex Mono,monospace;font-size:.72rem;color:#253545;"
    "margin-top:-8px;margin-bottom:16px'>"
    "BUY / HOLD / SELL signals computed for each of your tickers using the composite model</p>",
    unsafe_allow_html=True)

cols = st.columns(N)
for i,(t,sg) in enumerate(sigs.items()):
    df_t  = assets[t]; st_t = sstats(df_t["Ret"], rfr)
    ft    = sg["ft"]; lb = sg["label"]
    fg    = sg["fg"]; bg = sg["bg"]; bd = sg["bd"]; ar = sg["arrow"]
    sc    = sg["score"]
    r14_v = ft.get("r14", float("nan")) if ft else float("nan")
    m20_v = ft.get("m20", float("nan")) if ft else float("nan")
    tr_v  = ft.get("tr",  float("nan")) if ft else float("nan")

    tr_str = {2:"ABOVE BOTH", 1:"ABOVE ONE", 0:"MIXED", -1:"BELOW ONE", -2:"BELOW BOTH"}.get(
        int(tr_v) if not math.isnan(tr_v) else 99, "N/A")

    with cols[i]:
        st.markdown(f"""
<div style="background:{bg};border:1px solid {bd};border-radius:4px;
            padding:20px 16px 16px 16px;font-family:'IBM Plex Mono',monospace">

  <!-- signal label -->
  <div style="font-size:.6rem;color:{fg};opacity:.7;letter-spacing:.2em;
              margin-bottom:2px">SIGNAL</div>
  <div style="font-size:2rem;font-weight:700;color:{fg};line-height:1;
              margin-bottom:10px">{ar} {lb}</div>

  <!-- ticker + price -->
  <div style="font-size:1rem;font-weight:700;color:#e8f4ff;
              margin-bottom:2px">{t}</div>
  <div style="font-size:1.2rem;font-weight:700;color:#c8d8e8;
              margin-bottom:14px">{f(float(df_t["Close"].iloc[-1]),"$")}</div>

  <!-- separator -->
  <div style="border-top:1px solid {bd};opacity:.3;margin-bottom:12px"></div>

  <!-- stats grid -->
  <table style="width:100%;border-collapse:collapse">
    <tr>
      <td style="padding:3px 0;font-size:.62rem;color:#4a5a6a">SCORE</td>
      <td style="padding:3px 0;font-size:.62rem;color:#4a5a6a">SHARPE</td>
    </tr>
    <tr>
      <td style="padding:2px 0 8px 0;font-size:.9rem;font-weight:700;color:#e8f4ff">{f(sc)}</td>
      <td style="padding:2px 0 8px 0;font-size:.9rem;font-weight:700;color:#e8f4ff">{f(st_t["sh"])}</td>
    </tr>
    <tr>
      <td style="padding:3px 0;font-size:.62rem;color:#4a5a6a">RSI 14</td>
      <td style="padding:3px 0;font-size:.62rem;color:#4a5a6a">MOM 20D</td>
    </tr>
    <tr>
      <td style="padding:2px 0 8px 0;font-size:.9rem;font-weight:700;color:#e8f4ff">{f(r14_v)}</td>
      <td style="padding:2px 0 8px 0;font-size:.9rem;font-weight:700;color:#e8f4ff">{f(m20_v,"%")}</td>
    </tr>
    <tr>
      <td style="padding:3px 0;font-size:.62rem;color:#4a5a6a">ANN RET</td>
      <td style="padding:3px 0;font-size:.62rem;color:#4a5a6a">ANN VOL</td>
    </tr>
    <tr>
      <td style="padding:2px 0 8px 0;font-size:.9rem;font-weight:700;color:#e8f4ff">{f(st_t["ar"],"%")}</td>
      <td style="padding:2px 0 8px 0;font-size:.9rem;font-weight:700;color:#e8f4ff">{f(st_t["av"],"%")}</td>
    </tr>
    <tr>
      <td colspan="2" style="padding:3px 0;font-size:.62rem;color:#4a5a6a">MA TREND</td>
    </tr>
    <tr>
      <td colspan="2" style="padding:2px 0;font-size:.82rem;font-weight:700;color:#e8f4ff">{tr_str}</td>
    </tr>
  </table>
</div>
""", unsafe_allow_html=True)

with st.expander("▸ scoring methodology", expanded=False):
    st.markdown("""
```
COMPOSITE SCORE WEIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total Return          +30%
  Sharpe Ratio          +25%
  20-Day Momentum       +20%
  60-Day Momentum       +15%
  Trend Score           +10%   (price vs MA20/MA50)
  RSI Score              +5%   (45-65 healthy, >75 penalty, <25 bonus)
  Volatility Penalty    -10%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BUY   score >  0.15
  HOLD  score   -0.05 to 0.15
  SELL  score < -0.05
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
""")

st.divider()


# ══════════════════════════════════════════════════════════════════
#  SECTION 2 ── KEY METRICS
# ══════════════════════════════════════════════════════════════════
st.markdown(f"## 02 / KEY METRICS  [{pt}]")
pst = sstats(pdf["Ret"], rfr)

m1,m2,m3,m4 = st.columns(4)
m1.metric("CLOSE",     f(float(pdf["Close"].iloc[-1]),"$"))
m2.metric("TOTAL RET", f(pst["tot"],"%"))
m3.metric("ANN RET",   f(pst["ar"],"%"))
m4.metric("SHARPE",    f(pst["sh"]))

m5,m6,m7,m8 = st.columns(4)
m5.metric("ANN VOL",   f(pst["av"],"%"))
m6.metric("SKEWNESS",  f(pst["sk"]))
m7.metric("KURTOSIS",  f(pst["ku"]))
m8.metric("AVG DAILY", f(pst["mu"],"%4"))

m9,m10,m11,m12 = st.columns(4)
m9.metric("PERIOD HIGH",  f(float(pdf["Close"].max()),"$"))
m10.metric("PERIOD LOW",  f(float(pdf["Close"].min()),"$"))
m11.metric("JB STAT",     f(pst["jbs"]))
m12.metric("JB P-VALUE",  f(pst["jbp"]))

st.divider()


# ══════════════════════════════════════════════════════════════════
#  SECTION 3 ── PRICE CHART
# ══════════════════════════════════════════════════════════════════
st.markdown(f"## 03 / PRICE & MOVING AVERAGE  [{pt}]")

fp = go.Figure()
fp.add_trace(go.Scatter(x=pdf.index, y=pdf["Close"],
    mode="lines", name="CLOSE",
    line=dict(width=2, color="#4fc3f7")))
fp.add_trace(go.Scatter(x=pdf.index, y=pdf[f"MA{maw}"],
    mode="lines", name=f"MA{maw}",
    line=dict(width=1.5, dash="dash", color="#f9a825")))
fp.update_layout(height=420, xaxis_title="DATE", yaxis_title="USD", **CT)
st.plotly_chart(fp, use_container_width=True)

if maw > len(pdf):
    st.warning(f"MA{maw} exceeds available data ({len(pdf)} days)")

st.divider()


# ══════════════════════════════════════════════════════════════════
#  SECTION 4 ── PERFORMANCE & RISK
# ══════════════════════════════════════════════════════════════════
st.markdown("## 04 / PERFORMANCE & RISK")

c4a, c4b = st.columns(2, gap="large")
with c4a:
    st.markdown("#### DAILY VOLUME")
    fv = go.Figure()
    fv.add_trace(go.Bar(x=pdf.index, y=pdf["Volume"], name="VOLUME",
        marker_color="#4fc3f7", opacity=0.8))
    fv.update_layout(height=320, xaxis_title="DATE", yaxis_title="SHARES", **CT)
    st.plotly_chart(fv, use_container_width=True)

with c4b:
    st.markdown("#### RETURN DISTRIBUTION")
    fh = go.Figure()
    fh.add_trace(go.Histogram(x=rc, nbinsx=60, opacity=0.8,
        histnorm="probability density", marker_color="#4fc3f7", name="RETURNS"))
    if not rc.empty:
        xr=np.linspace(float(rc.min()),float(rc.max()),200)
        mu_=float(rc.mean()); sg_=float(rc.std())
        if sg_>0:
            fh.add_trace(go.Scatter(x=xr, y=stats.norm.pdf(xr,mu_,sg_),
                mode="lines", name="NORMAL FIT",
                line=dict(width=2.5, color="#ef5350")))
    fh.update_layout(height=320, xaxis_title="DAILY RETURN", yaxis_title="DENSITY", **CT)
    st.plotly_chart(fh, use_container_width=True)
    st.caption(f"Jarque-Bera  stat={pst['jbs']:.2f}  p={pst['jbp']:.4f}")

c4c, c4d = st.columns(2, gap="large")
with c4c:
    st.markdown("#### CUMULATIVE RETURN")
    fc = go.Figure()
    fc.add_trace(go.Scatter(x=pdf.index, y=pdf["CumRet"],
        mode="lines", fill="tozeroy", name="CUM RET",
        line=dict(color="#66bb6a", width=2),
        fillcolor="rgba(102,187,106,.08)"))
    fc.update_layout(height=360, xaxis_title="DATE", yaxis_title="RETURN",
                     yaxis_tickformat=".0%", **CT)
    st.plotly_chart(fc, use_container_width=True)

with c4d:
    st.markdown("#### ROLLING VOLATILITY")
    eq_vol = ret.fillna(0).mean(axis=1).rolling(volw).std() * math.sqrt(TRADING_DAYS)
    frv = go.Figure()
    frv.add_trace(go.Scatter(x=pdf.index, y=pdf["RolVol"],
        mode="lines", name=f"{pt} {volw}D VOL",
        line=dict(width=2, color="#4fc3f7")))
    frv.add_trace(go.Scatter(x=eq_vol.index, y=eq_vol,
        mode="lines", name="EQ-WT PORTFOLIO",
        line=dict(width=2, dash="dash", color="#ef5350")))
    frv.update_layout(height=360, xaxis_title="DATE", yaxis_title="ANN VOL",
                      yaxis_tickformat=".0%", **CT)
    st.plotly_chart(frv, use_container_width=True)

st.divider()


# ══════════════════════════════════════════════════════════════════
#  SECTION 5 ── COMPARISON
# ══════════════════════════════════════════════════════════════════
st.markdown("## 05 / MULTI-TICKER COMPARISON")

fcmp = go.Figure()
for i,t in enumerate(cum.columns):
    fcmp.add_trace(go.Scatter(x=cum.index, y=cum[t],
        mode="lines", name=t,
        line=dict(width=2.5, color=PAL[i%len(PAL)])))
if not bench.empty:
    fcmp.add_trace(go.Scatter(x=bench.index, y=bench["CumRet"],
        mode="lines", name="S&P 500",
        line=dict(dash="dot", color="#253545", width=2)))
fcmp.add_trace(go.Scatter(x=eq_cum.index, y=eq_cum,
    mode="lines", name="EQ-WT PORTFOLIO",
    line=dict(width=3, color="#ab47bc", dash="dashdot")))
fcmp.update_layout(height=450, xaxis_title="DATE", yaxis_title="CUM RETURN",
                   yaxis_tickformat=".0%", **CT)
st.plotly_chart(fcmp, use_container_width=True)

st.markdown("#### SUMMARY STATISTICS")
rows = []
for t,df in assets.items():
    s = sstats(df["Ret"], rfr)
    rows.append({"TICKER":t, "SIGNAL":sigs[t]["label"],
                 "CLOSE":f(float(df["Close"].iloc[-1]),"$"),
                 "TOTAL RET":f(s["tot"],"%"), "ANN RET":f(s["ar"],"%"),
                 "ANN VOL":f(s["av"],"%"), "SHARPE":f(s["sh"]),
                 "SKEW":f(s["sk"]), "KURT":f(s["ku"]),
                 "JB P":f(s["jbp"]), "N":f(s["n"],"i")})
if not bench.empty:
    bs = sstats(bench["Ret"], rfr)
    rows.append({"TICKER":"^GSPC", "SIGNAL":"—",
                 "CLOSE":f(float(bench["Close"].iloc[-1]),"$"),
                 "TOTAL RET":f(bs["tot"],"%"), "ANN RET":f(bs["ar"],"%"),
                 "ANN VOL":f(bs["av"],"%"), "SHARPE":f(bs["sh"]),
                 "SKEW":f(bs["sk"]), "KURT":f(bs["ku"]),
                 "JB P":f(bs["jbp"]), "N":f(bs["n"],"i")})
st.dataframe(pd.DataFrame(rows).set_index("TICKER"), use_container_width=True, height=280)

st.divider()


# ══════════════════════════════════════════════════════════════════
#  SECTION 6 ── PORTFOLIO BUILDER  (custom weights)
# ══════════════════════════════════════════════════════════════════
st.markdown("## 06 / PORTFOLIO BUILDER")
st.markdown(
    "<p style='font-family:IBM Plex Mono,monospace;font-size:.72rem;color:#253545;"
    "margin-top:-8px;margin-bottom:16px'>"
    "set custom weights — auto-normalised to 100%</p>",
    unsafe_allow_html=True)

# Weight sliders
wt_in = {}
sl_cols = st.columns(N)
for i,t in enumerate(assets):
    with sl_cols[i]:
        wt_in[t] = st.slider(t, 0, 100, 100//N, 1, key=f"wt_{t}")

tot_w = sum(wt_in.values())
cw    = ({t:1/N for t in assets} if tot_w==0
         else {t:wt_in[t]/tot_w for t in assets})

# Normalised weight display
wc = st.columns(N)
for i,(t,w) in enumerate(cw.items()):
    sg = sigs[t]
    with wc[i]:
        bar_pct = int(w*100)
        st.markdown(f"""
<div style="background:#0c1018;border:1px solid #1a2535;border-radius:4px;
            padding:10px 12px;margin-top:6px;font-family:'IBM Plex Mono',monospace">
  <div style="font-size:.6rem;color:#4a5a6a;letter-spacing:.15em;margin-bottom:4px">{t} WEIGHT</div>
  <div style="font-size:1.5rem;font-weight:700;color:#e8f4ff;margin-bottom:8px">{w:.1%}</div>
  <div style="height:4px;background:#1a2535;border-radius:2px;margin-bottom:8px">
    <div style="height:4px;width:{bar_pct}%;background:{sg['fg']};border-radius:2px"></div>
  </div>
  <div style="font-size:.75rem;font-weight:700;color:{sg['fg']}">{sg['arrow']} {sg['label']}</div>
</div>
""", unsafe_allow_html=True)

if tot_w == 0:
    st.warning("all weights zero — falling back to equal weight")

cr, ccum, cst, nw = weighted_port(ret, cw, rfr)
cust_vol = cr.rolling(volw).std() * math.sqrt(TRADING_DAYS)

st.markdown("#### CUSTOM PORTFOLIO METRICS")
pm1,pm2,pm3,pm4 = st.columns(4)
pm1.metric("TOTAL RETURN", f(cst["tot"],"%"))
pm2.metric("ANN RETURN",   f(cst["ar"],"%"))
pm3.metric("SHARPE",       f(cst["sh"]))
pm4.metric("ANN VOL",      f(cst["av"],"%"))
pm5,pm6,pm7,pm8 = st.columns(4)
pm5.metric("SKEWNESS",     f(cst["sk"]))
pm6.metric("KURTOSIS",     f(cst["ku"]))
pm7.metric("JB P-VALUE",   f(cst["jbp"]))
pm8.metric("OBSERVATIONS", f(cst["n"],"i"))

# Performance chart
fpt = go.Figure()
fpt.add_trace(go.Scatter(x=ccum.index, y=ccum,
    mode="lines", name="CUSTOM PORTFOLIO",
    line=dict(width=3, color="#4fc3f7")))
fpt.add_trace(go.Scatter(x=eq_cum.index, y=eq_cum,
    mode="lines", name="EQUAL WEIGHT",
    line=dict(width=2, dash="dash", color="#f9a825")))
if not bench.empty:
    fpt.add_trace(go.Scatter(x=bench.index, y=bench["CumRet"],
        mode="lines", name="S&P 500",
        line=dict(dash="dot", color="#253545", width=2)))
fpt.update_layout(height=400, xaxis_title="DATE", yaxis_title="CUM RETURN",
                  yaxis_tickformat=".0%", **CT)
st.plotly_chart(fpt, use_container_width=True)

# Donut allocation chart
fpie = go.Figure(go.Pie(
    labels=list(nw.keys()), values=list(nw.values()), hole=.6,
    marker_colors=PAL[:len(nw)],
    textfont=dict(family="IBM Plex Mono, monospace", size=12, color="#e8f4ff"),
    textinfo="label+percent",
))
fpie.update_layout(
    height=280, showlegend=False, paper_bgcolor="#0c1018",
    margin=dict(l=10,r=10,t=10,b=10),
    font=dict(family="IBM Plex Mono, monospace", color="#8a9bb0"),
    annotations=[dict(text="ALLOC", x=.5, y=.5, font_size=11,
                      font_color="#4a5a6a", font_family="IBM Plex Mono, monospace",
                      showarrow=False)])
st.plotly_chart(fpie, use_container_width=True)

# ── Two-Asset Explorer ──
st.markdown("#### TWO-ASSET EXPLORER")
if len(ret.columns) < 2:
    st.info("add at least two tickers")
else:
    tc1,tc2,tc3 = st.columns(3)
    opts  = list(ret.columns)
    aa    = tc1.selectbox("ASSET A", opts, index=0)
    ab    = tc2.selectbox("ASSET B", opts, index=1 if len(opts)>1 else 0)
    wa_pc = tc3.slider("WEIGHT A %", 0, 100, 50, 5, key="ta")

    if aa==ab:
        st.error("assets must differ")
    else:
        wa = wa_pc/100
        ta = two_asset(ret, aa, ab, wa)
        wb_ = ta["wb"]
        if ta["pr"].empty:
            st.error("insufficient overlapping data")
        else:
            ta_s = sstats(ta["pr"], rfr)
            ta_c = (1+ta["pr"].fillna(0)).cumprod()-1

            tm1,tm2,tm3,tm4 = st.columns(4)
            tm1.metric("WEIGHT A",        f"{wa:.0%}")
            tm2.metric("WEIGHT B",        f"{wb_:.0%}")
            tm3.metric("PORT VARIANCE",   f(ta["pv"],"8f"))
            tm4.metric("PORT VOLATILITY", f"{ta['pvol']:.4%}")
            tm5,tm6,tm7,tm8 = st.columns(4)
            tm5.metric("COVARIANCE",  f(ta["cov"],"8f"))
            tm6.metric("CORRELATION", f"{ta['cor']:.4f}")
            tm7.metric("TOTAL RET",   f(ta_s["tot"],"%"))
            tm8.metric("SHARPE",      f(ta_s["sh"]))

            st.markdown(
                f"`σ²ₚ = {wa:.2f}²×{ta['va']:.6f} + {wb_:.2f}²×{ta['vb']:.6f} "
                f"+ 2×{wa:.2f}×{wb_:.2f}×{ta['cov']:.6f} = {ta['pv']:.8f}`")

            fta = go.Figure()
            fta.add_trace(go.Scatter(x=ta_c.index, y=ta_c,
                mode="lines", name="TWO-ASSET", line=dict(width=3, color="#4fc3f7")))
            fta.add_trace(go.Scatter(x=cum.index, y=cum[aa],
                mode="lines", name=aa, line=dict(dash="dash", color="#f9a825")))
            fta.add_trace(go.Scatter(x=cum.index, y=cum[ab],
                mode="lines", name=ab, line=dict(dash="dot", color="#ef5350")))
            fta.update_layout(height=400, xaxis_title="DATE", yaxis_title="CUM RETURN",
                               yaxis_tickformat=".0%", **CT)
            st.plotly_chart(fta, use_container_width=True)

st.divider()


# ══════════════════════════════════════════════════════════════════
#  SECTION 7 ── AI SCANNER
# ══════════════════════════════════════════════════════════════════
if show_scan:
    st.markdown("## 07 / AI SCANNER")
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:.72rem;color:#253545;"
        "margin-top:-8px;margin-bottom:16px'>"
        "program-picked candidates — independent of your tickers above</p>",
        unsafe_allow_html=True)

    univ, umsg = get_universe(int(pool_size))
    st.caption(f"▸ {umsg}")
    st.caption(f"▸ pool {int(pool_size):,}  ·  live scan {scan_limit:,}  ·  cached 1h")

    sdf = build_scan(univ, start_date, end_date, rfr, scan_limit)

    if sdf.empty:
        st.warning("no results for current scan settings")
    else:
        def fmt_scan(df_in):
            d = df_in[["Ticker","score","sh","m20","m60","ar","r14","px"]].copy()
            d.insert(1,"SIGNAL", d["score"].map(lambda x: signal(x)[0]))
            d["score"] = d["score"].map(f)
            d["sh"]    = d["sh"].map(f)
            d["m20"]   = d["m20"].map(lambda x: f(x,"%"))
            d["m60"]   = d["m60"].map(lambda x: f(x,"%"))
            d["ar"]    = d["ar"].map(lambda x: f(x,"%"))
            d["r14"]   = d["r14"].map(f)
            d["px"]    = d["px"].map(lambda x: f(x,"$"))
            d.columns  = ["TICKER","SIGNAL","SCORE","SHARPE","MOM 20D","MOM 60D","ANN RET","RSI 14","PRICE"]
            return d

        uc, dc = st.columns(2, gap="large")
        with uc:
            st.markdown(
                "<span style='font-family:IBM Plex Mono,monospace;font-size:.72rem;"
                "color:#00e676;letter-spacing:.1em'>▲ TOP 10 BUY CANDIDATES</span>",
                unsafe_allow_html=True)
            st.dataframe(fmt_scan(sdf.head(10)), use_container_width=True, height=380)
        with dc:
            st.markdown(
                "<span style='font-family:IBM Plex Mono,monospace;font-size:.72rem;"
                "color:#ff5252;letter-spacing:.1em'>▼ TOP 10 SELL / AVOID</span>",
                unsafe_allow_html=True)
            st.dataframe(fmt_scan(sdf.tail(10).sort_values("score")),
                         use_container_width=True, height=380)

    st.divider()


# ══════════════════════════════════════════════════════════════════
#  SECTION 8 ── STATISTICAL DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════
st.markdown("## 08 / STATISTICAL DIAGNOSTICS")

dgl, dgr = st.columns(2, gap="large")

with dgl:
    st.markdown("#### Q-Q PLOT")
    qqr = rc.tail(min(len(rc), qqn))
    fqq = go.Figure()
    if len(qqr)>1:
        osm,osr = stats.probplot(qqr, dist="norm", fit=False)
        sl,ic,_ = stats.probplot(qqr, dist="norm", fit=True)[1]
        fqq.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="OBSERVED",
            marker=dict(color="#4fc3f7", size=4, opacity=.7)))
        xl = np.linspace(min(osm), max(osm), 100)
        fqq.add_trace(go.Scatter(x=xl, y=sl*xl+ic, mode="lines", name="REFERENCE",
            line=dict(width=2, color="#ef5350")))
    fqq.update_layout(height=380, xaxis_title="THEORETICAL QUANTILES",
                      yaxis_title="SAMPLE QUANTILES", **CT)
    st.plotly_chart(fqq, use_container_width=True)

with dgr:
    if len(ret.columns)>=2:
        st.markdown("#### CORRELATION HEATMAP")
        cm  = ret.dropna(how="all").corr()
        fhm = go.Figure(go.Heatmap(
            z=cm.values, x=cm.columns, y=cm.index,
            zmin=-1, zmax=1, colorscale="RdBu_r",
            text=np.round(cm.values,2), texttemplate="%{text}",
            textfont=dict(family="IBM Plex Mono, monospace", size=13, color="#e8f4ff"),
        ))
        fhm.update_layout(height=380, **CT)
        st.plotly_chart(fhm, use_container_width=True)
    else:
        st.info("add ≥2 tickers for heatmap")

st.markdown("#### ROLLING CORRELATION")
if len(ret.columns)<2:
    st.info("add ≥2 tickers for rolling correlation")
else:
    rc_opts = list(ret.columns)
    rcc1,rcc2 = st.columns(2)
    rca = rcc1.selectbox("TICKER A", rc_opts, index=0,                              key="rca")
    rcb = rcc2.selectbox("TICKER B", rc_opts, index=1 if len(rc_opts)>1 else 0,     key="rcb")
    if rca==rcb:
        st.info("choose two different tickers")
    else:
        rlc = ret[rca].rolling(corw).corr(ret[rcb])
        frc = go.Figure()
        frc.add_trace(go.Scatter(x=rlc.index, y=rlc,
            mode="lines", name=f"{rca} / {rcb}",
            line=dict(width=2, color="#ab47bc")))
        frc.add_hline(y=0, line_dash="dot", line_color="#253545", line_width=1)
        frc.update_layout(height=380, xaxis_title="DATE",
                          yaxis_title="ROLLING CORRELATION",
                          yaxis=dict(range=[-1,1], gridcolor="#0f1825"),
                          **{k:v for k,v in CT.items() if k!="yaxis"})
        st.plotly_chart(frc, use_container_width=True)

st.divider()


# ══════════════════════════════════════════════════════════════════
#  SECTION 9 ── DATA EXPORT
# ══════════════════════════════════════════════════════════════════
st.markdown("## 09 / DATA EXPORT")

with st.expander(f"▸ last 60 rows  [{pt}]"):
    st.dataframe(pdf.tail(60), use_container_width=True)

st.download_button(
    label=f"▸ DOWNLOAD {pt} CSV",
    data=pdf.to_csv().encode("utf-8"),
    file_name=f"{pt.lower()}_data.csv",
    mime="text/csv",
)