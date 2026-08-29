"""
pages/_shared.py
----------------
Shared utilities imported by every page module.

Contains:
  - Global CSS injection (dark theme)
  - Shared constants (tickers, dates, colour palette)
  - @st.cache_data / @st.cache_resource helpers
  - Sidebar footer renderer with Gemini API Key setup and cached tickers.

Importing this module in a page file is the single source of truth
for all cross-page configuration, keeping pages DRY and consistent.
"""

import datetime
import sys
import os
import requests
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Path setup (safe to call multiple times) ──────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.downloader import download_stock_data, validate_ticker
from src.preprocess import preprocess_pipeline
from src.feature_engineering import add_features
from src.evaluator import load_metrics
from src.trainer import load_model
from src.predictor import MODEL_REGISTRY
from src.database import get_all_cached_tickers, cached_days_available

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_START: str = "2020-01-01"
DEFAULT_TICKER: str = "AAPL"
SUPPORTED_TICKERS: list[str] = [
    "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA",
    "META", "TCS.NS", "RELIANCE.NS", "INFY.NS",
]



PALETTE: dict[str, str] = {
    "primary":   "#6C63FF",
    "secondary": "#FF6584",
    "accent":    "#43E97B",
    "neutral":   "#A8B2D8",
    "bg_dark":   "#0D1117",
    "bg_card":   "#161B22",
    "bg_card2":  "#1C2230",
    "text":      "#E6EDF3",
    "text_muted":"#8B949E",
    "border":    "#30363D",
    "grid":      "#21262D",
}

# ─── Global CSS ───────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-dark:    #0D1117;
    --bg-card:    #161B22;
    --bg-card2:   #1C2230;
    --primary:    #6C63FF;
    --secondary:  #FF6584;
    --accent:     #43E97B;
    --neutral:    #A8B2D8;
    --text:       #E6EDF3;
    --text-muted: #8B949E;
    --border:     #30363D;
    --radius:     14px;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-dark) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0D1117 0%, #161B22 100%) !important;
    border-right: 1px solid var(--border);
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 14px 16px !important;
    height: 100% !important;
    min-height: 110px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.74rem !important;
    white-space: nowrap !important;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    white-space: normal !important;
    word-break: break-word !important;
    line-height: 1.25 !important;
}
[data-testid="stMetricValue"] > div {
    white-space: normal !important;
    text-overflow: unset !important;
    overflow: visible !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), #8B5CF6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    letter-spacing: 0.4px !important;
    transition: all 0.22s ease !important;
    box-shadow: 0 4px 15px rgba(108,99,255,0.30) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(108,99,255,0.45) !important;
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ── Section header ── */
.section-header {
    background: linear-gradient(90deg, rgba(108,99,255,0.13) 0%, rgba(13,17,23,0) 100%);
    border-left: 4px solid var(--primary);
    padding: 10px 18px;
    border-radius: 0 var(--radius) var(--radius) 0;
    margin-bottom: 18px;
}
.section-header h2 { margin: 0; color: var(--text); font-size: 1.25rem; font-weight: 600; }

/* ── Hero banner ── */
.hero-banner {
    display: flex;
    align-items: center;
    background: linear-gradient(135deg, #131824 0%, #0F172A 50%, #1A173B 100%);
    border: 1px solid rgba(108, 99, 255, 0.28);
    border-radius: 18px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 380px;
    height: 380px;
    background: radial-gradient(circle, rgba(108, 99, 255, 0.18) 0%, rgba(67, 233, 123, 0.05) 50%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-icon-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 52px;
    height: 52px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(108, 99, 255, 0.22), rgba(67, 233, 123, 0.15));
    border: 1px solid rgba(108, 99, 255, 0.40);
    font-size: 1.75rem;
    margin-right: 20px;
    flex-shrink: 0;
    box-shadow: 0 4px 16px rgba(108, 99, 255, 0.25);
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #FFFFFF 0%, #F1F5F9 50%, #A5B4FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
    letter-spacing: -0.4px;
}
.hero-subtitle {
    font-size: 0.94rem;
    color: #94A3B8;
    margin: 0;
    max-width: 680px;
    line-height: 1.55;
}

/* ── Badges ── */
.badge {
    display: inline-block; padding: 3px 11px;
    border-radius: 20px; font-size: 0.76rem; font-weight: 600; margin: 2px;
}
.badge-green  { background: rgba(67,233,123,0.13); color: #43E97B; border: 1px solid rgba(67,233,123,0.28); }
.badge-red    { background: rgba(255,101,132,0.13); color: #FF6584; border: 1px solid rgba(255,101,132,0.28); }
.badge-purple { background: rgba(108,99,255,0.13); color: #6C63FF; border: 1px solid rgba(108,99,255,0.28); }
.badge-blue   { background: rgba(56,189,248,0.13); color: #38BDF8; border: 1px solid rgba(56,189,248,0.28); }

/* ── Info card ── */
.info-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 18px 22px;
}

/* ── Prediction box ── */
.predict-box {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 28px 32px;
    text-align: center; margin-top: 20px;
}
.predict-box .label { color: var(--text-muted); font-size: 0.83rem; text-transform: uppercase; letter-spacing: 1px; }
.predict-box .value { font-size: 2.5rem; font-weight: 700; }
.predict-box .green { color: var(--accent); }
.predict-box .red   { color: var(--secondary); }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 22px 0 !important; }

/* ── DataFrame ── */
.dataframe thead th { background: var(--bg-card2) !important; color: var(--primary) !important; }
.dataframe tbody tr:hover { background: rgba(108,99,255,0.06) !important; }

/* ── Tab bar ── */
[data-testid="stTabs"] button[data-baseweb="tab"] {
    color: var(--text-muted) !important;
    font-weight: 500;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom-color: var(--primary) !important;
}
</style>
"""


def inject_css() -> None:
    """Inject the shared dark-theme CSS. Call once at the top of each page."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def get_gemini_api_key() -> str:
    """Retrieve Gemini API Key from session_state or environment variables."""
    if "gemini_api_key" in st.session_state and st.session_state["gemini_api_key"].strip():
        return st.session_state["gemini_api_key"].strip()
    return os.environ.get("GEMINI_API_KEY", "")


def render_sidebar_footer() -> None:
    """Render a consistent footer in the sidebar with model status, cached tickers, and Gemini API key config."""
    with st.sidebar:
        st.divider()

        # Model status
        st.markdown("**🤖 Model Status**")
        for name, path in MODEL_REGISTRY.items():
            ok = path.exists()
            badge = "badge-green" if ok else "badge-red"
            label = "● Ready" if ok else "● Missing"
            st.markdown(
                f'<span class="badge {badge}">{name}: {label}</span>',
                unsafe_allow_html=True,
            )

        st.divider()

        # Database stats / Cached tickers
        st.markdown("**📦 SQLite Cache Stats**")
        cached_tickers = get_all_cached_tickers()
        if cached_tickers:
            st.markdown(f"**Cached Tickers:**")
            ticker_badges = []
            for t in cached_tickers:
                cnt = cached_days_available(t)
                ticker_badges.append(f'<span class="badge badge-blue">{t} ({cnt}d)</span>')
            st.markdown(" ".join(ticker_badges), unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#8B949E;font-size:0.8rem;'>No cached data yet.</span>", unsafe_allow_html=True)

        st.divider()

        # Gemini API Config
        st.markdown("**🔑 Gemini AI Setup**")
        env_key = os.environ.get("GEMINI_API_KEY", "")
        if env_key:
            st.success("✓ API key loaded from environment")
        else:
            api_key_input = st.text_input(
                "Enter Gemini API Key",
                type="password",
                value=st.session_state.get("gemini_api_key", ""),
                help="Get a free key from Google AI Studio",
            )
            
            # Gemini Model selection
            model_opts = ["gemini-flash-latest", "gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite", "gemini-2.0-flash"]
            current_mod = st.session_state.get("gemini_model", "gemini-flash-latest")
            if current_mod not in model_opts:
                current_mod = "gemini-flash-latest"
                
            selected_model = st.selectbox(
                "Select Model",
                options=model_opts,
                index=model_opts.index(current_mod),
                help="Switch models if you hit rate limits."
            )
            
            if api_key_input:
                st.session_state["gemini_api_key"] = api_key_input
                
            if selected_model != st.session_state.get("gemini_model"):
                st.session_state["gemini_model"] = selected_model
                # Clear session state analysis so it regenerates on key/model change
                if "analysis_cache" in st.session_state:
                    del st.session_state["analysis_cache"]

        st.divider()
        st.markdown(
            '<div style="color:#8B949E; font-size:0.70rem; text-align:center;">'
            "Data · Yahoo Finance & SQLite<br>"
            "Models · scikit-learn<br>"
            f"AI · {st.session_state.get('gemini_model', 'gemini-flash-latest')}<br>"
            "Charts · Plotly<br><br>"
            f'<em>{datetime.datetime.now().strftime("%d %b %Y · %H:%M")}</em>'
            "</div>",
            unsafe_allow_html=True,
        )


@st.cache_data(ttl=3600, show_spinner=False)
def search_ticker_keyword(query: str) -> list[dict]:
    """Search for a stock ticker by company name or keyword using Yahoo Finance."""
    if not query:
        return []
    try:
        res = requests.get(
            f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=6",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        data = res.json()
        return [q for q in data.get("quotes", []) if q.get("quoteType") in ["EQUITY", "CRYPTOCURRENCY", "ETF"]]
    except Exception:
        return []

def render_ticker_input(key_prefix: str = "") -> str:
    """
    Renders a unified search input that looks up tickers by company name.

    Parameters
    ----------
    key_prefix : str
        Prefix for the Streamlit widget key.

    Returns
    -------
    str
        Validated uppercase ticker symbol, or empty string if invalid/empty.
    """
    query = st.text_input(
        "🔍 Search Company or Ticker (e.g. Apple, NVDA, TCS)",
        value=st.session_state.get(f"{key_prefix}_last_query", "Apple"),
        key=f"{key_prefix}_ticker_search",
    ).strip()
    
    st.session_state[f"{key_prefix}_last_query"] = query

    if not query:
        return ""

    with st.spinner("Searching tickers..."):
        results = search_ticker_keyword(query)

    if not results:
        # Fallback to direct ticker if search fails
        st.warning(f"No results found for '{query}'. Using as exact ticker.")
        return query.upper()

    options = []
    symbol_map = {}
    
    for r in results:
        sym = r.get("symbol")
        name = r.get("shortname", sym)
        exch = r.get("exchDisp", "")
        label = f"{sym} — {name} ({exch})" if exch else f"{sym} — {name}"
        options.append(label)
        symbol_map[label] = sym

    selected_label = st.selectbox(
        "Select matching asset:",
        options=options,
        key=f"{key_prefix}_ticker_select"
    )
    
    return symbol_map.get(selected_label, query.upper())


# ─── Cached data helpers ──────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def cached_download(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download + preprocess with 5-minute TTL cache."""
    raw = download_stock_data(ticker, start=start, end=end, save=False)
    return preprocess_pipeline(raw, ticker=ticker, save=False)


@st.cache_data(ttl=300, show_spinner=False)
def cached_features(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Return fully featured DataFrame (cached)."""
    clean = cached_download(ticker, start=start, end=end)
    return add_features(clean)


@st.cache_resource(show_spinner=False)
def cached_metrics() -> dict:
    """Load metrics.json once per session."""
    try:
        return load_metrics()
    except FileNotFoundError:
        return {}


@st.cache_resource(show_spinner=False)
def cached_model(model_name: str):
    """Load a .pkl model once per session (resource cache = persistent)."""
    return load_model(MODEL_REGISTRY[model_name])
