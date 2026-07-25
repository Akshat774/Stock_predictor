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
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.55rem !important; font-weight: 600 !important; }

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
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 45%, #0f3460 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 36px 48px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -40%; right: -15%;
    width: 460px; height: 460px;
    background: radial-gradient(circle, rgba(108,99,255,0.14) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(135deg, #fff 0%, var(--primary) 55%, var(--accent) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin: 0 0 8px 0;
}
.hero-subtitle {
    font-size: 1rem; color: var(--neutral);
    margin: 0; max-width: 580px; line-height: 1.65;
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
            label = "✓ Ready" if ok else "✗ Missing"
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
            if api_key_input:
                st.session_state["gemini_api_key"] = api_key_input
                # Clear session state analysis so it regenerates on key change
                if "analysis_cache" in st.session_state:
                    del st.session_state["analysis_cache"]

        st.divider()
        st.markdown(
            '<div style="color:#8B949E; font-size:0.70rem; text-align:center;">'
            "Data · Yahoo Finance & SQLite<br>"
            "Models · scikit-learn<br>"
            "AI · Gemini 2.0 Flash<br>"
            "Charts · Plotly<br><br>"
            f'<em>{datetime.datetime.now().strftime("%d %b %Y · %H:%M")}</em>'
            "</div>",
            unsafe_allow_html=True,
        )


def render_ticker_input(key_prefix: str = "") -> str:
    """
    Renders a unified search input for any ticker and validates it.

    Parameters
    ----------
    key_prefix : str
        Prefix for the Streamlit widget key.

    Returns
    -------
    str
        Validated uppercase ticker symbol, or empty string if invalid/empty.
    """
    col1, col2 = st.columns([3, 1])
    with col1:
        raw_ticker = st.text_input(
            "Search Ticker Symbol (e.g. AAPL, NVDA, TCS.NS, BTC-USD)",
            value="AAPL",
            key=f"{key_prefix}_ticker_search",
        ).strip().upper()
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        validate_btn = st.button("Validate Ticker", key=f"{key_prefix}_validate_btn")

    if validate_btn:
        if not raw_ticker:
            st.error("Please enter a ticker symbol.")
            return ""
        with st.spinner(f"Validating {raw_ticker} on Yahoo Finance..."):
            if validate_ticker(raw_ticker):
                st.success(f"✓ '{raw_ticker}' is valid!")
                # Add to session state of valid tickers to avoid re-validating
                st.session_state[f"valid_{raw_ticker}"] = True
            else:
                st.error(f"✗ Ticker '{raw_ticker}' not found on Yahoo Finance. Try standard symbols.")
                if f"valid_{raw_ticker}" in st.session_state:
                    del st.session_state[f"valid_{raw_ticker}"]
                return ""

    # Check if already validated in session state
    if raw_ticker and not st.session_state.get(f"valid_{raw_ticker}", False):
        # Silently validate if not yet validated
        with st.spinner(f"Validating {raw_ticker}..."):
            if validate_ticker(raw_ticker):
                st.session_state[f"valid_{raw_ticker}"] = True
            else:
                st.warning(f"Ticker '{raw_ticker}' validation pending or failed. Click 'Validate Ticker' if correct.")
                return raw_ticker

    return raw_ticker


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
