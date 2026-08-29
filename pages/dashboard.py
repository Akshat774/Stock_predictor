"""
pages/dashboard.py
------------------
Dashboard page — project overview, live market snapshot, ML workflow,
and feature engineering reference table.

This module contains ONLY UI logic for the Dashboard page.
All shared utilities (CSS, caching, constants) come from _shared.py.
"""

import datetime

import pandas as pd
import streamlit as st

from pages._shared import (
    inject_css,
    render_sidebar_footer,
    render_ticker_input,
    cached_download,
    SUPPORTED_TICKERS,
    DEFAULT_TICKER,
)
from src.predictor import MODEL_REGISTRY, models_exist

# ── Apply global theme ────────────────────────────────────────────────────────
inject_css()

# ── Sidebar footer ────────────────────────────────────────────────────────────
render_sidebar_footer()

# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-icon-badge">📈</div>
    <div>
        <div class="hero-title">Stock Price Predictor</div>
        <p class="hero-subtitle">
            A professional ML platform that forecasts the
            <strong style="color:#6C63FF;">next trading day's closing price</strong> using historical
            market data, Linear Regression, and Random Forest models.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# QUICK-STAT TILES
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
all_ready = models_exist()

with c1:
    st.metric("ML Models Trained", "2", help="Linear Regression + Random Forest")
with c2:
    st.metric("Input Features", "7", help="Open, High, Low, Volume, Prev_Close, MA5, MA10")
with c3:
    st.metric("Available Stocks", "All Global Stocks", help="Any stock worldwide supported via Yahoo Finance (US, NSE, BSE, etc.)")
with c4:
    st.metric("Data Source", "Yahoo Finance", help="Free via yfinance · No API key needed")
with c5:
    st.metric("Model Status", "🟢 Ready" if all_ready else "🔴 Not Trained")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# LIVE MARKET SNAPSHOT  +  MODEL CARDS
# ─────────────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([2.2, 1])

with left_col:
    st.markdown('<div class="section-header"><h2>📡 Live Market Snapshot</h2></div>',
                unsafe_allow_html=True)

    snap_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    snap_cols = st.columns(len(snap_tickers))
    today = datetime.date.today().isoformat()
    week_ago = (datetime.date.today() - datetime.timedelta(days=10)).isoformat()

    for col, ticker in zip(snap_cols, snap_tickers):
        with col:
            try:
                with st.spinner(f"{ticker}…"):
                    df_s = cached_download(ticker, week_ago, today)
                if len(df_s) >= 2:
                    latest = float(df_s["Close"].iloc[-1])
                    prev   = float(df_s["Close"].iloc[-2])
                    delta_pct = (latest - prev) / prev * 100
                    st.metric(ticker, f"${latest:.2f}", f"{delta_pct:+.2f}%")
                else:
                    st.metric(ticker, "N/A")
            except Exception:
                st.metric(ticker, "N/A")

    st.markdown("##### 🔍 Quick Live Search (Company Name or Ticker)")
    q_ticker = render_ticker_input("dash")
    
    @st.fragment(run_every=10)
    def render_live_quote_block(ticker: str):
        if not ticker:
            return
        from src.downloader import get_realtime_quote, validate_ticker
        import datetime
        
        with st.spinner(f"Updating quote for {ticker}..."):
            if validate_ticker(ticker):
                quote = get_realtime_quote(ticker)
                if quote and quote.get("last_price") is not None:
                    last_price = quote["last_price"]
                    prev_close = quote["previous_close"]
                    currency = quote.get("currency", "USD")
                    
                    delta = 0.0
                    delta_pct = 0.0
                    if prev_close:
                        delta = last_price - prev_close
                        delta_pct = (delta / prev_close) * 100
                    
                    qc1, qc2, qc3 = st.columns(3)
                    with qc1:
                        st.metric(f"{ticker} Last Price", f"{last_price:.2f} {currency}", f"{delta_pct:+.2f}%")
                    with qc2:
                        st.metric("Day High / Low", f"{quote.get('day_high', 0.0):.2f} / {quote.get('day_low', 0.0):.2f}")
                    with qc3:
                        st.metric("Avg Volume (3M)", f"{quote.get('volume', 0.0):,.0f}" if quote.get('volume') else "N/A")
                    
                    st.markdown(
                        f"<div style='text-align:right; font-size:0.72rem; color:#8B949E; margin-top:6px;'>"
                        f"Live Quote Stream · Updated at {datetime.datetime.now().strftime('%H:%M:%S')} (Auto-refreshes every 10s)</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.warning(f"Could not load quote details for '{ticker}'. It might be closed or untraded today.")
            else:
                st.error(f"Ticker '{ticker}' is not valid on Yahoo Finance.")

    if q_ticker:
        render_live_quote_block(q_ticker)



with right_col:
    st.markdown('<div class="section-header"><h2>🗂️ Available Models</h2></div>',
                unsafe_allow_html=True)

    model_icons = {
        "Linear Regression": ("📐", "Baseline · Fast · Interpretable"),
        "Random Forest":     ("🌲", "Ensemble · Non-linear · Accurate"),
    }
    for name, path in MODEL_REGISTRY.items():
        ok = path.exists()
        status_icon = "✅" if ok else "❌"
        icon, desc = model_icons.get(name, ("🤖", ""))
        st.markdown(
            f'<div class="info-card" style="margin-bottom:10px;">'
            f'{icon} <strong>{name}</strong> {status_icon}'
            f'<br><span style="color:#8B949E;font-size:0.76rem;">{desc}</span>'
            f'<br><span style="color:#8B949E;font-size:0.72rem;">'
            f'{"Ready for inference" if ok else "Run train_models.py first"}'
            f'</span></div>',
            unsafe_allow_html=True,
        )

    if not all_ready:
        st.warning("Run `python train_models.py` to train and save models.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ML WORKFLOW DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"><h2>🔄 End-to-End ML Workflow</h2></div>',
            unsafe_allow_html=True)

steps = [
    ("📥", "Download",  "yfinance"),
    ("🧹", "Clean",     "Remove NaN"),
    ("⚙️",  "Features", "MA5 · MA10 · Lag"),
    ("🏋️", "Train",     "LR + RF"),
    ("📊", "Evaluate",  "MAE · RMSE · R²"),
    ("💾", "Save",      ".pkl + .json"),
    ("🔮", "Predict",   "Next close"),
]

workflow_html = '<div style="display: flex; align-items: center; justify-content: space-between; overflow-x: auto; padding: 10px 0; gap: 8px;">'
for i, (icon, title, desc) in enumerate(steps):
    workflow_html += f"""
        <div style="flex: 1; text-align: center; padding: 14px 6px; background: #161B22; border: 1px solid #30363D; border-radius: 12px; min-width: 110px;">
            <div style="font-size: 1.5rem;">{icon}</div>
            <div style="font-weight: 600; font-size: 0.86rem; color: #E6EDF3; margin: 5px 0 2px;">{title}</div>
            <div style="font-size: 0.70rem; color: #8B949E;">{desc}</div>
        </div>
    """
    if i < len(steps) - 1:
        workflow_html += '<div style="color: #6C63FF; font-size: 1.2rem; font-weight: bold;">→</div>'
workflow_html += '</div>'
st.markdown(workflow_html, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"><h2>🧬 Feature Engineering Reference</h2></div>',
            unsafe_allow_html=True)

feature_df = pd.DataFrame({
    "Feature":     ["Open", "High", "Low", "Volume", "Prev_Close", "MA5", "MA10"],
    "Description": [
        "Opening price of the trading day",
        "Highest intraday price",
        "Lowest intraday price",
        "Total shares traded",
        "Prior day's closing price (Lag-1)",
        "5-day Simple Moving Average of Close",
        "10-day Simple Moving Average of Close",
    ],
    "Category": ["OHLCV", "OHLCV", "OHLCV", "OHLCV", "Lag", "Technical", "Technical"],
})

st.dataframe(feature_df, use_container_width=True, hide_index=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TECH STACK
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"><h2>🛠️ Tech Stack</h2></div>',
            unsafe_allow_html=True)

stack = [
    (
        '''<svg width="32" height="32" viewBox="0 0 128 128">
            <path fill="#3776AB" d="M63.2 0C30.4 0 32.5 13.9 32.5 13.9l.1 14.4h31.3v4.4H19.5S0 30.7 0 63.8c0 33.1 17.1 31.9 17.1 31.9h10.2v-14.3s-.6-17.1 16.8-17.1h28.8s16.2.3 16.2-15.9V16.2S91.6 0 63.2 0zm-17 9.8c2.8 0 5 2.3 5 5s-2.3 5-5 5-5-2.3-5-5 2.2-5 5-5z"/>
            <path fill="#FFD43B" d="M64.8 128c32.8 0 30.7-13.9 30.7-13.9l-.1-14.4H64.1v-4.4h44.4s19.5 1.9 19.5-31.2c0-33.1-17.1-31.9-17.1-31.9h-10.2v14.3s.6 17.1-16.8 17.1H55.1s-16.2-.3-16.2 15.9v32.2s-2.5 16.3 25.9 16.3zm17-9.8c-2.8 0-5-2.3-5-5s2.3-5 5-5 5 2.3 5 5-2.2 5-5 5z"/>
        </svg>''',
        "Python 3.12+",
        "Core Engine & Logic",
    ),
    (
        '''<svg width="32" height="32" viewBox="0 0 128 128">
            <path fill="#FF4B4B" d="M64 16l42 64H22L64 16zm46 64H18c-7 0-11 7.6-7.5 13.5l19 32.5c3.5 6 12 6 15.5 0l19-30.5 19 30.5c3.5 6 12 6 15.5 0l19-32.5c3.5-5.9-.5-13.5-7.5-13.5z"/>
        </svg>''',
        "Streamlit",
        "Reactive Dashboard UI",
    ),
    (
        '''<svg width="32" height="32" viewBox="0 0 128 128">
            <rect width="128" height="128" rx="28" fill="#6001D2"/>
            <path fill="#FFFFFF" d="M42 32l16 32v34h12V64l16-32H72L64 52 56 32H42zm46 44a8 8 0 100 16 8 8 0 000-16z"/>
        </svg>''',
        "yfinance",
        "Live Market Data API",
    ),
    (
        '''<svg width="32" height="32" viewBox="0 0 128 128">
            <rect x="14" y="24" width="26" height="80" rx="6" fill="#38BDF8"/>
            <rect x="51" y="44" width="26" height="60" rx="6" fill="#FFD43B"/>
            <rect x="88" y="16" width="26" height="88" rx="6" fill="#FF6584"/>
        </svg>''',
        "pandas / numpy",
        "Feature Engineering",
    ),
    (
        '''<svg width="32" height="32" viewBox="0 0 128 128">
            <circle cx="48" cy="64" r="36" fill="#3499CD"/>
            <circle cx="80" cy="64" r="36" fill="#F89939" opacity="0.9"/>
            <text x="64" y="70" fill="#FFFFFF" font-size="18" font-family="Inter, sans-serif" font-weight="bold" text-anchor="middle">ML</text>
        </svg>''',
        "scikit-learn",
        "LR & Random Forest",
    ),
    (
        '''<svg width="32" height="32" viewBox="0 0 128 128">
            <rect x="16" y="60" width="16" height="48" rx="4" fill="#3F4F75"/>
            <rect x="42" y="36" width="16" height="72" rx="4" fill="#38BDF8"/>
            <rect x="68" y="20" width="16" height="88" rx="4" fill="#6C63FF"/>
            <rect x="94" y="48" width="16" height="60" rx="4" fill="#43E97B"/>
        </svg>''',
        "Plotly",
        "Interactive Charts",
    ),
    (
        '''<svg width="32" height="32" viewBox="0 0 128 128">
            <ellipse cx="64" cy="30" rx="48" ry="16" fill="#0284C7"/>
            <path fill="#0EA5E9" d="M16 30v30c0 9 21.5 16 48 16s48-7 48-16V30c0 9-21.5 16-48 16S16 39 16 30z"/>
            <path fill="#38BDF8" d="M16 60v32c0 9 21.5 16 48 16s48-7 48-16V60c0 9-21.5 16-48 16S16 69 16 60z"/>
        </svg>''',
        "SQLite & joblib",
        "Cache & Persistence",
    ),
]

stack_cols = st.columns(len(stack))
for col, (svg_code, name, role) in zip(stack_cols, stack):
    with col:
        st.markdown(
            f'<div style="text-align:center;padding:16px 6px;'
            f'background:#161B22;border:1px solid #30363D;border-radius:12px;height:100%;'
            f'display:flex;flex-direction:column;align-items:center;justify-content:center;">'
            f'<div style="margin-bottom:8px;height:34px;display:flex;align-items:center;justify-content:center;">{svg_code}</div>'
            f'<div style="font-weight:600;font-size:0.84rem;color:#E6EDF3;margin:2px 0 3px;">{name}</div>'
            f'<div style="font-size:0.70rem;color:#8B949E;line-height:1.3;">{role}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
