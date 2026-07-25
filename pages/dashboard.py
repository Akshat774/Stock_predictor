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
    <div class="hero-title">📈 Stock Price Predictor</div>
    <p class="hero-subtitle">
        A professional ML platform that forecasts the
        <strong>next trading day's closing price</strong> using historical
        market data, Linear Regression, and Random Forest models.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# QUICK-STAT TILES
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
all_ready = models_exist()

with c1:
    st.metric("ML Models Trained", "2", help="Linear Regression + Random Forest")
with c2:
    st.metric("Input Features", "7", help="Open, High, Low, Volume, Prev_Close, MA5, MA10")
with c3:
    st.metric("Data Source", "Yahoo Finance", help="Free via yfinance · No API key needed")
with c4:
    st.metric("Model Status", "✅ Ready" if all_ready else "⚠️ Not Trained")

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

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 🔍 Quick Live Ticker Search")
    q_ticker = st.text_input("Enter any ticker (e.g. NVDA, TCS.NS, BTC-USD):", value="", key="dash_quick_search").strip().upper()
    
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

step_cols = st.columns(len(steps))
for col, (icon, title, desc) in zip(step_cols, steps):
    with col:
        st.markdown(
            f'<div style="text-align:center;padding:14px 6px;'
            f'background:#161B22;border:1px solid #30363D;border-radius:12px;">'
            f'<div style="font-size:1.5rem;">{icon}</div>'
            f'<div style="font-weight:600;font-size:0.86rem;color:#E6EDF3;margin:5px 0 2px;">{title}</div>'
            f'<div style="font-size:0.70rem;color:#8B949E;">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

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
    "Target": ["—", "—", "—", "—", "—", "—", "—"],
})

st.dataframe(feature_df, use_container_width=True, hide_index=True)

st.markdown("""
<div style="margin-top:10px; color:#8B949E; font-size:0.80rem;">
    <strong style="color:#6C63FF;">Target variable</strong>:
    <code>Next day's Close</code> — computed as <code>Close.shift(-1)</code>.
    The last row of any dataset is dropped because its target is unknown.
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TECH STACK
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"><h2>🛠️ Tech Stack</h2></div>',
            unsafe_allow_html=True)

stack = [
    ("🐍", "Python 3.12+", "Core language"),
    ("🌊", "Streamlit",    "Interactive dashboard"),
    ("🦁", "yfinance",     "Market data"),
    ("🐼", "pandas / numpy", "Data processing"),
    ("🤖", "scikit-learn", "ML models"),
    ("📉", "Plotly",       "Interactive charts"),
    ("💾", "joblib",       "Model persistence"),
]

stack_cols = st.columns(len(stack))
for col, (icon, name, role) in zip(stack_cols, stack):
    with col:
        st.markdown(
            f'<div style="text-align:center;padding:12px 4px;'
            f'background:#161B22;border:1px solid #30363D;border-radius:10px;">'
            f'<div style="font-size:1.3rem;">{icon}</div>'
            f'<div style="font-weight:600;font-size:0.80rem;color:#E6EDF3;margin:4px 0 2px;">{name}</div>'
            f'<div style="font-size:0.68rem;color:#8B949E;">{role}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
