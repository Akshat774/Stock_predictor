"""
pages/stock_analysis.py
-----------------------
Stock Analysis page — explore historical OHLCV data with interactive
Plotly charts: price history, volume, moving averages, and candlestick.

Imports shared utilities from _shared.py; no business logic is duplicated.
"""

import datetime

import pandas as pd
import streamlit as st

from pages._shared import (
    inject_css,
    render_sidebar_footer,
    cached_download,
    SUPPORTED_TICKERS,
    DEFAULT_START,
)
from utils.plotting import (
    plot_advanced_chart,
)

# ── Apply shared theme & sidebar ──────────────────────────────────────────────
inject_css()
render_sidebar_footer()

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-icon-badge">📊</div>
    <div>
        <div class="hero-title">Stock Analysis</div>
        <p class="hero-subtitle">
            Explore interactive candlestick charts, volume trends, and moving average overlays for any stock ticker.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])

with ctrl1:
    from pages._shared import render_ticker_input
    ticker = render_ticker_input("sa")

with ctrl2:
    start_date = st.date_input(
        "Start Date",
        value=datetime.date(2020, 1, 1),
        key="sa_start",
    )
with ctrl3:
    end_date = st.date_input(
        "End Date",
        value=datetime.date.today(),
        key="sa_end",
    )

if not ticker:
    st.stop()

if start_date >= end_date:
    st.error("⚠️ Start date must be earlier than end date.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(f"Loading {ticker} data…"):
    try:
        df = cached_download(ticker, str(start_date), str(end_date))
    except Exception as exc:
        st.error(f"Failed to load data for **{ticker}**: {exc}")
        st.stop()

if df.empty:
    st.warning("No data available for the selected ticker and date range.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# KEY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"><h2>📊 Key Statistics</h2></div>',
            unsafe_allow_html=True)

s1, s2, s3, s4, s5, s6 = st.columns(6)

daily_ret   = df["Close"].pct_change().iloc[-1] * 100
volatility  = df["Close"].pct_change().std() * 100
avg_vol     = df["Volume"].mean()
period_ret  = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

with s1:
    st.metric("Current Close",  f"${df['Close'].iloc[-1]:.2f}")
with s2:
    st.metric("Daily Change",   f"{daily_ret:+.2f}%")
with s3:
    st.metric("Period Return",  f"{period_ret:+.1f}%")
with s4:
    st.metric("52-W High",      f"${df['Close'].max():.2f}")
with s5:
    st.metric("52-W Low",       f"${df['Close'].min():.2f}")
with s6:
    st.metric("Avg Volume",     f"{avg_vol:,.0f}")

st.divider()

st.markdown(
    '<p style="color:#CBD5E1; font-size:0.88rem; margin-bottom:10px; font-weight:500;">'
    'Interactive advanced chart with Candlesticks, Volume, and Moving Averages.</p>',
    unsafe_allow_html=True,
)

last_n = st.slider(
    "Zoom: Number of recent trading days to display",
    min_value=30,
    max_value=min(365, len(df)),
    value=min(150, len(df)),
    key="candle_n",
)

st.plotly_chart(plot_advanced_chart(df, ticker, last_n=last_n), use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# RAW DATA VIEWER
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🗃️ View Raw OHLCV Data (most recent 200 rows)", expanded=False):
    display = df.copy()
    display.index = display.index.strftime("%Y-%m-%d")
    st.dataframe(
        display.sort_index(ascending=False).head(200),
        use_container_width=True,
    )
    st.download_button(
        label="⬇️ Download CSV",
        data=display.to_csv().encode("utf-8"),
        file_name=f"{ticker}_ohlcv.csv",
        mime="text/csv",
    )
