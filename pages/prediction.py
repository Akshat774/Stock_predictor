"""
pages/prediction.py
-------------------
Prediction page — select any stock ticker and a trained ML model, then
predict the next trading day's closing price.

Integrates Gemini API for market analysis explanations.
All inference is performed by loading the pre-saved .pkl files.
"""

import datetime
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pages._shared import (
    inject_css,
    render_sidebar_footer,
    render_ticker_input,
    cached_download,
    DEFAULT_START,
    PALETTE,
    get_gemini_api_key,
)
from src.predictor import predict_next_close, get_available_models, models_exist
from src.llm_analyzer import generate_analysis
from utils.plotting import plot_historical_close, plot_prediction_gauge

# ── Apply shared theme & sidebar ──────────────────────────────────────────────
inject_css()
render_sidebar_footer()

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner" style="padding:26px 40px;">
    <div class="hero-title" style="font-size:2rem;">🔮 Next-Day Prediction</div>
    <p class="hero-subtitle">
        Enter any Yahoo Finance ticker symbol and select a trained model to forecast tomorrow's closing price.
        Features persistent caching via SQLite to prevent API rate limits.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GUARD — models must exist
# ─────────────────────────────────────────────────────────────────────────────
if not models_exist():
    st.error(
        "**⚠️ No trained models found.**\n\n"
        "Please run the training script from your terminal:\n"
        "```\npython train_models.py\n```"
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# INPUT CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
col_ticker, col_model = st.columns([1.6, 1])

with col_ticker:
    ticker = render_ticker_input("pred")

with col_model:
    model_choice = st.selectbox(
        "Prediction Model",
        options=get_available_models(),
        key="pred_model",
        help="Linear Regression is the fast baseline; Random Forest captures non-linear patterns.",
    )

if not ticker:
    st.stop()

# Model description card
_descriptions = {
    "Linear Regression": (
        "📐 **Linear Regression** — Fast baseline model with StandardScaler preprocessing. "
        "High interpretability. Assumes a linear relationship between features and price."
    ),
    "Random Forest": (
        "🌲 **Random Forest** — Ensemble of 200 decision trees trained in parallel. "
        "Captures non-linear patterns and is generally more accurate for stock data."
    ),
}
st.markdown(
    f'<div class="info-card" style="margin-bottom:22px;">'
    f'{_descriptions.get(model_choice, "")}'
    f'</div>',
    unsafe_allow_html=True,
)

predict_btn = st.button("🔮  Predict Next Close", use_container_width=False)

# Resolve prediction results persistence
if "pred_result" not in st.session_state:
    st.session_state["pred_result"] = None

# If user clicks button, run prediction and update session state
if predict_btn:
    with st.spinner(f"Downloading latest {ticker} data and running {model_choice}..."):
        try:
            result = predict_next_close(
                ticker=ticker,
                model_name=model_choice,
                start=DEFAULT_START,
                end=datetime.date.today().isoformat(),
            )
            st.session_state["pred_result"] = result
            # Clear previous analysis since input changed
            if "analysis_cache" in st.session_state:
                st.session_state["analysis_cache"] = None
        except Exception as exc:
            st.error(f"Prediction failed: **{exc}**")
            st.session_state["pred_result"] = None

# ─────────────────────────────────────────────────────────────────────────────
# RENDER PREDICTION RESULTS
# ─────────────────────────────────────────────────────────────────────────────
result = st.session_state["pred_result"]

if result and result["ticker"] == ticker and result["model_name"] == model_choice:
    pct        = result["pct_change"]
    is_up      = pct >= 0
    arrow      = "▲" if is_up else "▼"
    color_cls  = "green" if is_up else "red"
    signal_txt = "Bullish 📈" if is_up else "Bearish 📉"

    # ── Live Real-Time Quote Stream ──────────────────────────────────────────
    @st.fragment(run_every=10)
    def render_prediction_live_quote(t: str):
        from src.downloader import get_realtime_quote
        import datetime
        quote = get_realtime_quote(t)
        if quote and quote.get("last_price") is not None:
            last_price = quote["last_price"]
            prev_close = quote["previous_close"]
            currency = quote.get("currency", "USD")
            
            delta = 0.0
            delta_pct = 0.0
            if prev_close:
                delta = last_price - prev_close
                delta_pct = (delta / prev_close) * 100
                
            color = "#43E97B" if delta_pct >= 0 else "#FF6584"
            st.markdown(
                f"""
                <div class="info-card" style="margin-bottom: 20px; border-left: 4px solid {color};">
                    <div style="font-size:0.78rem; color:#8B949E; text-transform:uppercase; font-weight:600; letter-spacing:0.5px;">🔴 Real-Time Price</div>
                    <div style="font-size:1.7rem; font-weight:700; color:#E6EDF3; margin: 4px 0;">
                        ${last_price:.2f} <span style="font-size:0.9rem; font-weight:normal; color:#8B949E;">{currency}</span>
                        <span style="font-size:1.15rem; margin-left:12px; color:{color}; font-weight:600;">
                            {delta_pct:+.2f}% ({delta:+.2f})
                        </span>
                    </div>
                    <div style="font-size:0.7rem; color:#8B949E;">
                        Last updated: {datetime.datetime.now().strftime('%H:%M:%S')} (Auto-refreshes every 10s)
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    render_prediction_live_quote(ticker)

    # ── Gauge chart ──────────────────────────────────────────────────────────
    st.plotly_chart(
        plot_prediction_gauge(
            current=result["current_close"],
            predicted=result["predicted_close"],
            pct_change=pct,
            ticker=ticker,
        ),
        use_container_width=True,
    )

    # ── Three metric tiles ───────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Current Close", f"${result['current_close']:.2f}",
                  help="Most recent available closing price")
    with m2:
        st.metric(
            f"Predicted Close · {result['prediction_date']}",
            f"${result['predicted_close']:.2f}",
            delta=f"{pct:+.2f}%",
        )
    with m3:
        st.metric("Signal", signal_txt)

    # ── Rich prediction box ───────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="predict-box">
            <div class="label">{model_choice} &nbsp;·&nbsp; {ticker}</div>
            <div class="value {color_cls}">${result['predicted_close']:.4f}</div>
            <div style="font-size:1.05rem;color:#A8B2D8;margin-top:8px;">
                {arrow} {abs(pct):.2f}% vs current close of ${result['current_close']:.4f}
            </div>
            <div style="font-size:0.80rem;color:#8B949E;margin-top:8px;">
                Estimated for: <strong>{result['prediction_date']}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Gemini AI Analysis Section ───────────────────────────────────────────
    st.markdown('<div class="section-header"><h2>🤖 Gemini AI Market Insight</h2></div>',
                unsafe_allow_html=True)
    
    api_key = get_gemini_api_key()
    
    if not api_key:
        st.info(
            "💡 **Want an AI-powered breakdown of this prediction?**\n\n"
            "Provide your Gemini API Key in the left sidebar to generate detailed "
            "market insights, trend analyses, and key risk factors."
        )
    else:
        # Load or generate analysis
        if st.session_state.get("analysis_cache") is None:
            with st.spinner("🤖 Gemini AI is analyzing indicators and generating market insight..."):
                analysis_text = generate_analysis(
                    ticker=ticker,
                    model_name=model_choice,
                    prediction_result=result,
                    latest_features=result["latest_features"],
                    recent_prices=result["recent_prices"],
                    api_key=api_key
                )
                st.session_state["analysis_cache"] = analysis_text
        
        st.markdown(
            f'<div class="info-card" style="line-height: 1.6;">'
            f'{st.session_state["analysis_cache"]}'
            f'</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # ── Recent price chart with prediction annotation ─────────────────────────
    st.markdown('<div class="section-header"><h2>📈 Recent Price Context</h2></div>',
                unsafe_allow_html=True)

    try:
        ctx_start = (datetime.date.today() - datetime.timedelta(days=90)).isoformat()
        df_ctx = cached_download(ticker, ctx_start, datetime.date.today().isoformat())

        fig_ctx = plot_historical_close(df_ctx, ticker)

        # Overlay predicted point
        last_date   = df_ctx.index[-1]
        next_date   = last_date + pd.offsets.BDay(1)
        dot_color   = PALETTE["accent"] if is_up else PALETTE["secondary"]

        fig_ctx.add_trace(go.Scatter(
            x=[last_date, next_date],
            y=[result["current_close"], result["predicted_close"]],
            name="Prediction",
            mode="lines+markers",
            line=dict(color=dot_color, width=2.5, dash="dot"),
            marker=dict(size=10, symbol="diamond"),
            hovertemplate="Predicted: $%{y:.2f}<extra></extra>",
        ))
        st.plotly_chart(fig_ctx, use_container_width=True)
    except Exception:
        pass  # Context chart is supplementary — silently skip on error

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.info(
        "⚠️ **Disclaimer**: This prediction is generated by a machine learning model "
        "trained on historical data. It is provided for **educational purposes only** "
        "and must **NOT** be used as financial advice or for real trading decisions. "
        "Past performance does not guarantee future results."
    )
elif predict_btn:
    st.info("Please enter a valid ticker above and click 'Predict Next Close'.")
