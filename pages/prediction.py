"""
pages/prediction.py
-------------------
Prediction page — select any stock ticker and a trained ML model, then
predict the next trading day's closing price.

Integrates Gemini API for market analysis explanations.
All inference is performed by loading the pre-saved .pkl files.
"""

import datetime
import re
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

def render_ai_analysis_cards(analysis_text: str) -> None:
    """Renders structured Gemini analysis as beautiful, high-contrast visual cards with zero LaTeX math collision."""
    if not analysis_text:
        return

    # Check for error or warning states
    if (
        analysis_text.startswith("**Invalid")
        or analysis_text.startswith("**Gemini API error")
        or analysis_text.startswith("**Gemini Service")
        or analysis_text.startswith("**Gemini API rate limit")
        or analysis_text.startswith("**Gemini API key not set")
        or analysis_text.startswith("**Gemini model not available")
        or analysis_text.startswith("**No response")
    ):
        st.warning(analysis_text)
        return

    # Ensure dollar signs are escaped to prevent Streamlit KaTeX LaTeX collision
    sanitized = re.sub(r'(?<!\\)\$', r'\$', analysis_text)

    # Section metadata configuration
    SECTION_META = {
        "trend": {
            "icon": "📈",
            "title": "Price Trend Analysis",
            "color": "#38BDF8",
            "badge": "Trend Momentum",
            "badge_cls": "badge-blue",
        },
        "technical": {
            "icon": "⚡",
            "title": "Technical Signal Breakdown",
            "color": "#818CF8",
            "badge": "Technical Signals",
            "badge_cls": "badge-purple",
        },
        "rationale": {
            "icon": "🎯",
            "title": "Prediction Rationale",
            "color": "#A78BFA",
            "badge": "ML Decision Logic",
            "badge_cls": "badge-purple",
        },
        "risk": {
            "icon": "🛡️",
            "title": "Key Risk Factors",
            "color": "#FFB86C",
            "badge": "Risk Assessment",
            "badge_cls": "badge-red",
        },
        "summary": {
            "icon": "📌",
            "title": "Executive Summary",
            "color": "#43E97B",
            "badge": "Key Takeaway",
            "badge_cls": "badge-green",
        },
    }

    # Split into sections using markdown headers
    pattern = r'(?m)^#{2,4}\s*(.+)$'
    matches = list(re.finditer(pattern, sanitized))

    # Top synthesis bar
    active_model = st.session_state.get("gemini_model", "gemini-flash-latest")
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:18px; padding:10px 18px; background:rgba(108,99,255,0.10); border:1px solid rgba(108,99,255,0.30); border-radius:12px;">
            <div style="font-size:0.86rem; color:#CBD5E1; display:flex; align-items:center; gap:8px;">
                <span>🤖</span> <span>AI Engine: <strong style="color:#F8FAFC;">{active_model}</strong></span>
            </div>
            <span class="badge badge-purple">Institutional Intelligence</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not matches:
        st.markdown(
            f'<div class="ai-card" style="border-left: 5px solid var(--primary);">{sanitized}</div>',
            unsafe_allow_html=True
        )
        return

    for i, match in enumerate(matches):
        raw_title = match.group(1).strip()
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(sanitized)
        content = sanitized[start:end].strip()

        # Match section type
        lower_t = raw_title.lower()
        key = "default"
        if "trend" in lower_t or "price" in lower_t:
            key = "trend"
        elif "technical" in lower_t or "signal" in lower_t or "indicator" in lower_t:
            key = "technical"
        elif "rationale" in lower_t or "prediction" in lower_t or "reason" in lower_t:
            key = "rationale"
        elif "risk" in lower_t or "factor" in lower_t or "caution" in lower_t:
            key = "risk"
        elif "summary" in lower_t or "conclusion" in lower_t or "takeaway" in lower_t:
            key = "summary"

        meta = SECTION_META.get(key, {
            "icon": "💡",
            "title": raw_title,
            "color": "#6C63FF",
            "badge": "Insight",
            "badge_cls": "badge-blue",
        })

        clean_title = re.sub(r'^[^\w\s]+\s*', '', raw_title) or meta["title"]
        icon = meta["icon"]
        card_color = meta["color"]
        badge_label = meta["badge"]
        badge_class = meta["badge_cls"]

        st.markdown(
            f"""
            <div class="ai-card" style="border-left: 5px solid {card_color}; margin-bottom: 16px;">
                <div class="ai-card-header">
                    <div class="ai-card-title">
                        <span>{icon}</span> <span>{clean_title}</span>
                    </div>
                    <span class="badge {badge_class}">{badge_label}</span>
                </div>
                <div class="ai-card-body">
            """,
            unsafe_allow_html=True
        )
        # Render markdown content natively
        st.markdown(content)
        st.markdown("</div></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-icon-badge">🔮</div>
    <div>
        <div class="hero-title">Next-Day Prediction</div>
        <p class="hero-subtitle">
            Enter any stock ticker and select a trained ML model to forecast tomorrow's closing price with SQLite caching and Gemini AI market insights.
        </p>
    </div>
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
                <div class="info-card" style="margin-bottom: 20px; border-left: 5px solid {color}; box-shadow: 0 4px 15px rgba(0,0,0,0.25);">
                    <div style="font-size:0.82rem; color:#CBD5E1; text-transform:uppercase; font-weight:700; letter-spacing:0.8px;">🔴 Real-Time Price Stream</div>
                    <div style="font-size:1.85rem; font-weight:700; color:#F8FAFC; margin: 4px 0;">
                        ${last_price:.2f} <span style="font-size:0.95rem; font-weight:600; color:#CBD5E1;">{currency}</span>
                        <span style="font-size:1.20rem; margin-left:14px; color:{color}; font-weight:700;">
                            {delta_pct:+.2f}% ({delta:+.2f})
                        </span>
                    </div>
                    <div style="font-size:0.76rem; color:#A8B2D8;">
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

    # ── Four metric tiles ───────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
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
        conf_val = result.get("confidence_pct", 95.0)
        st.metric(
            "Confidence Score",
            f"{conf_val:.1f}%",
            help="Ensemble decision tree agreement and variance uncertainty measure",
        )
    with m4:
        st.metric("Signal", signal_txt)

    # ── Rich prediction box ───────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="predict-box">
            <div class="label">{model_choice} &nbsp;·&nbsp; {ticker}</div>
            <div class="value {color_cls}">${result['predicted_close']:.4f}</div>
            <div style="font-size:1.10rem; color:#CBD5E1; font-weight:500; margin-top:8px;">
                {arrow} {abs(pct):.2f}% vs current close of ${result['current_close']:.4f}
            </div>
            <div style="display:inline-flex; flex-wrap:wrap; justify-content:center; align-items:center; gap:14px; margin-top:16px; padding:10px 22px; background:rgba(108,99,255,0.16); border:1px solid rgba(108,99,255,0.45); border-radius:28px; font-size:0.88rem; color:#F8FAFC; box-shadow:0 2px 10px rgba(108,99,255,0.15);">
                <span>🎯 <strong>Confidence:</strong> <span style="color:#43E97B; font-weight:700;">{conf_val:.1f}%</span></span>
                <span style="color:#A78BFA; font-weight:bold;">|</span>
                <span>📏 <strong>Expected 95% Range:</strong> <strong style="color:#F8FAFC;">${result.get('lower_bound', result['predicted_close']):.2f} – ${result.get('upper_bound', result['predicted_close']):.2f}</strong></span>
            </div>
            <div style="font-size:0.84rem; color:#CBD5E1; margin-top:14px;">
                Estimated Target Date: <strong style="color:#F8FAFC;">{result['prediction_date']}</strong>
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
                    api_key=api_key,
                    gemini_model_id=st.session_state.get("gemini_model", "gemini-flash-latest")
                )
                st.session_state["analysis_cache"] = analysis_text
        
        render_ai_analysis_cards(st.session_state["analysis_cache"])

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
