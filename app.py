"""
app.py
------
Entry point for the Stock Price Predictor Streamlit application.

Industry-standard pattern:
  - Page configuration is set ONCE here (must be the first Streamlit call).
  - st.navigation() declares all pages and handles routing.
  - No business logic or UI lives in this file.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

# ── Ensure project root is importable from all page modules ─────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Page configuration (MUST be the very first st call in the process) ───────
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/streamlit/streamlit",
        "Report a bug": None,
        "About": (
            "**Stock Price Predictor** · v1.0.0\n\n"
            "ML-powered next-day stock price forecasting using "
            "Linear Regression and Random Forest.\n\n"
            "Data via Yahoo Finance · Built with scikit-learn & Plotly."
        ),
    },
)

# ── Page registry (st.navigation — Streamlit ≥ 1.36) ─────────────────────────
pages = st.navigation(
    {
        "Main": [
            st.Page("pages/dashboard.py",        title="Dashboard",        icon="🏠", default=True),
        ],
        "Analysis": [
            st.Page("pages/stock_analysis.py",   title="Stock Analysis",   icon="📈"),
            st.Page("pages/model_performance.py", title="Model Performance", icon="📊"),
        ],
        "Prediction": [
            st.Page("pages/prediction.py",       title="Prediction",       icon="🔮"),
        ],
    }
)

# ── Run the selected page ─────────────────────────────────────────────────────
pages.run()
