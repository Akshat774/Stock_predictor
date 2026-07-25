"""
pages/model_performance.py
--------------------------
Model Performance page — compares Linear Regression and Random Forest
on the held-out test set using MAE, RMSE, and R² metrics, and renders
Actual vs Predicted charts for any supported ticker.
"""

import datetime

import numpy as np
import pandas as pd
import streamlit as st

from pages._shared import (
    inject_css,
    render_sidebar_footer,
    cached_features,
    cached_metrics,
    cached_model,
    SUPPORTED_TICKERS,
    DEFAULT_START,
)
from src.evaluator import get_predictions
from src.feature_engineering import get_feature_matrix
from src.trainer import chronological_split
from src.predictor import models_exist
from utils.metrics import compute_all_metrics, best_model_by_metric
from utils.plotting import (
    plot_actual_vs_predicted,
    plot_metrics_bar,
    plot_r2_bar,
)

# ── Apply shared theme & sidebar ──────────────────────────────────────────────
inject_css()
render_sidebar_footer()

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner" style="padding:26px 40px;">
    <div class="hero-title" style="font-size:2rem;">📊 Model Performance</div>
    <p class="hero-subtitle">
        Evaluate and compare Linear Regression and Random Forest on the
        chronological held-out test set (last 20% of training data).
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GUARD — models must exist
# ─────────────────────────────────────────────────────────────────────────────
if not models_exist():
    st.error(
        "**⚠️ No trained models found.**\n\n"
        "Please run:\n```\npython train_models.py\n```"
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD METRICS
# ─────────────────────────────────────────────────────────────────────────────
metrics_data = cached_metrics()

if not metrics_data:
    st.warning("⚠️ metrics.json is empty or missing. Please re-run `python train_models.py`.")
    st.stop()

# Separate per-model dicts from top-level metadata keys
model_metrics: dict[str, dict] = {
    k: v for k, v in metrics_data.items() if isinstance(v, dict)
}
best_rmse_model = metrics_data.get("best_model_by_rmse", "N/A")
best_r2_model   = metrics_data.get("best_model_by_r2",   "N/A")

# ─────────────────────────────────────────────────────────────────────────────
# WINNER BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="info-card" style="margin-bottom:20px;">'
    f'🏆 <strong>Best model by RMSE:</strong> '
    f'<span class="badge badge-purple">{best_rmse_model}</span>'
    f'&nbsp;&nbsp;|&nbsp;&nbsp;'
    f'🏆 <strong>Best model by R²:</strong> '
    f'<span class="badge badge-green">{best_r2_model}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# METRICS TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"><h2>📋 Combined Training-Set Metrics</h2></div>',
            unsafe_allow_html=True)

rows = []
for model_name, m in model_metrics.items():
    rows.append({
        "Model": model_name,
        "MAE  ↓": f"{m.get('MAE',  0):.4f}",
        "RMSE ↓": f"{m.get('RMSE', 0):.4f}",
        "R²   ↑": f"{m.get('R2',   0):.4f}",
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# VISUAL COMPARISON CHARTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"><h2>📊 Visual Comparison</h2></div>',
            unsafe_allow_html=True)

bar_col, r2_col = st.columns(2)
with bar_col:
    st.plotly_chart(plot_metrics_bar(model_metrics), use_container_width=True)
with r2_col:
    st.plotly_chart(plot_r2_bar(model_metrics), use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ACTUAL vs PREDICTED — per-ticker
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"><h2>🎯 Actual vs Predicted — Test Set</h2></div>',
            unsafe_allow_html=True)

from pages._shared import render_ticker_input
avp_ticker = render_ticker_input("perf")

if not avp_ticker:
    st.stop()


with st.spinner(f"Building test-set predictions for {avp_ticker}…"):
    try:
        df_feat = cached_features(
            avp_ticker,
            DEFAULT_START,
            datetime.date.today().isoformat(),
        )
        X_all, y_all = get_feature_matrix(df_feat)
        _, X_test, _, y_test = chronological_split(X_all, y_all)

        lr_model = cached_model("Linear Regression")
        rf_model = cached_model("Random Forest")

        y_pred_lr = get_predictions(lr_model, X_test)
        y_pred_rf = get_predictions(rf_model, X_test)

        fig_avp = plot_actual_vs_predicted(
            y_test=y_test.values,
            y_pred_lr=y_pred_lr,
            y_pred_rf=y_pred_rf,
            dates=y_test.index,
        )
        st.plotly_chart(fig_avp, use_container_width=True)

        # Per-ticker metrics
        st.markdown(f"**Per-ticker test-set metrics for {avp_ticker}:**")
        lr_m = compute_all_metrics(y_test, y_pred_lr)
        rf_m = compute_all_metrics(y_test, y_pred_rf)

        ticker_rows = [
            {"Model": "Linear Regression", **{k: f"{v:.4f}" for k, v in lr_m.items()}},
            {"Model": "Random Forest",     **{k: f"{v:.4f}" for k, v in rf_m.items()}},
        ]
        st.dataframe(pd.DataFrame(ticker_rows), use_container_width=True, hide_index=True)

        # Inline best model for this ticker
        ticker_full_metrics = {
            "Linear Regression": lr_m,
            "Random Forest":     rf_m,
        }
        best_here = best_model_by_metric(ticker_full_metrics, "RMSE")
        st.markdown(
            f'<div style="margin-top:8px; color:#8B949E; font-size:0.82rem;">'
            f'🏆 Best model for <strong>{avp_ticker}</strong> by RMSE: '
            f'<span class="badge badge-purple">{best_here}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    except Exception as exc:
        st.warning(f"Could not generate Actual vs Predicted chart: {exc}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# METRIC EXPLAINER
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📖 How to interpret these metrics", expanded=False):
    st.markdown("""
| Metric | Full Name | Good if… | Notes |
|--------|-----------|----------|-------|
| **MAE** | Mean Absolute Error | Lower ↓ | Average dollar error in predictions. Easy to interpret. |
| **RMSE** | Root Mean Squared Error | Lower ↓ | Like MAE but penalises large errors more harshly. |
| **R²** | R-squared | Closer to 1.0 ↑ | Proportion of price variance explained by the model. R² > 0.95 is excellent for stock-price forecasting. |

> A model with **low RMSE** and **high R²** is considered strong.
> Linear Regression may score well because stock prices have strong autocorrelation (today's price ≈ tomorrow's price),
> making even simple models surprisingly effective.
    """)
