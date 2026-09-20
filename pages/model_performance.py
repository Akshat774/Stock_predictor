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
<div class="hero-banner">
    <div class="hero-icon-badge">🎯</div>
    <div>
        <div class="hero-title">Model Performance</div>
        <p class="hero-subtitle">
            Evaluate and compare Linear Regression and Random Forest models on chronological test sets with custom monthly/yearly filtering.
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
    f'<div class="info-card" style="margin-bottom:20px; border-left: 4px solid var(--primary); display:flex; align-items:center; flex-wrap:wrap; gap:12px;">'
    f'<span style="font-size:0.92rem; color:#F8FAFC;">🏆 <strong>Best model by RMSE:</strong> '
    f'<span class="badge badge-purple">{best_rmse_model}</span></span>'
    f'<span style="color:#A78BFA;">|</span>'
    f'<span style="font-size:0.92rem; color:#F8FAFC;">🏆 <strong>Best model by R²:</strong> '
    f'<span class="badge badge-green">{best_r2_model}</span></span>'
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

        # Combine into results DataFrame
        df_results = pd.DataFrame(
            {
                "Actual": y_test.values,
                "Linear Regression": y_pred_lr,
                "Random Forest": y_pred_rf,
            },
            index=pd.to_datetime(y_test.index),
        )

        # ── Timeframe & Inspection Controls ──────────────────────────────────
        st.markdown("##### ⏱️ Prediction Inspection & Time Filter")
        c_mode, c_view = st.columns([1.5, 2.5])
        with c_mode:
            filter_mode = st.radio(
                "Filter Mode",
                [
                    "📅 All Test Data (Daily)",
                    "🗓️ Specific Year / Month",
                    "📊 Time Aggregation (Daily / Monthly / Yearly)",
                    "🔎 Single Day Inspector",
                ],
                key="perf_filter_mode",
            )

        df_display = df_results.copy()
        chart_title = f"{avp_ticker} — Actual vs Predicted (Test Set)"

        with c_view:
            if filter_mode == "🗓️ Specific Year / Month":
                avail_years = sorted(list(df_results.index.year.unique()))
                y_col, m_col = st.columns(2)
                with y_col:
                    sel_year = st.selectbox("Select Year", ["All Years"] + avail_years, index=0)
                with m_col:
                    month_map = {
                        "All Months": None,
                        "01 - January": 1, "02 - February": 2, "03 - March": 3,
                        "04 - April": 4, "05 - May": 5, "06 - June": 6,
                        "07 - July": 7, "08 - August": 8, "09 - September": 9,
                        "10 - October": 10, "11 - November": 11, "12 - December": 12,
                    }
                    sel_month_name = st.selectbox("Select Month", list(month_map.keys()), index=0)
                    sel_month = month_map[sel_month_name]

                # Apply year & month filter
                if sel_year != "All Years":
                    df_display = df_display[df_display.index.year == sel_year]
                if sel_month is not None:
                    df_display = df_display[df_display.index.month == sel_month]

                chart_title = f"{avp_ticker} — Actual vs Predicted ({sel_month_name.split(' - ')[-1] if sel_month else ''} {sel_year if sel_year != 'All Years' else 'All Years'})"

            elif filter_mode == "📊 Time Aggregation (Daily / Monthly / Yearly)":
                agg_type = st.selectbox(
                    "Aggregation Level",
                    ["Daily (Raw Points)", "Monthly Average", "Yearly Average"],
                    key="perf_agg_type",
                )
                if agg_type == "Monthly Average":
                    try:
                        df_display = df_results.resample("ME").mean().dropna()
                    except Exception:
                        df_display = df_results.resample("M").mean().dropna()
                    chart_title = f"{avp_ticker} — Monthly Average Actual vs Predicted"
                elif agg_type == "Yearly Average":
                    try:
                        df_display = df_results.resample("YE").mean().dropna()
                    except Exception:
                        df_display = df_results.resample("Y").mean().dropna()
                    chart_title = f"{avp_ticker} — Yearly Average Actual vs Predicted"

            elif filter_mode == "🔎 Single Day Inspector":
                dates_list = df_results.index.strftime("%Y-%m-%d").tolist()
                sel_date_str = st.selectbox("Select Trading Date", dates_list, index=len(dates_list) - 1)
                sel_date = pd.to_datetime(sel_date_str)
                day_row = df_results.loc[sel_date]

                act_val = float(day_row["Actual"])
                lr_val = float(day_row["Linear Regression"])
                rf_val = float(day_row["Random Forest"])
                lr_err = abs(lr_val - act_val)
                rf_err = abs(rf_val - act_val)
                lr_pct = (lr_err / act_val) * 100
                rf_pct = (rf_err / act_val) * 100

                st.markdown(f"**Date: `{sel_date_str}` Inspection**")
                d1, d2, d3 = st.columns(3)
                d1.metric("Actual Close", f"${act_val:.2f}")
                d2.metric("LR Prediction", f"${lr_val:.2f}", f"{lr_pct:+.2f}% err", delta_color="inverse")
                d3.metric("RF Prediction", f"${rf_val:.2f}", f"{rf_pct:+.2f}% err", delta_color="inverse")

                closer_model = "Linear Regression" if lr_err < rf_err else "Random Forest"
                st.info(f"🏆 On **{sel_date_str}**, **{closer_model}** had the closest prediction (Error: ${min(lr_err, rf_err):.2f}).")

                # Show a focused ±20 day window around this date on the chart
                sel_idx = df_results.index.get_loc(sel_date)
                start_idx = max(0, sel_idx - 15)
                end_idx = min(len(df_results), sel_idx + 16)
                df_display = df_results.iloc[start_idx:end_idx]
                chart_title = f"{avp_ticker} — Focus Window around {sel_date_str}"

        # ── Render Chart ─────────────────────────────────────────────────────
        if df_display.empty:
            st.warning("No data points match the selected date/month filter.")
        else:
            fig_avp = plot_actual_vs_predicted(
                y_test=df_display["Actual"].values,
                y_pred_lr=df_display["Linear Regression"].values,
                y_pred_rf=df_display["Random Forest"].values,
                dates=df_display.index,
                title=chart_title,
            )
            
            # If in Single Day Inspector, add a vertical line and annotation at selected date
            if filter_mode == "🔎 Single Day Inspector":
                fig_avp.add_vline(
                    x=sel_date,
                    line_width=2,
                    line_dash="dot",
                    line_color="#43E97B",
                    annotation_text=f"Selected Day ({sel_date_str})",
                    annotation_position="top left",
                    annotation_font_color="#43E97B",
                )
            
            st.plotly_chart(fig_avp, use_container_width=True)

            # Recalculate metrics on filtered subset if > 1 sample
            st.markdown(f"**Evaluation metrics for selected view ({len(df_display)} data point{'s' if len(df_display) > 1 else ''}):**")
            if len(df_display) >= 2:
                lr_m = compute_all_metrics(df_display["Actual"].values, df_display["Linear Regression"].values)
                rf_m = compute_all_metrics(df_display["Actual"].values, df_display["Random Forest"].values)

                ticker_rows = [
                    {"Model": "Linear Regression", **{k: f"{v:.4f}" for k, v in lr_m.items()}},
                    {"Model": "Random Forest",     **{k: f"{v:.4f}" for k, v in rf_m.items()}},
                ]
                st.dataframe(pd.DataFrame(ticker_rows), use_container_width=True, hide_index=True)

                ticker_full_metrics = {
                    "Linear Regression": lr_m,
                    "Random Forest":     rf_m,
                }
                best_here = best_model_by_metric(ticker_full_metrics, "RMSE")
                st.markdown(
                    f'<div style="margin-top:10px; color:#CBD5E1; font-size:0.88rem; font-weight:600;">'
                    f'🏆 Best model for this period by RMSE: '
                    f'<span class="badge badge-purple">{best_here}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Need at least 2 data points to compute statistical metrics.")

            # ── Detailed Data Table Expander ─────────────────────────────────
            with st.expander("📋 View Daily / Period Breakdown Table", expanded=False):
                breakdown_df = df_display.copy()
                breakdown_df["LR Diff ($)"] = breakdown_df["Linear Regression"] - breakdown_df["Actual"]
                breakdown_df["RF Diff ($)"] = breakdown_df["Random Forest"] - breakdown_df["Actual"]
                breakdown_df["LR Error (%)"] = (breakdown_df["LR Diff ($)"].abs() / breakdown_df["Actual"]) * 100
                breakdown_df["RF Error (%)"] = (breakdown_df["RF Diff ($)"].abs() / breakdown_df["Actual"]) * 100
                breakdown_df["Closest Model"] = np.where(
                    breakdown_df["LR Diff ($)"].abs() < breakdown_df["RF Diff ($)"].abs(),
                    "Linear Regression",
                    "Random Forest",
                )
                formatted_df = breakdown_df.reset_index().rename(columns={"index": "Date"})
                st.dataframe(formatted_df, use_container_width=True, hide_index=True)

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
