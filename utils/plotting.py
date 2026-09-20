"""
utils/plotting.py
-----------------
Reusable, parameterized Plotly figure factory functions.
All functions return plotly.graph_objects.Figure objects so they
can be embedded directly in Streamlit with st.plotly_chart().
"""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Shared colour palette ────────────────────────────────────────────────────
PALETTE = {
    "primary": "#6C63FF",       # Purple
    "secondary": "#FF6584",     # Pink
    "accent": "#43E97B",        # Green
    "neutral": "#A8B2D8",       # Muted blue-grey
    "bg_dark": "#0D1117",       # Dark background
    "bg_card": "#161B22",       # Card background
    "text": "#E6EDF3",          # Light text
    "grid": "#21262D",          # Grid lines
}

_LAYOUT_DEFAULTS = dict(
    plot_bgcolor=PALETTE["bg_dark"],
    paper_bgcolor=PALETTE["bg_dark"],
    font=dict(color=PALETTE["text"], family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=PALETTE["grid"]),
    xaxis=dict(
        gridcolor="rgba(33,38,45,0.5)",
        showgrid=True,
        zeroline=False,
        color=PALETTE["text"],
        showline=False,
    ),
    yaxis=dict(
        gridcolor="rgba(33,38,45,0.5)",
        showgrid=True,
        zeroline=False,
        color=PALETTE["text"],
        showline=False,
    ),
    hovermode="x unified",
)


def _apply_dark_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply the shared dark-theme layout to a figure."""
    fig.update_layout(title=dict(text=title, font=dict(size=18)), **_LAYOUT_DEFAULTS)
    return fig


# ─── Historical Price ─────────────────────────────────────────────────────────

def plot_historical_close(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Plot historical closing price as a filled area chart.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.
    ticker : str
        Ticker symbol for the title.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        name="Close",
        line=dict(color=PALETTE["primary"], width=2),
        fill="tozeroy",
        fillcolor=f"rgba(108,99,255,0.10)",
        hovertemplate="%{x|%Y-%m-%d}<br>Close: $%{y:.2f}<extra></extra>",
    ))
    return _apply_dark_layout(fig, f"{ticker} — Historical Closing Price")


# ─── Volume Chart ─────────────────────────────────────────────────────────────

def plot_volume(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Plot trading volume as a bar chart coloured by direction (up/down).

    Parameters
    ----------
    df : pd.DataFrame
    ticker : str

    Returns
    -------
    go.Figure
    """
    direction = df["Close"].diff().fillna(0)
    colors = [PALETTE["accent"] if d >= 0 else PALETTE["secondary"] for d in direction]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"],
        name="Volume",
        marker_color=colors,
        hovertemplate="%{x|%Y-%m-%d}<br>Volume: %{y:,.0f}<extra></extra>",
    ))
    return _apply_dark_layout(fig, f"{ticker} — Trading Volume")


# ─── Moving Averages ──────────────────────────────────────────────────────────

def plot_moving_averages(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Plot Close price with MA5 and MA10 overlaid.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Close, MA5, MA10 columns (or computed on the fly).
    ticker : str

    Returns
    -------
    go.Figure
    """
    plot_df = df.copy()
    if "MA5" not in plot_df.columns:
        plot_df["MA5"] = plot_df["Close"].rolling(5).mean()
    if "MA10" not in plot_df.columns:
        plot_df["MA10"] = plot_df["Close"].rolling(10).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["Close"],
        name="Close",
        line=dict(color=PALETTE["neutral"], width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Close: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["MA5"],
        name="MA5",
        line=dict(color=PALETTE["accent"], width=1.8, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>MA5: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["MA10"],
        name="MA10",
        line=dict(color=PALETTE["secondary"], width=1.8, dash="dash"),
        hovertemplate="%{x|%Y-%m-%d}<br>MA10: $%{y:.2f}<extra></extra>",
    ))
    return _apply_dark_layout(fig, f"{ticker} — Close with MA5 & MA10")


# ─── Advanced Combined Chart ──────────────────────────────────────────────────

def plot_advanced_chart(df: pd.DataFrame, ticker: str, last_n: int = 150) -> go.Figure:
    """
    Plot a professional combined chart:
      - Candlesticks for price
      - MA5 and MA10 overlays
      - Volume on a secondary Y-axis at the bottom

    Parameters
    ----------
    df : pd.DataFrame
    ticker : str
    last_n : int
        Number of recent rows to display to avoid clutter.
    """
    subset = df.tail(last_n).copy()
    if "MA5" not in subset.columns:
        subset["MA5"] = subset["Close"].rolling(5).mean()
    if "MA10" not in subset.columns:
        subset["MA10"] = subset["Close"].rolling(10).mean()

    # Create subplots with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 1. Volume Bar Chart (Secondary Y, drawn first so it's behind price)
    direction = subset["Close"].diff().fillna(0)
    vol_colors = [PALETTE["accent"] if d >= 0 else PALETTE["secondary"] for d in direction]
    
    fig.add_trace(
        go.Bar(
            x=subset.index,
            y=subset["Volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.6,
            hovertemplate="Volume: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=True,
    )

    # 2. Candlestick
    fig.add_trace(
        go.Candlestick(
            x=subset.index,
            open=subset["Open"],
            high=subset["High"],
            low=subset["Low"],
            close=subset["Close"],
            name="Price",
            increasing_line_color=PALETTE["accent"],
            decreasing_line_color=PALETTE["secondary"],
        ),
        secondary_y=False,
    )

    # 3. Moving Averages
    fig.add_trace(
        go.Scatter(
            x=subset.index, y=subset["MA5"],
            name="MA5",
            line=dict(color="#FFB86C", width=1.5, dash="dot"),
            hoverinfo="skip"
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=subset.index, y=subset["MA10"],
            name="MA10",
            line=dict(color="#8BE9FD", width=1.5, dash="dash"),
            hoverinfo="skip"
        ),
        secondary_y=False,
    )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    # Adjust axes
    # Volume scale should be small (bottom 20%)
    fig.update_yaxes(title_text="Price ($)", secondary_y=False)
    fig.update_yaxes(showgrid=False, secondary_y=True, range=[0, subset["Volume"].max() * 5], showticklabels=False)

    return _apply_dark_layout(fig, f"{ticker} — Interactive Advanced Chart")


# ─── Actual vs Predicted ──────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    y_test: pd.Series | np.ndarray,
    y_pred_lr: np.ndarray,
    y_pred_rf: np.ndarray,
    dates: pd.DatetimeIndex | None = None,
    title: str = "Actual vs Predicted — Test Set",
    show_markers: bool | None = None,
) -> go.Figure:
    """
    Plot actual vs. predicted closing prices for both models.

    Parameters
    ----------
    y_test : array-like
        Actual next-day close values.
    y_pred_lr : np.ndarray
        Linear Regression predictions.
    y_pred_rf : np.ndarray
        Random Forest predictions.
    dates : pd.DatetimeIndex | None
        Optional dates for the x-axis.
    title : str
        Chart title.
    show_markers : bool | None
        Explicitly force or disable scatter markers on lines.

    Returns
    -------
    go.Figure
    """
    x = dates if dates is not None else np.arange(len(y_test))
    use_markers = show_markers if show_markers is not None else (len(y_test) <= 80)
    trace_mode = "lines+markers" if use_markers else "lines"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y_test,
        name="Actual",
        mode=trace_mode,
        marker=dict(size=6, color="#FFFFFF"),
        line=dict(color="#FFFFFF", width=2.5),
        hovertemplate="<b>Actual</b>: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_pred_lr,
        name="Linear Regression",
        mode=trace_mode,
        marker=dict(size=6, color="#FF9F43"),
        line=dict(color="#FF9F43", width=2, dash="dot"),
        hovertemplate="<b>LR Pred</b>: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_pred_rf,
        name="Random Forest",
        mode=trace_mode,
        marker=dict(size=6, color=PALETTE["accent"]),
        line=dict(color=PALETTE["accent"], width=2, dash="dash"),
        hovertemplate="<b>RF Pred</b>: $%{y:.2f}<extra></extra>",
    ))

    # Add quick range selector buttons on x-axis
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="ALL"),
            ]),
            bgcolor="#161B22",
            activecolor="#6C63FF",
            font=dict(color="#E6EDF3", size=11),
        )
    )
    return _apply_dark_layout(fig, title)


# ─── Metrics Radar / Bar ──────────────────────────────────────────────────────

def plot_metrics_bar(metrics: dict[str, dict[str, float]]) -> go.Figure:
    """
    Grouped bar chart comparing MAE and RMSE for all models.

    Parameters
    ----------
    metrics : dict[str, dict[str, float]]
        {model_name: {"MAE": ..., "RMSE": ..., "R2": ...}}

    Returns
    -------
    go.Figure
    """
    model_names = list(metrics.keys())
    mae_vals = [metrics[m]["MAE"] for m in model_names]
    rmse_vals = [metrics[m]["RMSE"] for m in model_names]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="MAE",
        x=model_names,
        y=mae_vals,
        marker_color=PALETTE["primary"],
        hovertemplate="%{x}<br>MAE: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="RMSE",
        x=model_names,
        y=rmse_vals,
        marker_color=PALETTE["secondary"],
        hovertemplate="%{x}<br>RMSE: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(barmode="group")
    return _apply_dark_layout(fig, "MAE & RMSE Comparison")


def plot_r2_bar(metrics: dict[str, dict[str, float]]) -> go.Figure:
    """
    Bar chart of R² scores for each model.

    Parameters
    ----------
    metrics : dict[str, dict[str, float]]

    Returns
    -------
    go.Figure
    """
    model_names = list(metrics.keys())
    r2_vals = [metrics[m]["R2"] for m in model_names]
    colors = [PALETTE["accent"] if r >= 0 else PALETTE["secondary"] for r in r2_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="R²",
        x=model_names,
        y=r2_vals,
        marker_color=colors,
        hovertemplate="%{x}<br>R²: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color=PALETTE["neutral"], opacity=0.5)
    return _apply_dark_layout(fig, "R² Score Comparison")


def plot_prediction_gauge(
    current: float,
    predicted: float,
    pct_change: float,
    ticker: str,
) -> go.Figure:
    """
    Gauge chart showing confidence/magnitude of the predicted move.

    Parameters
    ----------
    current : float
        Current closing price.
    predicted : float
        Model's predicted next close.
    pct_change : float
        Percentage change between current and predicted.
    ticker : str

    Returns
    -------
    go.Figure
    """
    color = PALETTE["accent"] if pct_change >= 0 else PALETTE["secondary"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted,
        delta=dict(reference=current, valueformat=".2f"),
        title=dict(text=f"{ticker} — Predicted Next Close", font=dict(size=16, color=PALETTE["text"])),
        gauge=dict(
            axis=dict(
                range=[current * 0.90, current * 1.10],
                tickcolor=PALETTE["text"],
            ),
            bar=dict(color=color),
            bgcolor=PALETTE["bg_card"],
            bordercolor=PALETTE["grid"],
            steps=[
                dict(range=[current * 0.90, current], color=f"rgba(255,101,132,0.1)"),
                dict(range=[current, current * 1.10], color=f"rgba(67,233,123,0.1)"),
            ],
            threshold=dict(
                line=dict(color=PALETTE["neutral"], width=2),
                thickness=0.75,
                value=current,
            ),
        ),
        number=dict(prefix="$", valueformat=".2f", font=dict(color=PALETTE["text"])),
    ))
    fig.update_layout(
        paper_bgcolor=PALETTE["bg_dark"],
        font=dict(color=PALETTE["text"]),
        margin=dict(l=40, r=40, t=60, b=20),
    )
    return fig
