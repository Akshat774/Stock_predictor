"""
predictor.py
------------
Handles loading saved models from disk and performing inference
for a given feature row. This module is the interface used by
the Streamlit app (no retraining occurs here).
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.trainer import LR_MODEL_PATH, RF_MODEL_PATH, load_model
from src.feature_engineering import FEATURE_COLUMNS, build_latest_features
from src.downloader import download_stock_data
from src.preprocess import preprocess_pipeline

logger = logging.getLogger(__name__)

# ─── Model registry ──────────────────────────────────────────────────────────
MODEL_REGISTRY: dict[str, Path] = {
    "Linear Regression": LR_MODEL_PATH,
    "Random Forest": RF_MODEL_PATH,
}


def load_predictor_model(model_name: str) -> Any:
    """
    Load a saved sklearn Pipeline by name.

    Parameters
    ----------
    model_name : str
        One of 'Linear Regression' or 'Random Forest'.

    Returns
    -------
    fitted sklearn Pipeline

    Raises
    ------
    KeyError
        If model_name is not in the registry.
    FileNotFoundError
        If the .pkl file does not exist.
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    path = MODEL_REGISTRY[model_name]
    return load_model(path)


def predict_next_close(
    ticker: str,
    model_name: str,
    start: str = "2018-01-01",
    end: str | None = None,
) -> dict[str, Any]:
    """
    Predict the next trading day's closing price for a given ticker.

    Workflow:
    1. Download latest data from yfinance (using SQLite-backed smart cache).
    2. Clean / preprocess.
    3. Build the latest feature row.
    4. Load the requested model.
    5. Return prediction and supporting info.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    model_name : str
        'Linear Regression' or 'Random Forest'.
    start : str
        Historical data start date for rolling feature computation.
    end : str | None
        Data end date. If None, uses today.

    Returns
    -------
    dict[str, Any]
        Keys:
        - 'ticker'           : str
        - 'model_name'       : str
        - 'prediction_date'  : str  (estimated date of next trading day)
        - 'current_close'    : float
        - 'predicted_close'  : float
        - 'pct_change'       : float (percentage difference)
        - 'latest_features'  : dict (latest features for prompt generation)
        - 'recent_prices'    : list (last 10 closing prices for trend context)
    """
    import datetime

    if end is None:
        end = datetime.date.today().strftime("%Y-%m-%d")

    # Step 1 & 2: Download + preprocess
    raw_df = download_stock_data(ticker, start=start, end=end, save=False)
    clean_df = preprocess_pipeline(raw_df, ticker=ticker, save=False)

    # Step 3: Build inference feature row
    feature_row = build_latest_features(clean_df)

    # Step 4: Load model & predict
    model = load_predictor_model(model_name)
    predicted_close = float(model.predict(feature_row)[0])

    # Current close = last available Close in the dataset
    current_close = float(clean_df["Close"].iloc[-1])
    current_date = clean_df.index[-1]
    pct_change = (predicted_close - current_close) / current_close * 100

    # Estimate the next business day (simple +1 weekday heuristic)
    next_date = current_date + pd.offsets.BDay(1)

    # Latest features as dict
    latest_features_dict = feature_row.iloc[0].to_dict()

    # Recent close prices (up to last 10 days) for trend context
    recent_prices = clean_df["Close"].tail(10).tolist()

    result = {
        "ticker": ticker,
        "model_name": model_name,
        "prediction_date": str(next_date.date()),
        "current_close": round(current_close, 4),
        "predicted_close": round(predicted_close, 4),
        "pct_change": round(pct_change, 4),
        "latest_features": latest_features_dict,
        "recent_prices": [round(p, 4) for p in recent_prices],
    }

    logger.info(
        "[%s | %s] Current=%.4f  Predicted=%.4f  Change=%.2f%%",
        ticker,
        model_name,
        current_close,
        predicted_close,
        pct_change,
    )

    return result



def get_available_models() -> list[str]:
    """
    Return names of all registered models.

    Returns
    -------
    list[str]
    """
    return list(MODEL_REGISTRY.keys())


def models_exist() -> bool:
    """
    Check whether all model .pkl files have been generated.

    Returns
    -------
    bool
        True if all models are present on disk.
    """
    return all(path.exists() for path in MODEL_REGISTRY.values())
