"""
feature_engineering.py
-----------------------
Adds technical indicator features to the clean OHLCV DataFrame
and constructs the prediction target (next day's Close price).

Features created:
    - Prev_Close  : Close price shifted by 1 day (Lag-1)
    - MA5         : 5-day simple moving average of Close
    - MA10        : 10-day simple moving average of Close
    - Target      : Close shifted by -1 (next day's Close)

Feature columns (input to models):
    Open, High, Low, Volume, Prev_Close, MA5, MA10
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Ordered list of feature column names used for model training/inference
FEATURE_COLUMNS: list[str] = [
    "Open",
    "High",
    "Low",
    "Volume",
    "Prev_Close",
    "MA5",
    "MA10",
]
TARGET_COLUMN: str = "Target"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features and the target variable to a cleaned OHLCV DataFrame.

    Important: Preserves chronological order. No data leakage — each feature
    is derived only from past/current information except for Target, which is
    the next day's closing price (used during training/evaluation only).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV DataFrame with columns: Open, High, Low, Close, Volume.

    Returns
    -------
    pd.DataFrame
        DataFrame with added feature and target columns, NaN rows dropped.
    """
    df = df.copy()

    # Lag-1 feature: previous day's close price
    df["Prev_Close"] = df["Close"].shift(1)

    # Moving averages of Close
    df["MA5"] = df["Close"].rolling(window=5, min_periods=5).mean()
    df["MA10"] = df["Close"].rolling(window=10, min_periods=10).mean()

    # Target: next day's closing price (label)
    # Note: the last row will have NaN target — we drop it before training
    df["Target"] = df["Close"].shift(-1)

    # Drop rows where any feature or target is NaN (first ~10 rows and last row)
    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    after = len(df)

    logger.info(
        "Feature engineering: dropped %d rows (NaN features/target). %d rows remain.",
        before - after,
        after,
    )

    return df


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and target vector y from an engineered DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame output from add_features().

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) where X has shape (n, 7) and y has shape (n,).
    """
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in DataFrame. Run add_features() first. Missing: {missing}"
        )

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def build_latest_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for the most recent available trading day for inference.

    This function computes all features based on the latest row in `df`.
    It does NOT add a Target column (since we're predicting, not evaluating).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV DataFrame.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with FEATURE_COLUMNS as columns.
    """
    df = df.copy()
    df["Prev_Close"] = df["Close"].shift(1)
    df["MA5"] = df["Close"].rolling(window=5, min_periods=5).mean()
    df["MA10"] = df["Close"].rolling(window=10, min_periods=10).mean()

    # Take the last valid row (most recent trading day)
    latest = df.dropna(subset=FEATURE_COLUMNS).iloc[[-1]]

    logger.info(
        "Latest features extracted for date: %s",
        latest.index[-1].date() if not latest.empty else "N/A",
    )

    return latest[FEATURE_COLUMNS]
