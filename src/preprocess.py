"""
preprocess.py
-------------
Cleans raw OHLCV DataFrames: removes NaN/infinite values, enforces
correct dtypes, and saves the cleaned version to data/processed/.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def clean_dataframe(df: pd.DataFrame, ticker: str = "UNKNOWN") -> pd.DataFrame:
    """
    Clean a raw OHLCV DataFrame.

    Steps:
    1. Drop rows where any of Open, High, Low, Close, Volume is NaN.
    2. Replace infinite values with NaN, then drop those rows.
    3. Enforce non-negative Volume.
    4. Ensure Close > 0 (invalid data guard).

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame.
    ticker : str
        Ticker label used for logging only.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    original_len = len(df)
    required_cols = ["Open", "High", "Low", "Close", "Volume"]

    # Guard: ensure required columns exist
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame for {ticker}: {missing}")

    # Replace inf/-inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaN in required columns
    df = df.dropna(subset=required_cols)

    # Sanity checks: remove rows with non-positive Close or negative Volume
    df = df[df["Close"] > 0]
    df = df[df["Volume"] >= 0]

    dropped = original_len - len(df)
    if dropped > 0:
        logger.warning(
            "Dropped %d invalid rows for %s (%.1f%% of original)",
            dropped,
            ticker,
            100 * dropped / original_len,
        )
    else:
        logger.info("No invalid rows found for %s. Data is clean.", ticker)

    return df.sort_index()


def save_processed(df: pd.DataFrame, ticker: str) -> Path:
    """
    Save a cleaned DataFrame to data/processed/<ticker>.csv.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV DataFrame.
    ticker : str
        Ticker symbol used for the filename.

    Returns
    -------
    Path
        Path to the saved file.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = PROCESSED_DATA_DIR / f"{ticker.replace('.', '_')}_processed.csv"
    df.to_csv(filepath)
    logger.info("Processed data saved to %s", filepath)
    return filepath


def load_processed(ticker: str) -> pd.DataFrame:
    """
    Load previously saved processed CSV for a ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.

    Raises
    ------
    FileNotFoundError
        If no processed file exists for the ticker.
    """
    filepath = PROCESSED_DATA_DIR / f"{ticker.replace('.', '_')}_processed.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data not found for {ticker}: {filepath}"
        )
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    logger.info("Loaded processed data for %s: %d rows", ticker, len(df))
    return df


def preprocess_pipeline(df: pd.DataFrame, ticker: str = "UNKNOWN", save: bool = True) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame.
    ticker : str
        Ticker symbol.
    save : bool
        If True, saves the cleaned DataFrame to data/processed/.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering.
    """
    df_clean = clean_dataframe(df, ticker=ticker)

    if save:
        save_processed(df_clean, ticker=ticker)

    return df_clean
