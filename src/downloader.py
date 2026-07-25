"""
src/downloader.py
-----------------
Smart, DB-backed stock data downloader.

Architecture:
  Layer 1  SQLite (permanent disk cache)
  Layer 2  yfinance (Yahoo Finance API — only called for missing date ranges)

For ANY valid Yahoo Finance ticker symbol this module will:
  1. Check the local SQLite DB for already-cached rows.
  2. Detect which date ranges are missing.
  3. Fetch ONLY those missing ranges from yfinance.
  4. Upsert new rows into SQLite.
  5. Return the complete merged DataFrame.

This design keeps API calls to an absolute minimum while ensuring users
always see up-to-date data for the requested range.

Ticker validation:
  validate_ticker() performs a lightweight yfinance probe so the UI can
  show a user-friendly error for invalid symbols before doing anything else.
"""

import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.database import (
    get_missing_date_ranges,
    load_ohlcv,
    mark_gap_attempted,
    upsert_ohlcv,
    last_fetched_at,
)

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Seconds to wait between consecutive yfinance API calls (rate-limit guard)
_API_CALL_DELAY: float = 0.5

# Columns we keep from yfinance — all others (Dividends, Stock Splits) are dropped
_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_ticker(ticker: str) -> bool:
    """
    Check whether a ticker symbol is valid on Yahoo Finance.

    Uses a minimal 5-day history fetch — fast and lightweight.

    Parameters
    ----------
    ticker : str

    Returns
    -------
    bool
        True if Yahoo Finance recognises the ticker.
    """
    try:
        info = yf.Ticker(ticker).fast_info
        # fast_info.last_price is None / 0.0 for invalid tickers
        return bool(getattr(info, "last_price", None))
    except Exception:
        return False


# ─── Core fetch (internal) ────────────────────────────────────────────────────

def _fetch_from_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Raw download from Yahoo Finance for a given date range.

    Parameters
    ----------
    ticker : str
    start  : str  'YYYY-MM-DD'
    end    : str  'YYYY-MM-DD' (inclusive — converted to exclusive for yfinance)

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with timezone-naive DatetimeIndex, or empty DF.
    """
    import datetime as dt

    start_dt = dt.datetime.strptime(start, "%Y-%m-%d")
    end_dt = dt.datetime.strptime(end, "%Y-%m-%d")
    if start_dt > end_dt:
        return pd.DataFrame()

    # yfinance treats `end` as exclusive, so add one day to include the last date.
    yf_end = (end_dt + dt.timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("yfinance fetch: %s  %s -> %s", ticker, start, end)
    time.sleep(_API_CALL_DELAY)   # polite delay

    try:
        obj = yf.Ticker(ticker)
        df  = obj.history(start=start, end=yf_end, auto_adjust=True)
    except Exception as exc:
        logger.warning("yfinance error for %s: %s", ticker, exc)
        return pd.DataFrame()

    if df.empty:
        logger.debug("yfinance returned empty data for %s (%s->%s)", ticker, start, end)
        return pd.DataFrame()

    # Keep only OHLCV columns that actually exist
    cols = [c for c in _OHLCV_COLS if c in df.columns]
    df = df[cols].copy()

    # Ensure timezone-naive DatetimeIndex
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df.sort_index(inplace=True)
    logger.info("yfinance returned %d rows for %s", len(df), ticker)
    return df


# ─── Public API ───────────────────────────────────────────────────────────────

def download_stock_data(
    ticker: str,
    start: str = "2018-01-01",
    end: str | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Smart download: SQLite-first, yfinance only for missing ranges.

    Parameters
    ----------
    ticker : str
        Any valid Yahoo Finance symbol (AAPL, TCS.NS, BTC-USD, SPY, …).
    start  : str
        Start date 'YYYY-MM-DD'.
    end    : str | None
        End date. Defaults to today.
    save   : bool
        If True, also saves a raw CSV to data/raw/<ticker>.csv (for
        compatibility with train_models.py).

    Returns
    -------
    pd.DataFrame
        Complete OHLCV DataFrame for the requested range.

    Raises
    ------
    ValueError
        If no data could be obtained for the ticker.
    """
    import datetime
    if end is None:
        end = datetime.date.today().strftime("%Y-%m-%d")

    ticker_upper = ticker.strip().upper()

    # ── Step 1: Find missing date ranges ──────────────────────────────────────
    missing_ranges = get_missing_date_ranges(ticker_upper, start, end)

    # ── Step 2: Fetch only missing ranges from yfinance ───────────────────────
    if missing_ranges:
        logger.info(
            "%s: %d gap(s) to fetch from yfinance: %s",
            ticker_upper, len(missing_ranges), missing_ranges,
        )
        for gap_start, gap_end in missing_ranges:
            new_df = _fetch_from_yfinance(ticker_upper, gap_start, gap_end)
            if not new_df.empty:
                upsert_ohlcv(ticker_upper, new_df)
            else:
                mark_gap_attempted(ticker_upper, gap_start, gap_end)
    else:
        logger.info("%s: fully cached, skipping yfinance API call.", ticker_upper)

    # ── Step 3: Load full range from SQLite ───────────────────────────────────
    df = load_ohlcv(ticker_upper, start=start, end=end)

    if df.empty:
        raise ValueError(
            f"No data available for ticker '{ticker_upper}' "
            f"({start} → {end}). Verify the symbol is correct."
        )

    # ── Step 4: Optional CSV save (for train_models.py backward compat) ───────
    if save:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RAW_DATA_DIR / f"{ticker_upper.replace('.', '_')}.csv"
        df.to_csv(csv_path)
        logger.info("Raw CSV saved: %s", csv_path)

    return df


def get_realtime_quote(ticker: str) -> dict:
    """
    Fetch the latest available market quote for any ticker.

    Uses yf.Ticker.fast_info which is a single lightweight HTTP call —
    much cheaper than a full history download.

    Parameters
    ----------
    ticker : str

    Returns
    -------
    dict
        Keys: last_price, previous_close, day_high, day_low, volume,
              currency, market_cap (when available).
        Returns empty dict on failure.
    """
    try:
        info = yf.Ticker(ticker.upper()).fast_info
        return {
            "last_price":      getattr(info, "last_price",      None),
            "previous_close":  getattr(info, "previous_close",  None),
            "day_high":        getattr(info, "day_high",         None),
            "day_low":         getattr(info, "day_low",          None),
            "volume":          getattr(info, "three_month_average_volume", None),
            "currency":        getattr(info, "currency",         "USD"),
            "market_cap":      getattr(info, "market_cap",       None),
        }
    except Exception as exc:
        logger.warning("fast_info failed for %s: %s", ticker, exc)
        return {}


def download_multiple(
    tickers: list[str],
    start: str = "2018-01-01",
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Download data for multiple tickers using the smart DB-backed downloader.

    Parameters
    ----------
    tickers : list[str]
    start   : str
    end     : str | None

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    results: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            results[ticker] = download_stock_data(ticker, start=start, end=end, save=True)
        except Exception as exc:
            logger.warning("Skipping %s: %s", ticker, exc)
    return results


# Backward-compatible alias used by older code paths
def load_raw_data(ticker: str) -> pd.DataFrame:
    """Load from SQLite (fallback to CSV if DB is empty)."""
    df = load_ohlcv(ticker.upper())
    if not df.empty:
        return df
    # fallback: try the old CSV path
    csv_path = RAW_DATA_DIR / f"{ticker.replace('.', '_')}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)
    raise FileNotFoundError(f"No cached data found for {ticker}.")
