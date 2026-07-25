"""
src/database.py
---------------
SQLite-backed persistent cache for historical OHLCV stock data.

Design goals:
  - Permanent disk persistence (survives app restarts, unlike st.cache_data).
  - Narrowed API usage: only fetch dates that are NOT already in the DB.
  - Thread-safe via connection-per-call pattern (no shared connection object).
  - Self-initialising: calling get_db_path() or any public function creates
    the schema automatically if the DB file doesn't exist yet.

Database file: data/stock_cache.db
Schema:
    stock_prices(ticker TEXT, date TEXT, open REAL, high REAL,
                 low REAL, close REAL, volume REAL)
    PRIMARY KEY (ticker, date) — prevents duplicate rows on re-insert.
"""

import logging
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH  = DATA_DIR / "stock_cache.db"

# ─── DDL ─────────────────────────────────────────────────────────────────────
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS stock_prices (
    ticker  TEXT    NOT NULL,
    date    TEXT    NOT NULL,   -- ISO-8601 'YYYY-MM-DD'
    open    REAL,
    high    REAL,
    low     REAL,
    close   REAL,
    volume  REAL,
    PRIMARY KEY (ticker, date)
);
"""

_CREATE_META_SQL = """
CREATE TABLE IF NOT EXISTS fetch_log (
    ticker      TEXT    NOT NULL,
    fetched_at  TEXT    NOT NULL,   -- ISO-8601 datetime of last successful API fetch
    PRIMARY KEY (ticker)
);
"""

_CREATE_GAP_ATTEMPTS_SQL = """
CREATE TABLE IF NOT EXISTS gap_attempts (
    ticker     TEXT NOT NULL,
    gap_start  TEXT NOT NULL,   -- ISO-8601 'YYYY-MM-DD'
    gap_end    TEXT NOT NULL,   -- ISO-8601 'YYYY-MM-DD'
    PRIMARY KEY (ticker, gap_start, gap_end)
);
"""


def _get_conn() -> sqlite3.Connection:
    """
    Open (and if needed, initialise) the SQLite database.

    Returns a new connection object.  Each caller must close it after use.
    Using a per-call connection is the safest pattern for Streamlit because
    the framework may run callbacks from multiple threads.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")   # better concurrent read performance
    conn.execute(_CREATE_TABLE_SQL)
    conn.execute(_CREATE_META_SQL)
    conn.execute(_CREATE_GAP_ATTEMPTS_SQL)
    conn.commit()
    return conn


# ─── Public API ───────────────────────────────────────────────────────────────

def upsert_ohlcv(ticker: str, df: pd.DataFrame) -> int:
    """
    Insert or replace OHLCV rows for a ticker into the database.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    df : pd.DataFrame
        DataFrame with DatetimeIndex and columns Open, High, Low, Close, Volume.

    Returns
    -------
    int
        Number of rows written.
    """
    if df.empty:
        return 0

    records = [
        (
            ticker.upper(),
            idx.strftime("%Y-%m-%d"),
            float(row.get("Open",   0)),
            float(row.get("High",   0)),
            float(row.get("Low",    0)),
            float(row.get("Close",  0)),
            float(row.get("Volume", 0)),
        )
        for idx, row in df.iterrows()
    ]

    sql = """
        INSERT OR REPLACE INTO stock_prices
            (ticker, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    conn = _get_conn()
    try:
        conn.executemany(sql, records)
        # Update fetch log
        conn.execute(
            "INSERT OR REPLACE INTO fetch_log (ticker, fetched_at) VALUES (?, ?)",
            (ticker.upper(), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        logger.info("DB upsert: %d rows for %s", len(records), ticker)
    finally:
        conn.close()

    return len(records)


def load_ohlcv(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Load cached OHLCV data for a ticker from the database.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    start : str | None
        ISO date string 'YYYY-MM-DD'. If None, fetches all available.
    end : str | None
        ISO date string. If None, fetches up to today.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and columns Open, High, Low, Close, Volume.
        Returns an empty DataFrame if no cached rows exist.
    """
    conditions = ["ticker = ?"]
    params: list = [ticker.upper()]

    if start:
        conditions.append("date >= ?")
        params.append(start)
    if end:
        conditions.append("date <= ?")
        params.append(end)

    where = " AND ".join(conditions)
    sql = f"""
        SELECT date, open, high, low, close, volume
        FROM stock_prices
        WHERE {where}
        ORDER BY date ASC
    """

    conn = _get_conn()
    try:
        df = pd.read_sql_query(sql, conn, params=params, parse_dates=["date"])
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    df = df.set_index("date")
    df.index.name = None
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df


def _is_gap_attempted(ticker: str, gap_start: str, gap_end: str) -> bool:
    """Return True if this date gap was already fetched and returned no data."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            """
            SELECT 1 FROM gap_attempts
            WHERE ticker = ? AND gap_start = ? AND gap_end = ?
            """,
            (ticker.upper(), gap_start, gap_end),
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


def mark_gap_attempted(ticker: str, gap_start: str, gap_end: str) -> None:
    """
    Record that a date gap was fetched from yfinance but returned no rows.

    Prevents repeated API calls for holidays/weekends that have no market data.
    """
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO gap_attempts (ticker, gap_start, gap_end)
            VALUES (?, ?, ?)
            """,
            (ticker.upper(), gap_start, gap_end),
        )
        conn.commit()
    finally:
        conn.close()


def get_missing_date_ranges(
    ticker: str,
    requested_start: str,
    requested_end: str,
) -> list[tuple[str, str]]:
    """
    Smarter gap detector that queries SQLite min/max dates.
    This prevents continuous yfinance requests for known market holidays
    by checking if the requested range falls inside the already-fetched range.

    Parameters
    ----------
    ticker : str
    requested_start : str   'YYYY-MM-DD'
    requested_end   : str   'YYYY-MM-DD'

    Returns
    -------
    list[tuple[str, str]]
        List of (gap_start, gap_end) strings representing missing ranges.
    """
    ticker_upper = ticker.upper()
    conn = _get_conn()
    try:
        cur = conn.execute(
            "SELECT MIN(date), MAX(date) FROM stock_prices WHERE ticker = ?",
            (ticker_upper,)
        )
        db_min, db_max = cur.fetchone()
    finally:
        conn.close()

    if not db_min or not db_max:
        # Cache is completely empty, download whole range
        return [(requested_start, requested_end)]

    ranges: list[tuple[str, str]] = []

    req_start_dt = datetime.strptime(requested_start, "%Y-%m-%d")
    req_end_dt   = datetime.strptime(requested_end,   "%Y-%m-%d")
    db_min_dt    = datetime.strptime(db_min,          "%Y-%m-%d")
    db_max_dt    = datetime.strptime(db_max,          "%Y-%m-%d")

    # If requested start is before what we have cached, fetch the prefix
    if req_start_dt < db_min_dt:
        prefix_end_dt = db_min_dt - timedelta(days=1)
        if req_start_dt <= prefix_end_dt:
            prefix_bdays = pd.bdate_range(start=req_start_dt, end=prefix_end_dt)
            if not prefix_bdays.empty:
                gap = (requested_start, prefix_end_dt.strftime("%Y-%m-%d"))
                if not _is_gap_attempted(ticker_upper, *gap):
                    ranges.append(gap)

    # If requested end is after what we have cached, fetch the suffix
    # Note: If requested end is weekend/holiday, db_max might be the last business day.
    # We check if there are actual business days between db_max and requested_end.
    if req_end_dt > db_max_dt:
        all_bdays_suffix = pd.bdate_range(start=db_max_dt + timedelta(days=1), end=req_end_dt)
        if not all_bdays_suffix.empty:
            suffix_start = (db_max_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            gap = (suffix_start, requested_end)
            if not _is_gap_attempted(ticker_upper, *gap):
                ranges.append(gap)

    return ranges



def cached_days_available(ticker: str) -> int:
    """Return number of distinct trading days cached for a ticker."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM stock_prices WHERE ticker = ?",
            (ticker.upper(),),
        )
        return cur.fetchone()[0]
    finally:
        conn.close()


def last_fetched_at(ticker: str) -> str | None:
    """Return ISO datetime string of the last successful API fetch for a ticker."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            "SELECT fetched_at FROM fetch_log WHERE ticker = ?",
            (ticker.upper(),),
        )
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def get_all_cached_tickers() -> list[str]:
    """Return all distinct ticker symbols present in the database."""
    conn = _get_conn()
    try:
        cur = conn.execute("SELECT DISTINCT ticker FROM stock_prices ORDER BY ticker")
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()
