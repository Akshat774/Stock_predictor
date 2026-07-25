"""
src/llm_analyzer.py
-------------------
Google Gemini AI integration for stock market analysis.

Provides a single public function: generate_analysis()
  - Takes prediction result + technical features as context.
  - Builds a structured, data-rich prompt.
  - Calls Gemini API via the new google.genai SDK (google-genai package).
  - Returns a formatted markdown string with the AI's reasoning.

API Key:
  Set the environment variable GEMINI_API_KEY before running the app, or
  enter it in the Streamlit sidebar when prompted.

Rate limiting strategy:
  - Results are cached in st.session_state keyed by (ticker, model, date)
    so one analysis is generated per prediction, not per re-render.

Model:
  Uses gemini-2.0-flash (standard free-tier quota model).
  Falls back gracefully if the model is unavailable.
"""

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Load local .env variables manually to remain zero-dependency on dotenv
def load_env_file():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'").strip('"')
                        os.environ[k] = v
        except Exception as exc:
            logger.warning("Could not read .env file: %s", exc)


# Call load_env_file on module load
load_env_file()

# Model to use — gemini-2.0-flash is the recommended free-tier model
# that supports the new google.genai SDK
GEMINI_MODEL = "gemini-2.0-flash"


def _build_prompt(
    ticker: str,
    model_name: str,
    current_close: float,
    predicted_close: float,
    pct_change: float,
    prediction_date: str,
    features: dict[str, float],
    recent_prices: list[float] | None = None,
) -> str:
    """
    Build a structured, data-rich prompt for Gemini.

    Parameters
    ----------
    ticker           : Stock ticker symbol
    model_name       : 'Linear Regression' or 'Random Forest'
    current_close    : Most recent closing price
    predicted_close  : ML model's next-day prediction
    pct_change       : Percentage change (positive = up, negative = down)
    prediction_date  : Estimated date of prediction (ISO string)
    features         : Latest feature row dict (Open, High, Low, Volume, MA5, MA10, Prev_Close)
    recent_prices    : List of last N closing prices for trend context

    Returns
    -------
    str
        Formatted prompt string.
    """
    direction = "RISE" if pct_change >= 0 else "FALL"
    arrow = "UP" if pct_change >= 0 else "DOWN"
    trend_context = ""
    if recent_prices and len(recent_prices) >= 3:
        trend_str = " -> ".join([f"${p:.2f}" for p in recent_prices[-5:]])
        trend_context = f"\n  Recent closing prices (oldest to latest): {trend_str}"

    ma5  = features.get("MA5",  0)
    ma10 = features.get("MA10", 0)
    ma_signal = (
        "Bullish crossover (MA5 > MA10, short-term momentum up)"
        if ma5 > ma10 else
        "Bearish signal (MA5 < MA10, short-term momentum down)"
    )

    prompt = f"""
You are a professional quantitative stock market analyst. Based on the data below,
write a clear, insightful analysis explaining WHY the stock is predicted to {direction} tomorrow.

## Stock Data Summary
- Ticker: {ticker}
- Prediction Model: {model_name}
- Current Close Price: ${current_close:.4f}
- Predicted Next Close: ${predicted_close:.4f} (for {prediction_date})
- Expected Change: {pct_change:+.2f}% (Moving {arrow})

## Technical Indicators (Latest Trading Day)
- Open: ${features.get('Open', 0):.2f}
- High: ${features.get('High', 0):.2f}
- Low: ${features.get('Low', 0):.2f}
- Volume: {int(features.get('Volume', 0)):,} shares
- Previous Close (Lag-1): ${features.get('Prev_Close', 0):.4f}
- MA5 (5-day moving avg): ${ma5:.4f}
- MA10 (10-day moving avg): ${ma10:.4f}
- Moving Average Signal: {ma_signal}{trend_context}

## Instructions
Write a structured analysis with the following sections in MARKDOWN format:

### Price Trend Analysis
(Explain the recent price movement pattern based on current close, MA5, MA10, and recent prices)

### Technical Signal Breakdown
(Explain what each indicator tells us: MA crossover, volume, day range High-Low spread, momentum)

### Prediction Rationale
(Explain specifically why the {model_name} model is predicting a {direction} for {ticker})

### Key Risk Factors
(Mention 2-3 factors that could invalidate this prediction - keep it realistic and honest)

### Summary
(One or two sentence confident conclusion)

Keep the tone professional but accessible. Use actual numbers from the data above.
Do NOT include generic disclaimers about stock prediction being uncertain.
Total response should be 250-400 words.
""".strip()

    return prompt


def generate_analysis(
    ticker: str,
    model_name: str,
    prediction_result: dict[str, Any],
    latest_features: dict[str, float],
    recent_prices: list[float] | None = None,
    api_key: str | None = None,
) -> str:
    """
    Call Google Gemini API to generate a market analysis explanation.

    Uses the new google.genai SDK (google-genai package, not google-generativeai).

    Parameters
    ----------
    ticker             : Stock ticker symbol
    model_name         : Name of the ML model used
    prediction_result  : Dict returned by predict_next_close()
    latest_features    : Dict of latest feature values {Open, High, Low, ...}
    recent_prices      : Optional list of recent Close prices for trend context
    api_key            : Gemini API key. Falls back to GEMINI_API_KEY env var.

    Returns
    -------
    str
        Markdown-formatted AI analysis, or an error message string.
    """
    # Resolve API key
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return (
            "**Gemini API key not set.**\n\n"
            "Enter your API key in the sidebar to enable AI analysis. "
            "Get a free key at [aistudio.google.com](https://aistudio.google.com/app/apikey)."
        )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)

        prompt = _build_prompt(
            ticker=ticker,
            model_name=model_name,
            current_close=prediction_result["current_close"],
            predicted_close=prediction_result["predicted_close"],
            pct_change=prediction_result["pct_change"],
            prediction_date=prediction_result["prediction_date"],
            features=latest_features,
            recent_prices=recent_prices,
        )

        logger.info("Calling Gemini API (%s) for %s analysis...", GEMINI_MODEL, ticker)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        analysis = response.text.strip() if response.text else ""
        logger.info("Gemini response received (%d chars)", len(analysis))
        return analysis

    except ImportError:
        # Fallback: try the legacy google.generativeai if google.genai not installed
        try:
            import google.generativeai as genai_legacy  # type: ignore
            genai_legacy.configure(api_key=key)
            model_obj = genai_legacy.GenerativeModel("gemini-1.5-flash")
            prompt = _build_prompt(
                ticker=ticker,
                model_name=model_name,
                current_close=prediction_result["current_close"],
                predicted_close=prediction_result["predicted_close"],
                pct_change=prediction_result["pct_change"],
                prediction_date=prediction_result["prediction_date"],
                features=latest_features,
                recent_prices=recent_prices,
            )
            response = model_obj.generate_content(prompt)
            return response.text.strip()
        except Exception as exc2:
            return f"**Gemini SDK not found.** Please run: `pip install google-genai`\n\nError: {exc2}"

    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        error_msg = str(exc)

        # User-friendly error messages for common failures
        if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
            return "**Invalid Gemini API key.** Please double-check the key you entered."
        if "quota" in error_msg.lower() or "429" in error_msg:
            return (
                "**Gemini API rate limit reached.**\n\n"
                "The free-tier daily quota has been exceeded. "
                "Please wait until tomorrow or upgrade your plan at "
                "[aistudio.google.com](https://aistudio.google.com/app/apikey).\n\n"
                f"*Technical detail: {error_msg[:200]}*"
            )
        if "404" in error_msg or "not found" in error_msg.lower():
            return (
                "**Gemini model not available.**\n\n"
                f"Model `{GEMINI_MODEL}` may not be available on your API key's region. "
                "Try a different model or check your API key permissions."
            )
        return f"**Gemini API error:** {error_msg}"


def validate_api_key(api_key: str) -> bool:
    """
    Quick validation — attempts a minimal Gemini API call.

    Parameters
    ----------
    api_key : str

    Returns
    -------
    bool
        True if the key is valid and the API is reachable.
    """
    if not api_key or len(api_key) < 10:
        return False
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Say OK in one word.",
        )
        return bool(response.text)
    except Exception:
        return False
