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
import re
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
You are a senior quantitative stock market analyst. Based on the market data below,
write a clear, deeply insightful, professional analysis explaining WHY the stock is predicted to {direction} tomorrow.

## Stock Data Summary
- Ticker: {ticker}
- Prediction Model: {model_name}
- Current Close Price: \\${current_close:.4f}
- Predicted Next Close: \\${predicted_close:.4f} (for {prediction_date})
- Expected Change: {pct_change:+.2f}% (Moving {arrow})

## Technical Indicators (Latest Trading Day)
- Open: \\${features.get('Open', 0):.2f}
- High: \\${features.get('High', 0):.2f}
- Low: \\${features.get('Low', 0):.2f}
- Volume: {int(features.get('Volume', 0)):,} shares
- Previous Close (Lag-1): \\${features.get('Prev_Close', 0):.4f}
- MA5 (5-day moving avg): \\${ma5:.4f}
- MA10 (10-day moving avg): \\${ma10:.4f}
- Moving Average Signal: {ma_signal}{trend_context}

## Critical Formatting Requirements
1. DOLLAR SIGNS: Always write prices with an escaped backslash like \\${current_close:.2f} (e.g., \\$336.13, \\$337.00). NEVER write unescaped dollar signs, because Markdown parsers treat them as LaTeX math delimiters and scramble words.
2. Structure your analysis using the following EXACT markdown section headings:

### Price Trend Analysis
(Explain recent price movement, momentum, and position relative to MA5/MA10 in 2-3 clear sentences)

### Technical Signal Breakdown
(Provide 3 bullet points with bold sub-labels explaining: Moving Average Crossover, Volume & Volatility spread, and Intraday Momentum)

### Prediction Rationale
(Explain the mathematical reasoning why the {model_name} model forecasts a {direction} for {ticker} using 2 numbered points)

### Key Risk Factors
(List 2 realistic technical market factors that could invalidate this prediction in bullet points)

### Summary
(A concise, 1-2 sentence executive takeaway)

Keep the tone professional, institutional, and easy to read. Use actual numbers from the data. Total length: 250-380 words.
""".strip()

    return prompt


def _sanitize_markdown(text: str) -> str:
    """Escape solitary/currency dollar signs to prevent Streamlit KaTeX LaTeX collision."""
    if not text:
        return text
    # Escape dollar signs that are not already preceded by a backslash
    return re.sub(r'(?<!\\)\$', r'\$', text)


def generate_analysis(
    ticker: str,
    model_name: str,
    prediction_result: dict[str, Any],
    latest_features: dict[str, float],
    recent_prices: list[float] | None = None,
    api_key: str | None = None,
    gemini_model_id: str = "gemini-flash-latest",
) -> str:
    """
    Call Google Gemini API to generate a market analysis explanation.

    Uses the new google.genai SDK (google-genai package, not google-generativeai).
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

        # List of models to try in order of preference if primary experiences high demand (503)
        candidate_models = [gemini_model_id]
        fallback_pool = ["gemini-flash-latest", "gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite"]
        for m in fallback_pool:
            if m not in candidate_models:
                candidate_models.append(m)

        last_error = None
        for current_model in candidate_models:
            try:
                logger.info("Calling Gemini API (%s) for %s analysis...", current_model, ticker)
                response = client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                )
                if response and response.text:
                    analysis = response.text.strip()
                    logger.info("Gemini response received from %s (%d chars)", current_model, len(analysis))
                    return _sanitize_markdown(analysis)
            except Exception as exc_call:
                last_error = exc_call
                err_str = str(exc_call)
                # If it's a 503 high demand or 429, try the next model in the fallback pool
                if "503" in err_str or "high demand" in err_str.lower() or "429" in err_str:
                    logger.warning("Model %s busy/unavailable (%s). Trying fallback...", current_model, err_str[:100])
                    continue
                else:
                    # Other fatal errors (e.g. invalid key) should break immediately
                    raise exc_call

        if last_error:
            raise last_error

        return "**No response generated by Gemini AI.**"

    except ImportError:
        # Fallback: try the legacy google.generativeai if google.genai not installed
        try:
            import google.generativeai as genai_legacy  # type: ignore
            genai_legacy.configure(api_key=key)
            model_obj = genai_legacy.GenerativeModel(gemini_model_id)
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
            return _sanitize_markdown(response.text.strip())
        except Exception as exc2:
            return f"**Gemini SDK not found.** Please run: `pip install google-genai`\n\nError: {exc2}"

    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        error_msg = str(exc)

        # User-friendly error messages for common failures
        if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
            return "**Invalid Gemini API key.** Please double-check the key you entered in the sidebar or `.env`."
        if "503" in error_msg or "high demand" in error_msg.lower() or "UNAVAILABLE" in error_msg:
            return (
                "**⚠️ Gemini Service Temporarily Busy (503 High Demand)**\n\n"
                "Google's servers for this model are currently experiencing high traffic spikes. "
                "Your API key is valid.\n\n"
                "**Quick Solutions:**\n"
                "- Try selecting a different model from the **Gemini Model dropdown** in the sidebar (e.g., `gemini-3.5-flash` or `gemini-3.1-flash-lite`).\n"
                "- Click **Predict Next Close** again in a few seconds."
            )
        if "quota" in error_msg.lower() or "429" in error_msg:
            return (
                "**Gemini API rate limit reached (429).**\n\n"
                "The free-tier quota for this model has been reached. "
                "Try selecting an alternative model in the sidebar dropdown or wait a moment."
            )
        if "404" in error_msg or "not found" in error_msg.lower():
            return (
                "**Gemini model not available (404).**\n\n"
                f"Model `{gemini_model_id}` is not available. Please pick another model in the sidebar."
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
            model="gemini-1.5-flash",
            contents="Say OK in one word.",
        )
        return bool(response.text)
    except Exception:
        return False
