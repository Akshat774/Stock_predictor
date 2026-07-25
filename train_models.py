"""
train_models.py
---------------
End-to-end training script.

Usage:
    python train_models.py

What it does:
1. Downloads historical data for a default set of tickers.
2. Preprocesses and engineers features for each ticker.
3. Concatenates all ticker data into one combined dataset
   (so the saved models generalise across multiple stocks).
4. Trains Linear Regression and Random Forest Regressor.
5. Evaluates both models on the combined test split.
6. Saves models to models/ and metrics to models/metrics.json.

Note on multi-ticker strategy:
    Training on multiple tickers produces a generalised model that works
    reasonably well across different stocks without retraining. The
    Streamlit app uses this model for inference on any ticker symbol.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# ── Ensure project root is on PYTHONPATH when running from terminal ──────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.downloader import download_multiple
from src.preprocess import preprocess_pipeline
from src.feature_engineering import add_features, get_feature_matrix
from src.trainer import train_all
from src.evaluator import evaluate_all_models, save_metrics, get_predictions

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
TICKERS: list[str] = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "TSLA",
]
# Indian-market tickers may fail in offline/restricted environments;
# they are excluded from the default list to keep training reliable.
# To add them: TICKERS += ["TCS.NS", "RELIANCE.NS"]

START_DATE: str = "2018-01-01"
END_DATE: str = "2024-12-31"


def main() -> None:
    logger.info("=" * 60)
    logger.info("  Stock Price Predictor — Model Training Script")
    logger.info("=" * 60)

    # ── Step 1: Download ───────────────────────────────────────────────────
    logger.info("Step 1/5  Downloading historical data for: %s", TICKERS)
    raw_data = download_multiple(TICKERS, start=START_DATE, end=END_DATE)

    if not raw_data:
        logger.error("No data downloaded. Check your internet connection and ticker symbols.")
        sys.exit(1)

    successfully_downloaded = list(raw_data.keys())
    logger.info("Successfully downloaded: %s", successfully_downloaded)

    # ── Step 2: Preprocess ─────────────────────────────────────────────────
    logger.info("Step 2/5  Preprocessing and cleaning data...")
    clean_frames: list[pd.DataFrame] = []
    for ticker, raw_df in raw_data.items():
        try:
            clean_df = preprocess_pipeline(raw_df, ticker=ticker, save=True)
            clean_frames.append(clean_df)
        except Exception as exc:
            logger.warning("Skipping %s during preprocessing: %s", ticker, exc)

    if not clean_frames:
        logger.error("No clean data available after preprocessing. Aborting.")
        sys.exit(1)

    # ── Step 3: Feature Engineering ────────────────────────────────────────
    logger.info("Step 3/5  Engineering features...")
    featured_frames: list[pd.DataFrame] = []
    for df in clean_frames:
        try:
            featured = add_features(df)
            featured_frames.append(featured)
        except Exception as exc:
            logger.warning("Feature engineering failed for a ticker: %s", exc)

    if not featured_frames:
        logger.error("Feature engineering produced no usable data. Aborting.")
        sys.exit(1)

    # Concatenate all tickers — sort by date to preserve rough chronological order
    combined_df = pd.concat(featured_frames, axis=0).sort_index()
    logger.info("Combined dataset: %d rows across %d tickers.", len(combined_df), len(featured_frames))

    X, y = get_feature_matrix(combined_df)
    logger.info("Feature matrix: %s  |  Target vector: %s", X.shape, y.shape)

    # ── Step 4: Train ─────────────────────────────────────────────────────
    logger.info("Step 4/5  Training models (chronological 80/20 split)...")
    trained_models, X_test, y_test = train_all(X, y)

    # ── Step 5: Evaluate & Save ───────────────────────────────────────────
    logger.info("Step 5/5  Evaluating models and saving metrics...")
    metrics = evaluate_all_models(trained_models, X_test, y_test)
    save_metrics(metrics)

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 55)
    logger.info("  Training Complete - Results Summary")
    logger.info("=" * 55)
    for model_name, m in metrics.items():
        if isinstance(m, dict):
            logger.info(
                "  %-22s  MAE=%7.4f  RMSE=%7.4f  R2=%6.4f",
                model_name, m["MAE"], m["RMSE"], m["R2"],
            )
    logger.info("")
    logger.info("  Models saved:")
    logger.info("    -> models/linear_regression.pkl")
    logger.info("    -> models/random_forest.pkl")
    logger.info("    -> models/metrics.json")
    logger.info("")
    logger.info("  Run the dashboard with:  streamlit run app.py")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
