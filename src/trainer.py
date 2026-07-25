"""
trainer.py
----------
Splits data chronologically and trains Linear Regression and
Random Forest Regressor models. Models are serialized to disk
using joblib.

Key design decisions:
- NO random shuffle for time series (avoids data leakage).
- An 80/20 chronological split is used (first 80% = train, last 20% = test).
- scikit-learn Pipelines with StandardScaler are used for Linear Regression;
  Random Forest does not require scaling but is included in a Pipeline for
  consistency.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LR_MODEL_PATH = MODELS_DIR / "linear_regression.pkl"
RF_MODEL_PATH = MODELS_DIR / "random_forest.pkl"

TRAIN_RATIO: float = 0.80  # 80% train, 20% test
RANDOM_STATE: int = 42     # Reproducibility seed for Random Forest


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into train/test sets preserving chronological order.

    NO shuffling is performed — this is critical for time-series data to
    prevent future-data leakage into the training set.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    train_ratio : float
        Fraction of data to use for training (default 0.80).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(
        "Chronological split: train=%d rows | test=%d rows (ratio=%.0f%%)",
        len(X_train),
        len(X_test),
        train_ratio * 100,
    )
    return X_train, X_test, y_train, y_test


def build_linear_regression_pipeline() -> Pipeline:
    """
    Build a scikit-learn Pipeline for Linear Regression.

    Includes StandardScaler before the regressor because Linear Regression
    benefits from scaled features (affects interpretation and numerical stability).

    Returns
    -------
    Pipeline
        Unfitted sklearn Pipeline.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])


def build_random_forest_pipeline(n_estimators: int = 200) -> Pipeline:
    """
    Build a scikit-learn Pipeline for Random Forest Regressor.

    Random Forest is invariant to feature scale, but we wrap it in a
    Pipeline for consistent API usage.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest. Default 200 for good accuracy
        while remaining training-time reasonable.

    Returns
    -------
    Pipeline
        Unfitted sklearn Pipeline.
    """
    return Pipeline([
        # PassThrough step — RF doesn't need scaling; keeps the pipeline API uniform.
        ("regressor", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None,       # Allow trees to grow fully for best accuracy
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,            # Use all available CPU cores
        )),
    ])


def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
) -> Pipeline:
    """
    Fit a pipeline on training data.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted sklearn Pipeline.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training targets.
    model_name : str
        Label for logging.

    Returns
    -------
    Pipeline
        Fitted sklearn Pipeline.
    """
    logger.info("Training %s on %d samples...", model_name, len(X_train))
    pipeline.fit(X_train, y_train)
    logger.info("Finished training %s.", model_name)
    return pipeline


def save_model(model: Pipeline, path: Path, model_name: str) -> None:
    """
    Serialize a fitted model to disk using joblib.

    Parameters
    ----------
    model : Pipeline
        Fitted sklearn Pipeline.
    path : Path
        Destination .pkl file path.
    model_name : str
        Label for logging.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved %s -> %s", model_name, path)


def load_model(path: Path) -> Pipeline:
    """
    Load a serialized sklearn Pipeline from disk.

    Parameters
    ----------
    path : Path
        Path to .pkl file.

    Returns
    -------
    Pipeline
        Loaded sklearn Pipeline.

    Raises
    ------
    FileNotFoundError
        If the .pkl file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. "
            "Please run train_models.py first to generate the .pkl files."
        )
    model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return model


def train_all(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[dict[str, Pipeline], pd.DataFrame, pd.Series]:
    """
    Run the full training pipeline for both models.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix (all rows, chronologically ordered).
    y : pd.Series
        Full target vector.

    Returns
    -------
    tuple[dict[str, Pipeline], pd.DataFrame, pd.Series]
        (trained_models_dict, X_test, y_test)
        where trained_models_dict maps model_name -> fitted Pipeline.
    """
    X_train, X_test, y_train, y_test = chronological_split(X, y)

    # ── Linear Regression ──
    lr_pipeline = build_linear_regression_pipeline()
    lr_fitted = train_model(lr_pipeline, X_train, y_train, "Linear Regression")
    save_model(lr_fitted, LR_MODEL_PATH, "Linear Regression")

    # ── Random Forest ──
    rf_pipeline = build_random_forest_pipeline()
    rf_fitted = train_model(rf_pipeline, X_train, y_train, "Random Forest")
    save_model(rf_fitted, RF_MODEL_PATH, "Random Forest")

    trained_models = {
        "Linear Regression": lr_fitted,
        "Random Forest": rf_fitted,
    }

    return trained_models, X_test, y_test
