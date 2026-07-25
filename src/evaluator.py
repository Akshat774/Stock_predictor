"""
evaluator.py
------------
Evaluates trained models on the test split and persists results to
models/metrics.json.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.metrics import compute_all_metrics, best_model_by_metric

logger = logging.getLogger(__name__)

METRICS_PATH = Path(__file__).resolve().parent.parent / "models" / "metrics.json"


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict[str, float]:
    """
    Run inference on the test set and compute MAE, RMSE, R².

    Parameters
    ----------
    model : sklearn estimator
        A fitted scikit-learn model with a .predict() method.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Ground-truth target values.
    model_name : str
        Human-readable model name for logging.

    Returns
    -------
    dict[str, float]
        Metric dictionary with keys 'MAE', 'RMSE', 'R2'.
    """
    y_pred = model.predict(X_test)
    metrics = compute_all_metrics(y_test, y_pred)

    logger.info(
        "[%s] MAE=%.4f  RMSE=%.4f  R²=%.4f",
        model_name,
        metrics["MAE"],
        metrics["RMSE"],
        metrics["R2"],
    )
    return metrics


def evaluate_all_models(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict[str, float]]:
    """
    Evaluate multiple models and return a comparison dictionary.

    Parameters
    ----------
    models : dict[str, Any]
        Mapping of model_name -> fitted sklearn estimator.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Ground-truth target vector.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict: {model_name: {metric: value}}.
    """
    results: dict[str, dict[str, float]] = {}
    for model_name, model in models.items():
        results[model_name] = evaluate_model(model, X_test, y_test, model_name)

    best = best_model_by_metric(results, metric="RMSE")
    logger.info("Best model by RMSE: %s", best)
    return results


def get_predictions(
    model: Any,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Return raw prediction array from a fitted model.

    Parameters
    ----------
    model : sklearn estimator
    X_test : pd.DataFrame

    Returns
    -------
    np.ndarray
    """
    return model.predict(X_test)


def save_metrics(metrics: dict[str, dict[str, float]]) -> None:
    """
    Persist evaluation metrics to models/metrics.json.

    Adds a 'best_model_by_rmse' and 'best_model_by_r2' key for convenience.

    Parameters
    ----------
    metrics : dict[str, dict[str, float]]
        Nested metrics dict as returned by evaluate_all_models().
    """
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = dict(metrics)  # shallow copy
    output["best_model_by_rmse"] = best_model_by_metric(metrics, metric="RMSE")
    output["best_model_by_r2"] = best_model_by_metric(metrics, metric="R2")

    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=4)

    logger.info("Metrics saved to %s", METRICS_PATH)


def load_metrics() -> dict[str, Any]:
    """
    Load evaluation metrics from models/metrics.json.

    Returns
    -------
    dict
        Loaded metrics dictionary.

    Raises
    ------
    FileNotFoundError
        If metrics.json has not been generated yet.
    """
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"metrics.json not found at {METRICS_PATH}. "
            "Please run train_models.py first."
        )
    with open(METRICS_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info("Loaded metrics from %s", METRICS_PATH)
    return data
