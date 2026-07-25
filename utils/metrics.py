"""
utils/metrics.py
----------------
Standalone metric calculation helpers.
Used by evaluator.py and any reporting component.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """
    Compute Mean Absolute Error.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        MAE value (lower is better).
    """
    return float(mean_absolute_error(y_true, y_pred))


def compute_rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """
    Compute Root Mean Squared Error.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        RMSE value (lower is better).
    """
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def compute_r2(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """
    Compute R-squared (coefficient of determination).

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        R² value (higher is better; max 1.0).
    """
    return float(r2_score(y_true, y_pred))


def compute_all_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """
    Compute MAE, RMSE, and R² in one call.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    dict[str, float]
        Dictionary with keys 'MAE', 'RMSE', 'R2'.
    """
    return {
        "MAE": compute_mae(y_true, y_pred),
        "RMSE": compute_rmse(y_true, y_pred),
        "R2": compute_r2(y_true, y_pred),
    }


def best_model_by_metric(
    metrics: dict[str, dict[str, float]],
    metric: str = "RMSE",
) -> str:
    """
    Determine the best model name given a metrics comparison dict.

    Parameters
    ----------
    metrics : dict[str, dict[str, float]]
        Dictionary mapping model_name -> metric_dict.
        e.g. {"Linear Regression": {"MAE": 1.2, "RMSE": 1.5, "R2": 0.95}}
    metric : str
        The metric key to compare on. For 'R2', higher is better.
        For 'MAE' and 'RMSE', lower is better.

    Returns
    -------
    str
        Name of the best-performing model.
    """
    if metric == "R2":
        return max(metrics, key=lambda m: metrics[m].get(metric, float("-inf")))
    return min(metrics, key=lambda m: metrics[m].get(metric, float("inf")))
