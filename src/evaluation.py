"""Model evaluation utilities.
MAPE formulae is: MAPE = (1/n) * sum(|(y_true - y_pred) / y_true|) * 100.
"""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models import TrainedModel

logger = logging.getLogger(__name__)


def mean_absolute_percentage_error_safe(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8
) -> float:
    """Calculate MAPE while avoiding division by zero."""

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    #to avoid division by 0, if the true value is very close to 0 it is replaced with nan.
    denominator = np.where(np.abs(y_true_array) < epsilon, np.nan, np.abs(y_true_array))
    percentage_errors = np.abs((y_true_array - y_pred_array) / denominator)
    if np.isnan(percentage_errors).all():
        return float("nan")
    return float(np.nanmean(percentage_errors) * 100)


def evaluate_predictions(
    trained_models: dict[str, TrainedModel],
    y_test: pd.Series
) -> pd.DataFrame:
    """Evaluate all model predictions with MAE, RMSE, MAPE, and R2."""

    rows: list[dict[str, float | str]] = []
    for name, trained_model in trained_models.items():
        predictions = trained_model.predictions
        rows.append(
            {
                "model": name,
                "MAE": mean_absolute_error(y_test, predictions),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "MAPE": mean_absolute_percentage_error_safe(y_test, predictions),
                "R2": r2_score(y_test, predictions),
            }
        )

    results = pd.DataFrame(rows).sort_values("MAE", ascending=True).reset_index(drop=True)
    return results


def save_results(results: pd.DataFrame, output_path: Path) -> None:
    """Persist benchmark metrics as a CSV report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info("Saved model results to %s", output_path)


def get_best_model_name(results: pd.DataFrame) -> str:
    """Return the model with the lowest MAE."""

    if results.empty:
        raise ValueError("Cannot choose a best model from an empty results table.")
    return str(results.loc[results["MAE"].idxmin(), "model"])
